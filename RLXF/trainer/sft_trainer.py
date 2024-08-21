import os.path as osp
from abc import ABC

import torch
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import GPTLMLoss


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = GPTLMLoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb setting
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter
            logdir = f"{self.args.logging_path}/tensorboard/"
            self._wandb = SummaryWriter(log_dir=logdir)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            loss_mean = 0
            for inputs, labels, attention_masks in self.train_dataloader:
                if global_step < 3:
                    self.strategy.print(self.tokenizer.batch_decode(inputs, skip_special_tokens=False))
                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                labels = labels.to(torch.cuda.current_device())
                _, output = self.model(inputs, attention_mask=attention_mask, 
                                               return_output=True,
                                               packing_samples=self.args.packing_samples)
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                # loss function
                gpt_loss = self.loss_fn(output["logits"], labels)
                loss = gpt_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * gpt_loss.item()
                logs_dict = {"gpt_loss": gpt_loss.item(), "loss_mean": loss_mean}
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # logs/checkpoints/evaluation
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1

            epoch_bar.update()
            save_path = args.save_path + f"/epoch{epoch+1}"
            self.strategy.save_model(self.model, self.tokenizer, save_path)

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._wandb.add_scalar(f"train/{k}", v, global_step=global_step)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_model(
                self.model.model,
                self.tokenizer,
                osp.join(args.save_path, f"sft_{tag}"),
            )

    @torch.no_grad()
    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        loss_sum = 0
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )

        for inputs, labels, attention_masks in eval_dataloader:
            inputs = inputs.to(torch.cuda.current_device())
            attention_mask = attention_masks.to(torch.cuda.current_device())
            labels = labels.to(torch.cuda.current_device())
            _, output = self.model(inputs, attention_mask=attention_mask, 
                                return_output=True,
                                packing_samples=self.args.packing_samples)
            # loss function
            loss = self.loss_fn(output["logits"], labels)

            times += 1
            loss_sum += loss.item()
            bar_dict = {"eval gpt_loss": loss_sum / times}
            step_bar.update()
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

        if self._wandb is not None and self.strategy.is_rank_0():
            for k, v in logs.items():
                self._wandb.add_scalar(f"train/{k}", v, global_step=steps)
        self.model.train()  # reset model state
