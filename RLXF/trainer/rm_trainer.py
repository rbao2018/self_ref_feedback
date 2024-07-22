import os.path as osp
from abc import ABC

import torch
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from openrlhf.models import LogExpLoss, PairWiseLoss

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

class LogExpWithLMLoss(nn.Module):
    """
    Pairwise Loss and Language Model Loss for Reward Model
    """
    
    def __init__(self, use_margin=True, lm_loss_coef=1e-6):
        super().__init__()
        self.use_margin = use_margin
        self.lm_loss_coef = lm_loss_coef

    def forward(
        self, 
        chosen_reward: torch.Tensor, 
        reject_reward: torch.Tensor,
        chosen_labels: torch.Tensor,
        logits: torch.Tensor,
        margin: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        if margin is not None and self.use_margin:
            rm_loss = torch.log(1 + torch.exp(margin + reject_reward - chosen_reward)).mean()
        else:
            rm_loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        loss = rm_loss
        if self.lm_loss_coef > 1e-6:
            shift_logits = logits[:chosen_labels.size(0), :chosen_labels.size(1) - 1 , :]
            shift_labels = chosen_labels[..., 1:]
            shift_logits = shift_logits.contiguous().view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.contiguous().view(-1).to(shift_logits.device)
            lm_loss = F.cross_entropy(shift_logits, shift_labels)
            loss += self.lm_loss_coef * lm_loss
        return loss
    
class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model written by baorong in 2024-02-16.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        max_epochs (int, defaults to 1): the number of epochs to train
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 1,
        loss="sigmoid",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        elif loss == "logexpwithlm":
            self.loss_fn = LogExpWithLMLoss(
                use_margin=self.args.use_margin, 
                lm_loss_coef=self.args.lm_loss_coef
            )
            self.strategy.print(f"LogExp With LM Loss with margin {self.args.use_margin} and lm_loss_coef {self.args.lm_loss_coef}")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter
            logdir = f"{self.args.logging_path}/tensorboard/"
            self._wandb = SummaryWriter(log_dir=logdir)

    def train(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            acc_mean = 0
            loss_mean = 0
            for batch in self.train_dataloader:
                batch = to_device(batch, torch.cuda.current_device())
                outputs = self.concatenated_forward(self.model, batch)
                outputs.update(batch)
                chosen_reward, reject_reward = outputs["chosen_reward"], outputs["reject_reward"]
                preference_loss = self.loss_fn(**outputs)
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc_mean = acc_mean * 0.9 + 0.1 * (chosen_reward > reject_reward).float().mean().item()
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                # optional rm info
                logs_dict = {
                    "preference_loss": preference_loss.item(),
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "acc_mean": acc_mean,
                    "loss_mean": loss_mean,
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)
                step_bar.update()
                global_step += 1
            epoch_bar.update()
            save_path = args.save_path + f"/epoch{epoch}"
            self.strategy.save_model(self.model, self.tokenizer, save_path)

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.close()

    # logs/checkpoints/evaluate
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
            # self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                osp.join(args.save_path, f"actor_{tag}"),
            )

    @torch.inference_mode()
    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        
        acc_mean = 0
        chosen_reward_mean = 0
        reject_reward_mean = 0
        loss_sum = 0
        for i, batch in enumerate(eval_dataloader):
            if i < 3:
                self.strategy.print(self.tokenizer.batch_decode(batch["concat_ids"]))
            batch = to_device(batch, torch.cuda.current_device())
            outputs = self.concatenated_forward(self.model, batch)
            outputs.update(batch)
            loss = self.loss_fn(**outputs)
            chosen_reward_mean += self.strategy.all_reduce(torch.mean(outputs["chosen_reward"]))
            reject_reward_mean += self.strategy.all_reduce(torch.mean(outputs["reject_reward"]))
            chosen_reward, reject_reward = outputs["chosen_reward"], outputs["reject_reward"]
            acc_mean += self.strategy.all_reduce((chosen_reward > reject_reward).float().mean())
            loss_sum += self.strategy.all_reduce(loss)
            bar_dict = {
                "eval_loss": loss_sum.item(),
                "acc_mean": acc_mean.item(),
                "chosen_reward_mean": chosen_reward_mean.item(),
                "reject_reward_mean": reject_reward_mean.item(),
            }
            step_bar.set_postfix(bar_dict)
            step_bar.update()

        acc_mean = acc_mean / self.eval_dataloader.__len__()
        loss_mean = loss_sum / self.eval_dataloader.__len__()
        chosen_reward_mean /= self.eval_dataloader.__len__()
        reject_reward_mean /= self.eval_dataloader.__len__()

        # save mean std
        self.strategy.print("Set reward mean std")

        logs = {
            "eval_loss": loss_mean.float(),
            "acc_mean": acc_mean.float(),
            "chosen_reward_mean": chosen_reward_mean.float(),
            "reject_reward_mean": reject_reward_mean.float(),
        }
        step_bar.set_postfix(logs)
        self.strategy.print(logs)

        if self._wandb is not None and self.strategy.is_rank_0():
            for k, v in logs.items():
                self._wandb.add_scalar(f"eval/{k}", v, global_step=steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, batch):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = batch["concat_ids"], batch["concat_masks"]
        half_batch_size = att_masks.shape[0] // 2
        all_values, logits = model(input_ids, 
                                   attention_mask=att_masks, 
                                   return_output=True, 
                                   packing_samples=self.args.packing_samples)
        chosen_reward = all_values[: half_batch_size]
        reject_reward = all_values[half_batch_size:]
        if batch.get("extras", None) is not None:
            margin = torch.tensor(batch["extras"], dtype=chosen_reward.dtype, device=chosen_reward.device)

        return {
            "chosen_reward": chosen_reward,
            "reject_reward": reject_reward,
            "logits": logits,
            "margin": margin
        }
