import os
import itertools
import math
import socket
import torch.distributed
from tqdm import tqdm
from typing import Callable, Dict, List

import ray
import torch
import gc
from torch.utils.data import DistributedSampler
from transformers.trainer import get_scheduler
from openrlhf.datasets import SFTDataset
from openrlhf.utils.distributed_util import init_process_group


from .launcher import BasePPORole
from .utils import blending_datasets, get_tokenizer
from .dataset.prompt_dataset import PromptDataset
from .trainer.ppo_trainer import PPOTrainer
from .experience_maker import Experience, RemoteExperienceMaker
from .replay_buffer import NaiveReplayBuffer
from .fsdp_strategy import FSDPStrategy
from .model.actor_model import get_actor_model


class ActorPPOTrainer(PPOTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        critic_train_remote: bool = False,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.critic_train_remote = critic_train_remote
        self.experience_maker = RemoteExperienceMaker(
            self.actor,
            self.critic,
            self.reward_model,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.reward_fn,
            vllm_engines=self.vllm_engines,
        )
        self.replay_buffer = NaiveReplayBuffer(self.args)
        # Create torch group with deepspeed rank 0 and all vllm ranks
        # to update vllm engine's weights after each training stage.
        #
        # Say we have 3 vllm engines and eache of them has 4 GPUs,
        # then the torch group is:
        # [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
        # |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines, vllm_tensor_parallel_size = (
                self.strategy.args.vllm_num_engines,
                self.strategy.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend="nccl",
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            self._model_update_group = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
            )
            ray.get(refs)
        torch.distributed.barrier()

    def training_step(self, experience: Experience) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def fit(
        self,
        prompts_dataloader,
        pretrain_dataloader,
        args,
    ) -> None:
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = prompts_dataloader.__len__() // update_timesteps
            # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 0

        for episode in range(args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )
            critic_status_ref = None
            global_step = 1 + math.ceil(global_step / update_timesteps) * update_timesteps
            self.experience_maker.clear()
            for input_prompts in self.prompts_dataloader:
                
                self.experience_maker.add_requests(input_prompts)

                if global_step % update_timesteps == 0:
                    self._broadcast_to_vllm()
                    torch.cuda.empty_cache()
                    rank = self.strategy.get_rank()
                    total_group = self.strategy.world_size // len(self.vllm_engines)
                    vllm_outputs = self.experience_maker.generate_vllm(
                        rank = rank,
                        total_group = total_group,
                        **self.generate_kwargs
                    )
                    generate_text = self.tokenizer.decode(
                        vllm_outputs["sequences"][0],
                        skip_special_tokens=(global_step > 3 * update_timesteps),
                    )
                    self.strategy.print(generate_text)
                    # wait critic model training done for optimize pipeline
                    if critic_status_ref is not None:
                        critic_status = ray.get(critic_status_ref)
                    else:
                        critic_status = {}
                    torch.distributed.barrier()

                    self.experience_maker.make_experience(
                        self.replay_buffer,
                        **vllm_outputs,
                        batch_size=self.micro_rollout_batch_size,
                        **self.generate_kwargs
                    )

                    self.replay_buffer.normalize("advantages", self.strategy)
                    self.experience_maker.flush()
                    torch.cuda.empty_cache()
                    if self.critic_train_remote:
                        critic_status_ref = self.critic.fit.remote()

                    # update kl control
                    status = self.ppo_train()
                    self.replay_buffer.clear()
                    status.update(critic_status)
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    
                    # logs/checkpoints
                    self.save_logs_and_checkpoints(
                        args, global_step // update_timesteps, pbar, status
                    )
                pbar.update()
                global_step = global_step + 1
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                os.path.join(args.save_path, f"actor_epoch{episode}"),
            )
            torch.distributed.barrier()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.close()

    def _broadcast_to_vllm(self):
        model_state_dict = self.strategy.get_full_msd(self.actor)
        if self.strategy.get_rank() == 0:
            count, num_params = 0, len(list(model_state_dict.keys()))
            for name, param in model_state_dict.items():
                count += 1  # empty_cache at last param
                # Fire all vllm engines for broadcast
                shape = param.shape
                # refs = []
                for engine in self.vllm_engines:
                    # refs.append()
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache = count ==num_params)
                device_data = param.data.to("cuda")
                torch.distributed.broadcast(device_data, 0, group=self._model_update_group)
                del device_data
                # ray.get(refs)
                if count % 8 == 0: torch.cuda.empty_cache()
        del model_state_dict
        gc.collect()
        torch.cuda.empty_cache()


@ray.remote
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: FSDPStrategy, pretrain):
        self.strategy = strategy
        strategy.setup_distributed()
        
        model = get_actor_model(pretrain, "actor", strategy.args)

        strategy.print(model)

        self.actor = strategy.prepare_model(model)
        if strategy.args.enable_ema:
            ema_model = self.actor
        else:
            ema_model = None
        self.actor_optim = strategy.create_optimizer(
            self.actor,
            lr = strategy.args.actor_learning_rate,
            beta = (0.9, 0.95),
            weight_decay = strategy.args.l2
        )
        # configure tokenizer
        self.tokenizer = get_tokenizer(pretrain, model)

        args = strategy.args

        self.prepare_datasets()

        # configure scheduler
        num_update_steps_per_episodes = (
            int(len(self.prompts_dataloader) * (args.micro_rollout_batch_size / strategy.micro_train_batch_size))
            * args.max_epochs // strategy.accumulated_gradient
        )
        self.max_step = math.ceil(args.num_episodes * num_update_steps_per_episodes)

        self.actor_scheduler = get_scheduler(
            "cosine",
            self.actor_optim,
            num_warmup_steps=math.ceil(self.max_step * 0.03),
            num_training_steps=self.max_step,
        )
        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare_model(ema_model)
        else:
            self.ema_model = None

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        prompts_dataset = PromptDataset(prompts_data, self.tokenizer, 
                                        strategy, input_template=args.input_template)
        self.prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset, args.micro_rollout_batch_size, 
            pin_memory=True, 
            collate_fn=prompts_dataset.collate_fn,
            shuffle=True
        )

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        self.strategy.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=self.strategy.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            buffer_limit=args.buffer_limit,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        trainer.fit(self.prompts_dataloader, self.pretrain_dataloader, args)

    def max_steps(self):
        """Return the maximum number of steps."""
        return self.max_step

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            os.path.join(args.save_path, f"actor_latest")
        )
