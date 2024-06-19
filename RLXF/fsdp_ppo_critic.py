import math
from typing import Dict, Optional

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from .launcher import BasePPORole
from .model.reward_model import get_llm_for_sequence_regression
from .trainer.ppo_trainer import PPOTrainer
from .experience_maker import Experience
from .fsdp_strategy import FSDPStrategy

class CriticPPOTrainer(PPOTrainer):
    def ppo_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=True# not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                # for DP
                status = self.strategy.all_reduce(status)

                status_list.append(status)
                pbar.set_postfix(status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        return self.training_step_critic(experience)


@ray.remote
class CriticModelRayActor(BasePPORole):
    def init_model_from_pretrained(
            self, 
            strategy: FSDPStrategy, 
            pretrain: str, 
            max_steps: int
        ):
        self.strategy = strategy
        strategy.setup_distributed()
        model = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            bf16=strategy.args.bf16,
            global_rank=strategy.get_rank(),
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            use_flash_attention_2=strategy.args.flash_attn
        )
        strategy.print(model)
        self.critic = strategy.prepare_model(model)
        self.critic_optim = strategy.create_optimizer(
            self.critic,
            lr = strategy.args.critic_learning_rate,
            beta = (0.9, 0.95),
            weight_decay = strategy.args.l2
        )
        # configure scheduler
        self.critic_scheduler = get_scheduler(
            "cosine",
            self.critic_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )
        # configure Trainer
        # only use wandb at actor model
        strategy.args.use_wandb = False
        self.trainer = CriticPPOTrainer(
            strategy,
            actor=None,
            critic=self.critic,
            reward_model=None,
            initial_model=None,
            ema_model=None,
            actor_optim=None,
            critic_optim=self.critic_optim,
            actor_scheduler=None,
            critic_scheduler=self.critic_scheduler,
            max_epochs=strategy.args.max_epochs,
            micro_train_batch_size=strategy.micro_train_batch_size,
            micro_rollout_batch_size=strategy.args.micro_rollout_batch_size,
            gradient_checkpointing=strategy.args.gradient_checkpointing,
            prompt_max_len=strategy.args.prompt_max_len,
            value_clip=strategy.args.value_clip,
            eps_clip=strategy.args.eps_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(sequences.to(device), action_mask.to(device), attention_mask.to(device))
        self.critic.train()  # reset model state
        return value.to("cpu")

    def append(self, experience):
        """Append experience to replay buffer."""
        self.trainer.replay_buffer.append(experience)

    def fit(self):
        """Train critic model with the replay buffer."""
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.ppo_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        return status
