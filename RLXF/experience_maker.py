import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.distributed
import torch.nn as nn

from openrlhf.models.utils import compute_reward, masked_mean
from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)

# NOTE: Used in step 3 for reward normalization and scaling
class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.values = self.values.pin_memory()
        self.returns = self.returns.pin_memory()
        self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: nn.Module,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.running_reward_stats = RunningMoments()

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, 
                        prompts: Union[str, List[str], List[List[int]]],
                        **generate_kwargs) -> Experience:
        self.actor.eval()
        self.critic.eval()
        self.initial_model.eval()
        self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        value = self.critic(sequences, action_mask, attention_mask)

        # rewards
        r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # reset model state
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.input_ids = []

    def clear(self):
        self.input_ids = []

    def add_requests(self, input_prompts):
        rank = torch.distributed.get_rank()
        llm = self.vllm_engines[rank % len(self.vllm_engines)]
        ray.get(llm.add_request.remote(input_prompts))

    @torch.no_grad()
    def make_experience(self, 
                        replay_buffer,
                        sequences_vllm: torch.Tensor,
                        attention_mask_vllm: torch.Tensor,
                        action_mask_vllm: torch.Tensor,
                        batch_size: int, 
                        **generate_kwargs
                        ) -> Experience:
        self.actor.eval()
        device = torch.cuda.current_device()
        a = torch.split(sequences_vllm, batch_size, dim=0)
        b = torch.split(attention_mask_vllm, batch_size, dim=0)
        c = torch.split(action_mask_vllm, batch_size, dim=0)
        for (sequences_cpu, attention_mask_cpu, action_mask_cpu) in zip(a,b,c):
            num_actions = action_mask_cpu.size(1)
            sequences, attention_mask, action_mask = (
                sequences_cpu.to(device),
                attention_mask_cpu.to(device),
                action_mask_cpu.to(device)
            )
            # values
            value_ref = self.critic.forward.remote(sequences_cpu, action_mask_cpu, attention_mask_cpu)

            # rewards
            r_refs = []
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))

            # log probs
            start = time.time()
            action_log_probs = self.actor(
                sequences, 
                num_actions, 
                attention_mask
            )
            actor_time = time.time() - start

            # init log probs
            base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)
            # self.strategy.print(f"log_probs: {action_log_probs[0][-num_actions:]}")

            # wait initial/critic/reward model done
            start = time.time()
            ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            wait_time = time.time() - start

            base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
            base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)
            rewards = [r.to(device) for r in rewards]
            r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

            self.running_reward_stats.update(r)

            if self.strategy.args.normalize_reward:
                r = (r - self.running_reward_stats.mean) / self.running_reward_stats.std
            elif self.strategy.args.scale_reward:
                r = r / self.running_reward_stats.std
            else:
                pass

            reward, kl = compute_reward(
                r,
                self.kl_ctl.value,
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
            )
            advantage, returns = self.get_advantages_and_returns(
                value,
                reward,
                action_mask,
                generate_kwargs["gamma"],
                generate_kwargs["lambd"],
            )

            info = {
                "kl": masked_mean(kl, action_mask, dim=-1),
                "reward": r,
                "return": reward.sum(dim=-1),
                "response_length": action_mask.float().sum(dim=-1),
                "total_length": attention_mask.float().sum(dim=-1),
            }

            experience = Experience(
                sequences,
                action_log_probs,
                value,
                returns,
                advantage,
                attention_mask,
                action_mask,
                info,
            )

            if self.strategy.args.perf:
                # batch_size = 1 if isinstance(sequences_cpu, str) else sequences_cpu.size(0)
                # info["generate_time"] = torch.full(
                #     (batch_size,), generate_time, device=device
                # )
                info["actor_time"] = torch.full((batch_size,), actor_time, device=device)
                info["wait_time"] = torch.full((batch_size,), wait_time, device=device)

            # send experience to critic
            experience_cpu = deepcopy(experience)
            experience_cpu.to_device("cpu")
            self._ref = self.critic.append.remote(experience_cpu)
            replay_buffer.append(experience)
        self.actor.train()  # reset model state
        # return experience

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        return self.actor.generate(**inputs, **kwargs)

    def generate_vllm(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # no round-robin load balance
        rank = kwargs.get("rank", 0)
        total_group = kwargs.get("total_group", 2)
        if rank == 0:
            # llm = self.vllm_engines[rank]
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 5),
                repetition_penalty=kwargs.get("repetition_penalty",1.0),
                min_tokens=kwargs.get("min_tokens",16),
                max_tokens=kwargs.get("max_new_tokens", 1024)
            )
            # using prompt token ids for vLLM       
            ray.get([engine.generate.remote(
                sampling_params=sampling_params,
                use_tqdm=False) for engine in self.vllm_engines
                ]
            )
        torch.distributed.barrier()
        engine = self.vllm_engines[rank % len(self.vllm_engines)]
        outputs = ray.get(engine.get_output.remote(rank // len(self.vllm_engines), total_group))

        # NOTE: concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        for i, output in enumerate(outputs):
            # TODO: how to force vLLM generate at least one token?
            output_token_ids = output.outputs[0].token_ids
            if output_token_ids[0] == self.tokenizer.eos_token_id:
                logger.warning(f"Only EOS output for prompt: {output}")
                outputs[i] = outputs[(i+1) % len(outputs)]
            max_input_len = max(max_input_len, len(output.prompt_token_ids))
            max_output_len = max(max_output_len, len(output_token_ids))

        # self.strategy.print(f"log prob outputs:{outputs[0].outputs[0].logprobs}")

        sequences = []
        for output in outputs:
            # left padding input
            input_len = len(output.prompt_token_ids)
            input_ids = [pad_token_id] * (max_input_len - input_len) + output.prompt_token_ids

            # right padding output
            output_len = len(output.outputs[0].token_ids)
            output_ids = output.outputs[0].token_ids + [pad_token_id] * (max_output_len - output_len)
            if output_ids[output_len - 1] != eos_token_id:
                assert output_len == max_output_len
                output_ids[-1] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)
        attention_mask = sequences.ne(pad_token_id).to(dtype=torch.long)
        state_seq = sequences[:, max_input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        # return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")

        return {
            "sequences_vllm": sequences,
            "attention_mask_vllm": attention_mask,
            "action_mask_vllm": action_mask
            }

    def flush(self):
        "Ensure all experience has been send to critic"
        ray.get(self._ref)
        self._ref = None
