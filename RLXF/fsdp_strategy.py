import os
import random
import gc
from collections import defaultdict
from datetime import timedelta
from typing import Tuple, Union

import numpy as np
import ray
from safetensors.torch import save_file
from transformers.modeling_utils import shard_checkpoint
import torch
import torch.distributed as dist
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload, FullStateDictConfig, StateDictType

from openrlhf.utils.deepspeed import DeepspeedStrategy

from .fsdp_utils import (
    sharding_dict,
    layer_type_dict,
    get_layer_wrapper,
    get_hsdp_device_mesh,
    apply_fsdp_checkpointing,
)

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class FSDPStrategy(DeepspeedStrategy):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        seed: int = 42,
        micro_train_batch_size=1,
        train_batch_size=8,
        sharding_strategy="FULL_SHARD",
        fsdp_cpu_offload=False,
        is_actor=False,
        backward_prefetch=False,
        forward_prefetch=False,
        fsdp_activation_checkpointing=False,
        gradient_clipping_threshold: float = 0.0,
        args=None,
    ) -> None:
        super().__init__()
        self.args = args
        self.seed = seed
        self.gradient_clipping_threshold = gradient_clipping_threshold
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.sharding_strategy = sharding_strategy
        self.fsdp_cpu_offload = fsdp_cpu_offload
        self.is_actor = is_actor
        self.backward_prefetch = backward_prefetch
        self.forward_prefetch = forward_prefetch
        self.fsdp_activation_checkpointing = fsdp_activation_checkpointing
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", "fp32")
        # disable_trace_cache
        # self.disable_trace_cache = getattr(args, "disable_trace_cache", False)
        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=8)) -> None:
        self.set_seed(self.seed)
        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:
            # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])
        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group("nccl", timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=kwargs.get("lr", 1e-6),
            betas=kwargs.get("betas", (0.9, 0.95)),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )
        return optimizer

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
    ):
        # DDP only mode, replay buffers on each rank are different.
        if sampler is None:
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
            )

        return DataLoader(
            replay_buffer,
            num_workers=8,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def prepare_model(
        self,
        model: nn.Module,
    ) -> nn.Module:
        fsdp_config = self.get_fsdp_config()
        layer_type = getattr(layer_type_dict, model.config.model_type)
        model = FSDP(
            model,
            auto_wrap_policy=get_layer_wrapper(layer_type),
            cpu_offload=CPUOffload(True) if self.fsdp_cpu_offload else None,
            device_id=torch.cuda.current_device(),
            forward_prefetch=self.forward_prefetch,
            backward_prefetch=self.backward_prefetch,
            **fsdp_config,
        )
        self.print(f"get model: {model}")
        if self.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
        return model

    def get_fsdp_config(self):
        if self.is_actor and self.args.pretrain_data is not None:
            self.train_batch_size *= 2
            self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16 if self.args.bf16 else torch.float16,
            reduce_dtype=torch.bfloat16 if self.args.bf16 else torch.float16,
            buffer_dtype=torch.bfloat16 if self.args.bf16 else torch.float32,
        )
        hsdp_device_mesh = None
        if self.sharding_strategy == "HYBRID_SHARD":
            hsdp_device_mesh = get_hsdp_device_mesh(
                replica_group_size=self.args.replica_group_size, 
                sharding_group_size=self.args.sharding_group_size
            )

        config = {
            "mixed_precision": mixed_precision,
            "device_mesh": hsdp_device_mesh,
            "sharding_strategy": getattr(sharding_dict, self.sharding_strategy),
        }
        return config

    def backward(
            self, 
            loss: torch.Tensor, 
            model: nn.Module, 
            optimizer: Optimizer, 
            **kwargs
    ) -> None:
        loss = loss / self.accumulated_gradient
        return loss.backward()

    def optimizer_step(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        lr_scheduler: optim.lr_scheduler,
        name="model",
        **kwargs,
    ) -> None:
        self.time_steps["total_train_steps"] += 1
        if self.time_steps["total_train_steps"] % self.accumulated_gradient == 0:
            if self.gradient_clipping_threshold > 0.0:
                model.clip_grad_norm_(self.gradient_clipping_threshold)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        raise NotImplementedError

    def optimizer_offload_cpu(self, optimizer: Optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

    def optimizer_recover(self, optimizer: Optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(torch.cuda.current_device())

    def get_full_msd(self, model: nn.Module):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        return cpu_state

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank()

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs):
        model_state_dict = self.get_full_msd(model)
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)
            self.print(f"model going to save at {output_dir}")
            shards, index = shard_checkpoint(
                model_state_dict,
                max_shard_size="5GB",
                weights_name="model.safetensors")
            for shard_file, shard in shards.items():
                save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt"})
            del shards
            gc.collect()
            if index is not None:
                import json
                save_index_file = os.path.join(output_dir, "model.safetensors.index.json")
                # Save the index as well
                with open(save_index_file, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
            output_config_file = os.path.join(output_dir, "config.json")
            model.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(output_dir)
        else:
            import time
            time.sleep(self.args.save_wait_time)
        

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        pass
