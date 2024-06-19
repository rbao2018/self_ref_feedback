
from dataclasses import dataclass

import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import ShardingStrategy

from transformers.models.llama.modeling_llama import LlamaDecoderLayer


@dataclass
class sharding_dict:
    """
    sharding dict
    """
    NO_SHARD=ShardingStrategy.NO_SHARD
    SHARD_GRAD_OP=ShardingStrategy.SHARD_GRAD_OP
    FULL_SHARD=ShardingStrategy.FULL_SHARD
    HYBRID_SHARD=ShardingStrategy.HYBRID_SHARD


@dataclass
class layer_type_dict:
    """"
    asdasdasd
    """
    llama = LlamaDecoderLayer

    
def get_layer_wrapper(layer_type: nn.Module):
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper
    import functools
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_type},
    )
    return llama_auto_wrap_policy


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")
    from functools import partial
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


def get_hsdp_device_mesh(
        world_size : int,
        replica_group_size, 
        sharding_group_size, 
        device=None):
    """
     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.

    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """

    if replica_group_size is None or sharding_group_size is None:
        raise ValueError("Both replica_group_size and sharding_group_size must be provided.")

    device = device or f"cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(
            f"World size {world_size} is not evenly divisible by " f"sharding group size {sharding_group_size}."
        )

    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(
            f"The calculated number of replica groups is not evenly divisible by "
            f"replica_group_size {replica_group_size}."
        )

    device_mesh = init_device_mesh(device, (replica_group_size, sharding_group_size))
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    return device_mesh
