from typing import List

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from RLXF.fsdp_strategy import FSDPStrategy
from RLXF.parse import parse_args
from RLXF.fsdp_ppo_actor import ActorModelRayActor
from RLXF.fsdp_ppo_critic import CriticModelRayActor
from RLXF.launcher import (
    PPORayActorGroup, 
    ReferenceModelRayActor, 
    RewardModelRayActor
)
from RLXF.trainer.vllm_engine import LLMRayActor

def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)
    # NOTE: reward function for multiple reward models, replace this with your own function!


def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node

    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"
    assert (
        critic_world_size & (critic_world_size - 1)
    ) == 0, f"critic_world_size must be power of 2, got {critic_world_size}"
    assert (
        actor_world_size % critic_world_size == 0
    ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"
    assert (
        args.vllm_num_engines <= actor_world_size
    ), "vLLM engine should be less than actor world size"

def train(args):
    _validate_args(args)
    # configure strategy
    # strategy = get_strategy(args)
    args.input_template = args.input_template.replace('\\n', '\n')
    actor_strategy = FSDPStrategy(
        seed=args.seed,
        micro_train_batch_size=args.actor_train_batch_size,
        train_batch_size=args.train_batch_size,
        sharding_strategy="FULL_SHARD",
        is_actor=True,
        fsdp_activation_checkpointing=args.gradient_checkpointing,
        gradient_clipping_threshold=args.max_norm,
        args=args
    )
    critic_strategy = FSDPStrategy(
        seed=args.seed,
        micro_train_batch_size=args.critic_train_batch_size,
        train_batch_size=args.train_batch_size,
        sharding_strategy="FULL_SHARD",
        fsdp_activation_checkpointing=args.gradient_checkpointing,
        gradient_clipping_threshold=args.max_norm,
        args=args
    )
    reference_strategy = FSDPStrategy(sharding_strategy="FULL_SHARD", args=args)
    reward_strategy = FSDPStrategy(sharding_strategy="FULL_SHARD", args=args)

    pg = None
    if args.colocate_actor_ref:
        assert (
            args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [
            {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node * 8}
            for _ in range(args.actor_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

    # NOTE(wuxibin): Why don't we allocate 0.5 gpu for each actor when colocate models?
    # Say we have 1 node with 4 GPUs, and num_gpus_per_node for each model is 4.
    # If we allocate 0.5 gpu for both actor and ref model, then gpu allocation is
    #   |actor|actor|actor|actor|  ref | ref  | ref  | ref |
    #   |GPU0 |GPU0 |GPU1 |GPU1 | GPU2 | GPU2 | GPU3 | GPU3 |
    #
    # So 0.75/0.25 gpu is a tricky to let Ray spread all models evenly on all gpus.
    #   |actor| ref  |actor| ref  |actor| ref  |actor|ref  |
    #   |GPU0 | GPU0 |GPU1 | GPU1 |GPU2 | GPU2 |GPU3 | GPU3 |
    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    ref_model = PPORayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.25 if pg else 1,
    )
    pg = None
    if args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [
            {"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node * 8}
            for _ in range(args.critic_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())

    critic_model = PPORayActorGroup(
        args.critic_num_nodes,
        args.critic_num_gpus_per_node,
        CriticModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    # multiple reward models
    reward_pretrains = args.reward_pretrain.split(",")
    reward_models = []
    for _ in reward_pretrains:
        reward_models.append(
            PPORayActorGroup(
                args.reward_num_nodes,
                args.reward_num_gpus_per_node,
                RewardModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )
        )


    # init reference/reward/actor model
    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(reference_strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(actor_strategy, args.pretrain))
    for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
        refs.extend(
            reward_model.async_init_model_from_pretrained(reward_strategy, reward_pretrain)
        )

    # init vLLM engine for text generation
    vllm_engines = []
    if args.vllm_num_engines is not None:
        for _ in range(args.vllm_num_engines):
            # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
            num_gpus = int(args.vllm_tensor_parallel_size == 1)
            scheduling_strategy = None

            if args.vllm_tensor_parallel_size > 1:
                bundles = [{"GPU": 1, "CPU": 1}] * args.vllm_tensor_parallel_size
                pg = placement_group(bundles)
                ray.get(pg.ready())

                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
                )

            vllm_engines.append(
                LLMRayActor.options(
                    num_cpus=args.vllm_tensor_parallel_size,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    args.pretrain,
                    load_format="dummy",
                    trust_remote_code=True,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    seed=args.seed,
                )
            )

    # critic scheduler initialization depends on max_step, so we have to init critic after actor
    # TODO: use first reward model as critic model
    max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
    refs.extend(
        critic_model.async_init_model_from_pretrained(
            critic_strategy, reward_pretrains[0], max_steps
        )
    )
    ray.get(refs)

    # train actor and critic mdoel
    refs = actor_model.async_fit_actor_model(
        critic_model,
        ref_model,
        reward_models,
        reward_fn=reward_fn,
        vllm_engines=vllm_engines,
    )
    ray.get(refs)

    # save model
    # ray.get(actor_model.async_save_actor_model())


if __name__ == "__main__":
    args = parse_args()
    train(args)
