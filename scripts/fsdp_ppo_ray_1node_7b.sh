#!/bin/bash
NNODES=$1
ACTOR_MODEL=$2
CRITIC_MODEL=$3
Actor_Lr=$4
Critic_Lr=$5
DATASET=$6
PROBS=$7
LOGDIR=$8
PREFIX=$9

if [[ -z "$NNODES"|| -z "$Actor_Lr" || -z "$Critic_Lr"  ]]; then
    echo "Error: Variable is not set or is empty."
    exit 1
fi

if [[ "$NNODES" == "1" ]]; then
    MASTER_ADDR="localhost"
fi

mkdir -p $LOGDIR/$PREFIX
ip_head="$MASTER_ADDR:6379"

ray start --head --node-ip-address=$MASTER_ADDR --port=6379

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export NCCL_ALGO=Tree

now_date=$(date +%Y_%m%d_%H%M)

ray job submit --runtime-env-json='{"working_dir": "/root/OpenRLHF"}' -- python /root/self_ref_feedback/fsdp_ppo_ray.py \
    --colocate_actor_ref \
    --colocate_critic_reward \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --colocate_reward_ref \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 2 \
    --pretrain $ACTOR_MODEL \
    --reward_pretrain $CRITIC_MODEL \
    --logging_path $LOGDIR/$PREFIX \
    --save_path /root/temp/output/$PREFIX \
    --critic_train_batch_size 4 \
    --actor_train_batch_size 8 \
    --train_batch_size 128 \
    --rollout_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --num_episodes 1 \
    --max_epochs 1 \
    --logging_steps 1 \
    --apply_chat_template \
    --input_key message \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --repetition_penalty 1.02 \
    --bf16 \
    --packing_samples \
    --actor_learning_rate $Actor_Lr \
    --critic_learning_rate $Critic_Lr \
    --init_kl_coef 0.01 \
    --prompt_data $DATASET \
    --prompt_data_probs $PROBS \
    --use_wandb \
    --actor_init_on_gpu \
    --gradient_checkpointing \
    --flash_attn >>"$LOGDIR/$PREFIX/$now_date"_train.log 2>"$LOGDIR/$PREFIX/$now_date"_train.err