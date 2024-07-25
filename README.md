<div align="center">
    <b><font size="5">Aligning Large Language Models from Self-Reference AI Feedback</font></b><br>
    <b><font size="5">with one General Principle</font></b>
    <sup>
    </sup>
    <div> </div>
</div>


<div align="center">

[📘License](#license) |
[🤗Dataset](https://huggingface.co/datasets/rbao2018/Self_Ref_Feedback) |
[📜Paper](https://arxiv.org/abs/2406.11190)

</div>

## Introduction
Our project has implemented the training process of the self-reference AI feedback with one general principle. We also partially refactoring the [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF) framework to improve the efficiency of the PPO algorithm.

## Quick Start

### Installation

```bash
git clone git@github.com:rbao2018/self_ref_feedback.git
cd self_ref_feedback
bash install.sh
```

> [!NOTE]
>vLLM and flash-attn will specify the versions of PyTorch and CUDA. We recommend installing them on machines with CUDA version >= 12. We recommend using vLLM 0.4.2, as versions 0.4.3+ currently only support weight synchronization (DeepSpeed to vLLM) via Gloo (`--vllm_sync_backend gloo`).

### Reward Model Training
```bash
NNODES=1
DATASET=/root/Self_Ref_Feedback/llama2_70b_7b_mavo_4_ref
PROBS=0.95
BS=4
LR=1e-5
LOGDIR=/root/log
PREFIX=test

if [ "$LOGDIR" == "" ]; then
    LOGDIR=/root/output
fi

if [ "$PREFIX" == "" ]; then
    PREFIX=test
fi
if [ "$NNODES" == "1" ]; then
    MASTER_ADDR=localhost
    RANK=0
fi
mkdir -p $LOGDIR/$PREFIX

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export MAX_SEQ_LEN=2048
export NCCL_ALGO=Tree

now_date=$(date +%Y_%m%d_%H%M)

torchrun --nproc_per_node 8 --nnodes $NNODES --master_addr $MASTER_ADDR --master_port 6666 --node_rank $RANK /root/self_ref_feedback/train_rm_llama2.py \
    --logging_path $LOGDIR/$PREFIX \
    --save_path /root/temp/output/$PREFIX \
    --save_steps -1 \
    --logging_steps 10 \
    --eval_steps 128 \
    --train_batch_size 256 \
    --critic_train_batch_size $BS \
    --pretrain /root/huggingface/models/Llama-2-7b-hf \
    --packing_samples \
    --loss logexpwithlm \
    --apply_chat_template \
    --prompt_key message \
    --chosen_key chose \
    --rejected_key reject \
    --max_epochs 1 \
    --zero_stage 3 \
    --max_len $MAX_SEQ_LEN \
    --learning_rate $LR \
    --dataset $DATASET  \
    --dataset_probs $PROBS \
    --use_wandb \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing

# RM samples packing
# --packing_samples
```
> [!NOTE]
> We have made further improvements to the `--packing_samples` method implemented in the OpenRLHF framework. [based on `--flash_attn`] (https://github.com/OpenRLHF/OpenRLHF/blob/v0.3.8/openrlhf/models/packing_utils.py)

### PPO with Ray and vLLM

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export MAX_JOBS=32
export NCCL_ALGO=Tree

ray job submit --runtime-env-json='{"working_dir": "/root/some_dir"}' -- python /root/self_ref_feedback/fsdp_ppo_ray.py \
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
    --pretrain /root/meta-llama/Llama-2-7b-chat-hf \ # for test
    --reward_pretrain /root/meta-llama/Llama-2-7b-chat-hf \ # for test
    --logging_path /root/temp/output/log \
    --save_path /root/temp/output/save_model \
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
    --actor_learning_rate 1e-6 \
    --critic_learning_rate 5e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /root/Self_Ref_Feedback/llama2_70b_7b_mavo_4_ref \
    --prompt_data_probs 1.0 \
    --use_wandb \
    --actor_init_on_gpu \
    --gradient_checkpointing \
    --flash_attn
```
> [!NOTE]
> Do not set `--vllm_num_engines` means not using the vLLM engine.
> You can also use ``setup_commands`` to let Ray automatically deploy the environment, such as `--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`.

### Model Deployment
- Utilize [lmdeploy](https://github.com/InternLM/lmdeploy) for deploying models, enabling quick access to AI feedback and model generation.

### PPO Algorithm Efficiency Improvements
1. Replace the original Deepspeed framework with the FSDP framework to reduce GPU memory usage and increase training speed.
2. Optimize the scheduling algorithm for asynchronous actor-critic training in the PPO training process to enhance overall framework efficiency.
3. Improve the implementation of experience replay generation to avoid the inefficiency of multiple small-batch reply generations by Vllm.


## License

The code is licensed under Apache-2.0, while model weights are fully open for academic research.

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of generative AI:

- [vLLm](https://github.com/vllm-project/vllm)
- [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)
- [lmdeploy](https://github.com/InternLM/lmdeploy)
