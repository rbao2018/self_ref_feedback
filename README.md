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

To use OpenRLHF, first launch the docker container (**Recommended**) and `pip install` openrlhf inside the docker container:

```bash
# Launch the docker container
docker run --runtime=nvidia -it --rm --shm-size="8g" --cap-add=SYS_ADMIN nvcr.io/nvidia/pytorch:23.10-py3 bash

# apt-get & pip install
apt-get update
apt-get -y install libaio-dev tmux net-tools
pip uninstall transformer-engine transformers -y
pip install vllm==0.4.2
pip uninstall flash-attn -y
MAX_JOBS=32 pip install flash-attn==2.6.1 --no-build-isolation
pip install openrlhf==0.3.7

# You can also install the latest version of OpenRLHF from git repo
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .
```

> [!NOTE]
>vLLM, flash-attn and other libraries will specify the versions of PyTorch and CUDA. We recommend installing them on machines with CUDA version >= 12. We recommend using vLLM 0.4.2, as versions 0.4.3+ currently only support weight synchronization (DeepSpeed to vLLM) via Gloo (`--vllm_sync_backend gloo`).

### Reward Model Training
```bash
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

# RM samples packing
# --packing_samples
```
> [!NOTE]
> OpenRLHF SFT/DPO/RewardModel trainers support `--packing_samples` [based on `--flash_attn`](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing)

### PPO with Ray and vLLM

To improve RLHF training speed or support 70B models, we can use the PPO with Ray and vLLM acceleration

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 2 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 2 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2 \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --ref_reward_offload \
  --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
  --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
  --micro_train_batch_size 8 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 1024 \
  --max_samples 100000 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb {wandb_token}

```
> [!NOTE]
> Do not set `--vllm_num_engines` means not using the vLLM engine.
> You can also use ``setup_commands`` to let Ray automatically deploy the environment, such as `--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`.

The launch scripts and documents for supported algorithms are in [example/scripts](./examples/scripts/) and [Documents - Usage](https://openrlhf.readthedocs.io/en/latest/usage.html)


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

- [Vllm](https://github.com/vllm-project/vllm)
- [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)
- [lmdeploy](https://github.com/InternLM/lmdeploy)
