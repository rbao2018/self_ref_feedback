import argparse
import os
from datetime import datetime

def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='OpenRLHF Arguments', allow_abbrev=False)
    # Standard arguments.
    parser = _add_initialize_args(parser)
    parser = _add_training_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_rlhf_args(parser)
    args = parser.parse_args()
    return args


def _add_initialize_args(parser):
    group = parser.add_argument_group(title='initialize')
    group.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    group.add_argument("--reward_pretrain", type=str, default=None)
    group.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")
    group.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    group.add_argument("--save_path", type=str, default="/tmp/ckpt")
    group.add_argument("--save_steps", type=int, default=-1)
    group.add_argument("--logging_steps", type=int, default=10)
    group.add_argument("--logging_path", type=str, default="/tmp/")
    group.add_argument("--eval_steps", type=int, default=-1)
    group.add_argument("--ckpt_path", type=str, default="/tmp/ckpt/checkpoints_rm")
    group.add_argument("--max_ckpt_num", type=int, default=3)
    group.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
    group.add_argument("--max_epochs", type=int, default=1)
    group.add_argument("--num_workers", type=int, default=64, help="Number of workers for processing the data.")
    group.add_argument("--micro_train_batch_size", type=int, default=8)
    group.add_argument("--actor_train_batch_size", type=int, default=8)
    group.add_argument("--critic_train_batch_size", type=int, default=8)
    group.add_argument("--train_batch_size", type=int, default=8)
    group.add_argument("--max_samples", type=int, default=10000000)
    group.add_argument("--load_checkpoint", action="store_true", default=False)
    group.add_argument("--max_norm", type=float, default=1.0)
    group.add_argument("--max_len", type=int, default=512)
    group.add_argument("--l2", type=float, default=0.0)
    group.add_argument("--loss", type=str, default="sigmoid")
    group.add_argument("--seed", type=int, default=42)
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    group.add_argument("--zero_stage", type=int, default=2)
    group.add_argument("--bf16", action="store_true", default=False)
    group.add_argument("--learning_rate", type=float, default=1e-5)
    group.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    group.add_argument("--adam_offload", action="store_true", default=False)
    group.add_argument("--flash_attn", action="store_true", default=True)
    group.add_argument("--compute_fp32_loss", action="store_true", default=False)
    group.add_argument("--pretrain_mode", action="store_true", default=False)
    group.add_argument("--aux_loss_coef", type=float, default=0.0)
    group.add_argument("--lm_loss_coef", type=float, default=1.0)
    group.add_argument("--grad_accum_dtype", type=str, default=None)
    group.add_argument("--disable_trace_cache", action="store_true", default=False)
    group.add_argument("--load_in_4bit", action="store_true", default=False)
    group.add_argument("--lora_rank", type=int, default=0)
    group.add_argument("--lora_alpha", type=int, default=16)
    group.add_argument("--target_modules", type=list, default=None)
    group.add_argument("--packing_samples", action="store_true", default=False)
    group.add_argument("--gradient_checkpointing", action="store_true", default=False)
    group.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")
    group.add_argument("--lr_scheduler", type=str, default="cosine")
    group.add_argument("--disable_fast_tokenizer", action="store_true")
    group.add_argument("--enable_fsdp", action="store_true")
    group.add_argument("--low_cpu_fsdp", action="store_true")
    group.add_argument("--enable_hsdp", action="store_true")
    group.add_argument("--replica_group_size", type=int, default=1)
    group.add_argument("--sharding_group_size", type=int, default=1)
    group.add_argument("--fsdp_cpu_offload", action="store_true")
    group.add_argument("--fsdp_activation_checkpointing", action="store_true")
    group.add_argument("--sharding_strategy", type=str, default="FULL_SHARD")
    group.add_argument("--gradient_clipping", action="store_true")
    group.add_argument("--gradient_clipping_threshold", type=float, default=1.0)
    group.add_argument("--checkpoint_type", type=str, default="FULL_STATE_DICT")
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')
    group.add_argument('--oss_bucket_prefix', type=str, default=None,
                       help='oss prefix to save checkpoints to.')
    group.add_argument('--oss_output_dir', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--checkpoints_save_strategy', type=str, default='epoch',
                       help='oss prefix to save checkpoints to.')
    group.add_argument('--output_dir', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save_interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--checkpoints_save_num', type=int, default=3,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no_save_afs', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--save_wait_time', type=int, default=30)
    return parser


def _add_rlhf_args(parser):
    group = parser.add_argument_group(title='(Step 3) RLHF training arguments')
    ## Make parameters for reward modeling
    group.add_argument("--perf", action="store_true", default=False)
    group.add_argument('--use_sigmoid', action='store_true', default=False)
    group.add_argument('--use_margin', action='store_true', default=False)
    group.add_argument("--normalize_reward", action="store_true", default=False)
    group.add_argument('--scale_reward', action='store_true', default=False)
    ## Make parameters for model generation
    group.add_argument("--top_p", type=float, default=0.9, 
                       help="Float that controls the cumulative probability of the top tokens")
    group.add_argument("--top_k", type=int, default=5, 
                       help="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.")
    group.add_argument("--temperature", type=float, default=1.0, 
                       help="Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.")
    group.add_argument("--frequency_penalty", type=float,default=0.0,
                       help="Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.")
    group.add_argument("--repetition_penalty", type=float,default=1.0,
                       help="Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.")
    group.add_argument("--length_penalty", type=float,default=1.0,
                       help="Float that penalizes sequences based on their length. Used in beam search.")
    group.add_argument("--gpu_memory_utilization", type=float,default=0.9,
                       help="Make gpu save memory for vllm, which differs from model size!")
    group.add_argument("--actor_learning_rate", type=float,default=9.65e-6,
                       help="Initial learning rate (after the potential warmup period) to use.")
    group.add_argument("--critic_learning_rate", type=float, default=5e-6,
                       help="Initial learning rate (after the potential warmup period) to use.")
    group.add_argument("--actor_weight_decay", type=float, default=0.,
                        help="Weight decay to use.")
    group.add_argument("--critic_weight_decay", type=float, default=0.,
                        help="Weight decay to use.")
    # Make parameters for ray ppo training
    group.add_argument("--ref_num_nodes", type=int, default=1, 
                       help="number of nodes for reference")
    group.add_argument("--ref_num_gpus_per_node", type=int, default=1, 
                       help="number of gpus per node for reference")
    group.add_argument("--reward_num_nodes", type=int, default=1, 
                       help="number of nodes for reward model")
    group.add_argument("--reward_num_gpus_per_node", type=int, default=1, 
                       help="number of gpus per node for reward model")
    group.add_argument("--colocate_critic_reward", action="store_true", default=False,
                       help="whether to colocate critic and reward model, if true, they will share same gpus.")
    group.add_argument("--colocate_reward_ref", action="store_true", default=False,
                       help="whether to colocate reference and reward model, if true, they will share same gpus.")
    group.add_argument("--actor_num_nodes", type=int, default=1, 
                       help="number of nodes for actor")
    group.add_argument("--actor_num_gpus_per_node", type=int, default=1, 
                       help="number of gpus per node for actor")
    group.add_argument("--critic_num_nodes", type=int, default=1, 
                       help="number of nodes for critic")
    group.add_argument("--critic_num_gpus_per_node", type=int, default=1, 
                       help="number of gpus per node for critic")
    group.add_argument("--colocate_actor_ref", action="store_true", default=False,
                       help="whether to colocate actor and ref model, if true, they will share same gpus.")

    # optional vLLM for text generation
    group.add_argument("--vllm_num_engines", type=int, default=None, 
                       help="number of vLLM Engines")
    group.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                       help="tensor parallel size of vLLM Engine for multi-GPU inference")
    group.add_argument("--prompt_data", type=str, default=None)
    group.add_argument("--prompt_data_probs", type=str, default="1.0",
                       help="sampling probs for datasets")
    group.add_argument("--pretrain_data", type=str, default=None)
    group.add_argument("--pretrain_data_probs", type=str, default="1.0",
                       help="sampling probs for datasets")
    group.add_argument("--num_episodes", type=int, default=1)
    group.add_argument("--buffer_limit", type=int, default=0)
    group.add_argument("--buffer_cpu_offload", type=bool, default=True)
    group.add_argument("--buffer_history_ratio", type=float, default=0.0)
    group.add_argument("--rollout_batch_size", type=int, default=512)
    group.add_argument("--micro_rollout_batch_size", type=int, default=8)
    group.add_argument("--n_samples_per_prompt", type=int, default=1)
    group.add_argument("--prompt_max_len", type=int, default=1024)
    group.add_argument("--generate_max_len", type=int, default=1024)
    group.add_argument("--ptx_coef", type=float, default=0.05)
    group.add_argument("--eps_clip", type=float, default=0.2)
    group.add_argument("--value_clip", type=float, default=0.2)
    group.add_argument("--lambd", type=float, default=0.95)
    group.add_argument("--gamma", type=float, default=1.0)
    group.add_argument("--kl_target", type=float, default=None)
    group.add_argument("--init_kl_coef", type=float, default=0.02)
    ## Make EMA as an optional feature
    group.add_argument("--enable_ema", action="store_true", 
                       help="Enable EMA checkpoint for the model.")
    group.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')
    group.add_argument('--eval_iters', type=int, default=512,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval_num', type=int, default=1,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument('--skip_train', action='store_true',
                       default=False, help='If set, bypass the training loop, '
                       'optionally do evaluation for validation/test, and exit.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')
    # custom dataset key name
    group.add_argument("--input_key", type=str, default=None)
    group.add_argument("--output_key", type=str, default=None)
    group.add_argument("--prompt_key", type=str, default=None)
    group.add_argument("--chosen_key", type=str, default="chosen")
    group.add_argument("--rejected_key", type=str, default="rejected")
    group.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    group.add_argument("--apply_chat_template", action="store_true", default=False, 
                       help="Use HF tokenizer chat template")
    group.add_argument("--tokenizer_chat_template", type=str, default=None)
    # wandb pamameters
    group.add_argument("--use_wandb", action='store_true')
    group.add_argument("--wandb_org", type=str, default="OpenRLHF")
    group.add_argument("--wandb_group", type=str, default="Alignment")
    group.add_argument("--wandb_project", type=str, default="test")
    group.add_argument("--wandb_run_name", type=str,
                       default="%s" % datetime.now().strftime("%m%dT%H:%M"))
    return parser
