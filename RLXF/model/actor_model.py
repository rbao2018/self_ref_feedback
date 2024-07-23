from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from openrlhf.utils.logging import init_logger
from openrlhf.models.utils import find_all_linear_names

from .packing_utils import patch_for_block_diag_attn

logger = init_logger(__name__)

# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_actor_model(
    model_name_or_path: str,
    model_type: str,
    args,
    **kwargs) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic" or "actor".
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        flash_attn (bool, optional): Whether use Flash Attention 2.0. Defaults to False.

    Returns:
        nn.Module: pretrained transformer model.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.lm_head_name = kwargs.get("lm_head_name", "lm_head")

    try:
        base_class = AutoModelForCausalLM._model_mapping[type(config)]
        # base_pretrained_class = base_class.__base__
        if model_type == "actor":
            cls_class = _get_actor_model(base_class)
        else:
            raise NotImplementedError
        logger.info(f"base_class: {base_class}")
    except Exception as e:
        print("Failed to load from AutoModelForCausalLM, construct from modelling file.")
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # special case
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)

        if model_type == "actor":
            cls_class = _get_actor_model(base_class)
        else:
            raise NotImplementedError

    model = cls_class.from_pretrained(
        model_name_or_path,
        config = config,
        trust_remote_code = True,
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float16,
        attn_implementation = "flash_attention_2" if args.flash_attn else "eager",
        **kwargs,
    )
    if model_type == "actor":
        model.config.architectures = config.architectures

    # LoRA
    if args.lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules or find_all_linear_names(model),
            lora_dropout=0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    
    
    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True
    
    # packing samples using Flash Attention 2
    if args.packing_samples:
        assert args.flash_attn, "Only support `--packing_samples` with Flash Attention 2."
        model_type = getattr(model.config, "model_type", None)
        patch_for_block_diag_attn(model_type)

    return model


def _get_actor_model(base_llm_model):
    class LLMForSequenceRegression(base_llm_model):
        supports_gradient_checkpointing = True

        def forward(
            self,
            input_ids,
            num_actions: int = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            packing_samples=False,
        ) -> torch.Tensor:
            """Returns action log probs"""
            seq_lens = attention_mask.sum(dim=-1)
            max_len = seq_lens.max().item()
            if packing_samples:
                # 创建一个范围张量
                range_tensor = torch.arange(max_len, device=attention_mask.device).unsqueeze(0)
                # 创建一个掩码，用于筛选有效的位置
                mask = range_tensor < seq_lens.unsqueeze(1)
                # 使用掩码来选择有效的元素
                position_ids = range_tensor.expand(seq_lens.size(0), max_len)[mask]
                # 输入的inputs_ids是一个batch(shape=(1, sum_of_seq_lens))的数据，所以需要在前面增加一个维度
                position_ids = position_ids.unsqueeze(0)
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            output = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            labels = input_ids[:, 1:].unsqueeze(-1) # bsz * seq_len * 1
            if packing_samples:
                padding = torch.zeros(1, 1, 1, dtype=input_ids.dtype, device=input_ids.device)
                labels = torch.cat((labels, padding), dim=1)
            last_hidden_state = output["last_hidden_state"]
            logits = getattr(self, self.config.lm_head_name)(last_hidden_state).float()
            
            if not packing_samples:
                log_probs = F.log_softmax(logits, dim=-1)
            else:
                log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

            log_probs = log_probs.gather(dim=-1, index=labels).squeeze(-1)

            if return_output:
                if packing_samples or num_actions is None:
                    return log_probs, output
                else:
                    return log_probs[:,-num_actions:], output
            else:
                if packing_samples or num_actions is None:
                    return log_probs
                return log_probs[:, -num_actions:]

    return LLMForSequenceRegression