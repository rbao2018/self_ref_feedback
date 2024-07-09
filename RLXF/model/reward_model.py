from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from openrlhf.utils.logging import init_logger
from openrlhf.models.utils import find_all_linear_names

logger = init_logger(__name__)

# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    bf16=True,
    global_rank=0,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    use_flash_attention_2 = True,
    init_value_head: bool = False,
    **kwargs,
) -> nn.Module:
    """Get transformer with a sequence classification head on top (linear layer).

    Args:
        model_name_or_path (str): Path to pretrained model.
        model_type (str): Either "reward" or "critic" or "actor".
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.

    Returns:
        nn.Module: pretrained transformer model.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.lm_head_name = kwargs.get("lm_head_name", "lm_head")

    try:
        base_class = AutoModelForCausalLM._model_mapping[type(config)]
        # base_pretrained_class = base_class.__base__
        if model_type == "reward":
            cls_class = _get_reward_model(base_class)
        elif model_type == "critic":
            cls_class = _get_critic_model(base_class)
        elif model_type == "actor":
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

        # base_pretrained_class = get_class_from_dynamic_module(
        #     f"{module_file}.{pretrained_model_name}", model_name_or_path
        # )
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)
   
        if model_type == "reward":
            cls_class = _get_reward_model(base_class)
        elif model_type == "critic":
            cls_class = _get_critic_model(base_class)
        elif model_type == "actor":
            cls_class = _get_actor_model(base_class)
        else:
            raise NotImplementedError

    model = cls_class.from_pretrained(
        model_name_or_path,
        config = config,
        trust_remote_code = True,
        torch_dtype = torch.bfloat16 if bf16 else torch.float16,
        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager",
        **kwargs,
    )
    if model_type == "actor":
        model.config.architectures = config.architectures

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules or find_all_linear_names(model),
            lora_dropout=0,
            bias="none",
        )
        model = get_peft_model(model, lora_config)


    # Mixtral 8x7b - balancing loss
    if "output_router_logits" in model.config.to_dict():
        print("[Mixtral 8x7b] set output_router_logits as True")
        model.config.output_router_logits = True

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
    return model


def _get_reward_model(base_llm_model):
    class LLMForSequenceRegression(base_llm_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # setattr(self, self.base_model_prefix, base_pretrained_model(config))
            # setattr(self, config.lm_head_name, nn.Linear(config.hidden_size, config.vocab_size, bias=False))
            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            last_hidden_state = outputs["last_hidden_state"]
            all_values = self.value_head(last_hidden_state).squeeze(-1)
            logits = getattr(self, self.config.lm_head_name)(last_hidden_state).float()
            # outputs["logits"] = logits
            # print(f"code is here in line 153!!! for logits shape{logits.shape}", flush=True)
            # finding the last "1" in attention mask for eos_token indices
            eos_indices = attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            eos_indices = attention_mask.size(1) - 1 - eos_indices
            reward = all_values.gather(dim=1, index=eos_indices).squeeze(1)
            if return_output:
                return reward, logits
            else:
                return reward

    return LLMForSequenceRegression


def _get_critic_model(base_llm_model):
    class LLMForSequenceRegression(base_llm_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # setattr(self, self.base_model_prefix, base_pretrained_model(config))
            # setattr(self, config.lm_head_name, nn.Linear(config.hidden_size, config.vocab_size, bias=False))
            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
            last_hidden_state = outputs["last_hidden_state"]
            all_values = self.value_head(last_hidden_state).squeeze(-1)
            value = all_values[:, :-1]
            # print(f"return_output:{return_output}", flush=True)
            # print(f"action_mask:{action_mask}", flush=True)
            if action_mask is not None:
                value = value[:, -action_mask.size(1):]
            if return_output:
                return (value, outputs)
            else:
                return value

    return LLMForSequenceRegression

def _get_actor_model(base_llm_model):
    class LLMForSequenceRegression(base_llm_model):
        supports_gradient_checkpointing = True

        def forward(
            self,
            input_ids,
            num_actions: int = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            """Returns action log probs"""
            # https://github.com/OpenLLMAI/OpenRLHF/issues/217
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
            labels = input_ids[:, 1:].unsqueeze(-1)
            last_hidden_state = output["last_hidden_state"]
            logits = getattr(self, self.config.lm_head_name)(last_hidden_state).float()
            
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            log_probs = log_probs.gather(dim=-1, index=labels).squeeze(-1)

            if return_output:
                return output if num_actions is None else (log_probs[:,-num_actions:], output)
            else:
                return log_probs[:, -num_actions:]

    return LLMForSequenceRegression