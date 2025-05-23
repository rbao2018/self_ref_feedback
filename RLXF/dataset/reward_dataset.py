# -*- coding: utf-8 -*-

from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.datasets.utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            if isinstance(data[chosen_key], str) and isinstance(data[rejected_key], str):
                chosen_list = [{"role":"assistant", "content":data[chosen_key]}]
                rejected_list = [{"role":"assistant", "content":data[rejected_key]}]
                chosen = apply_chat_template(data[prompt_key] + chosen_list, tokenize=False, add_generation_prompt=False)[len(prompt) :]
                rejected = apply_chat_template(data[prompt_key] + rejected_list, tokenize=False, add_generation_prompt=False)[len(prompt) :]
            else:
                raise NotImplementedError
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False, add_generation_prompt=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False, add_generation_prompt=False)

            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, margin


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        is_dpo=False,
        num_processors=8
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo

        self.prompts = []
        self.chosens = []
        self.rejects = []
        if self.is_dpo:
            self.prompt_ids_lens = []
        else:
            self.margins = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_dpo = is_dpo

        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template


        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, reject, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        else:
            extra = self.extras[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )
    
    def collate_fn(self, item_list):
        chosen_masks = []
        rejects_masks = []
        chosen_ids = []
        reject_ids = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_masks.append(chosen_mask.flatten())
            reject_ids.append(reject_id.flatten())
            rejects_masks.append(rejects_mask.flatten())
            extras.append(float(extra) if not self.is_dpo else list(extra))

        concat_ids = zero_pad_sequences(chosen_ids + reject_ids, side="right", value=self.tokenizer.pad_token_id)
        concat_masks = zero_pad_sequences(chosen_masks + rejects_masks, side="right")
        chosen_labels = zero_pad_sequences(chosen_ids, side="right", value=-100)

        return {
            "concat_ids": concat_ids,
            "concat_masks": concat_masks,
            "chosen_labels": chosen_labels,
            "extras": extras
        }

    def packing_collate_fn(self, item_list):
        extras = []
        chosen_ids = []
        chosen_att_masks = []
        rejected_ids = []
        rejected_att_masks = []
        
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(chosen_mask.flatten())
            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(rejects_mask.flatten())
            extras.append(float(extra) if not self.is_dpo else list(extra))
        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0) # 1 * chose_len + reject_len
        concat_masks = zero_pad_sequences(chosen_att_masks + rejected_att_masks, side="right", value=0)
        chosen_labels = torch.cat(chosen_ids, dim=0).unsqueeze(0)

        return {
            "concat_ids": packed_input_ids,
            "concat_masks": concat_masks,
            "chosen_labels": chosen_labels,
            "extras": extras
            }
