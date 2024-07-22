from typing import Callable

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False)

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
        input_template=None,
        is_dpo=False,
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

        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        chosen_key = getattr(self.strategy.args, "chosen_key", None)
        rejected_key = getattr(self.strategy.args, "rejected_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        for data in tqdm(dataset, desc="Tokenizing", disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, margin = preprocess_data(
                data, input_template, prompt_key, chosen_key, rejected_key, apply_chat_template, self.is_dpo
            )
            if self.is_dpo:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                else:
                    self.prompt_ids_lens.append(prompt_ids_len)
            else:
                self.margins.append(margin)

            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]
        else:
            extra = self.margins[idx]

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
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        chosen_ids = zero_pad_sequences(chosen_ids, side="right", value=self.tokenizer.pad_token_id)
        chosen_labels = zero_pad_sequences(chosen_ids, side="right", value=-100)
        chosen_masks = zero_pad_sequences(chosen_masks, side="right")
        reject_ids = zero_pad_sequences(reject_ids, side="right", value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side="right")
        concat_ids = zero_pad_sequences(chosen_ids + reject_ids, side="right", value=self.tokenizer.pad_token_id)
        concat_masks = zero_pad_sequences(chosen_masks + rejects_masks, side="right")
        
        assert concat_ids.shape == concat_masks.shape

        return {
            "concat_ids": concat_ids,
            "concat_masks": concat_masks,
            "chosen_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "reject_ids": reject_ids,
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
            chosen_att_masks.append(chosen_mask)
            extras.append(extra)
            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(rejects_mask)
        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0) # 1 * chose_len + reject_len
        concat_masks = zero_pad_sequences(chosen_att_masks + rejected_att_masks, side="right", value=0)
        chose_id_lens = torch.sum(concat_masks[:len(item_list),:])
        chosen_labels = torch.cat(chosen_ids, dim=0).unsqueeze(0)
        assert chose_id_lens == chosen_labels.size(1)
        return {
            "concat_ids": packed_input_ids,
            "concat_masks": concat_masks,
            "chosen_labels": chosen_labels,
            "extras": extras
            }
