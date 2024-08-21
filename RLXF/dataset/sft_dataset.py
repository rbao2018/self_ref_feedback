from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.datasets.utils import zero_pad_sequences


def preprocess_data(data, input_template=None, input_key="messages", output_key="assistant", apply_chat_template=False):
    prompt, response = [], []
    if apply_chat_template:
        messages = []
        for message in data[input_key]:
            messages.append(message)
            if message.get("role", None) == output_key:
                response.append(apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
                prompt.append(apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True))
                messages = []
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""

    assert len(prompt) == len(response), f"prompt length: {len(prompt)}, response length: {len(response)}"

    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        ignore_index = -100,
        num_processors=8  # Specify the number of processors you want to use
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.ignore_index = ignore_index

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        

    def process_data(self, data):
        prompt, response = preprocess_data(
            data,
            self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=self.apply_chat_template,
        )
        return {"prompt": prompt, "response": response}
  

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        input_ids = []
        labels = []
        for (p,r) in zip(prompt, response):
            input_token = self.tokenizer.encode(r, add_special_tokens=False)
            input_token[-1] = self.tokenizer.eos_token_id
            input_ids.extend(input_token)
            if not self.pretrain_mode:
                prompt_token = self.tokenizer.encode(p, add_special_tokens=False)
                label = [self.ignore_index] * len(prompt_token) + input_token[len(prompt_token):]
                labels.extend(label)
            else:
                labels.extend(input_token)
        return input_ids[ :self.max_length], labels[ :self.max_length]

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        labels = []
        for input_id, label in item_list:
            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            attention_masks.append(torch.ones_like(input_id))
            labels.append(torch.tensor(label, dtype=torch.long))

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right", 0)
        labels = zero_pad_sequences(labels, "right", self.ignore_index)
        
        return input_ids, labels, attention_masks

    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        packed_labels = []
        for input_id, label in item_list:
            packed_input_ids.extend(input_id)
            packed_attention_masks.append(torch.ones(len(input_id), dtype=torch.long))
            packed_labels.extend(label)
        packed_input_ids = torch.tensor(packed_input_ids, dtype=torch.long).unsqueeze(0)
        packed_attention_masks = zero_pad_sequences(packed_attention_masks, "right", 0)
        packed_labels = torch.tensor(packed_labels, dtype=torch.long).unsqueeze(0)

        return packed_input_ids, packed_labels, packed_attention_masks