from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.datasets.utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template):
    prompts = []
    outputs = []
    chosen = ""
    reject = ""
    if exist_and_not_none(data, "conversations"):
        for con in data["conversations"]:
            temp_input = input_template.format(con["instruction"])
            prompts.append(temp_input)
            outputs.append(con["output"])

    elif exist_and_not_none(data, "instruction"):
        prompts.append(input_template.format(data["instruction"]))
        outputs.append(data["output"])
    else:
        raise ValueError("Unknown reward dataset")
    
    if exist_and_not_none(data, "ranked_responses"):
        chosen = data["ranked_responses"][0]
        reject = data["ranked_responses"][1]

    # margin loss
    margin = float(data["margin"]) if exist_and_not_none(data, "margin") else 0.0

    return prompts, outputs, chosen, reject, margin


class RLHFDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for rlhf reward modeling and RL
        self.tokenizer: self.tokenizer for reward model
        self.strategy: deepspeed strategy for default
        self.zero_out_input: zero out input tokens loss for reward modeling
        self.ignor_index: ignore index for zero out input tokens
        self.input_template: input template for reward model
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        strategy,
        zero_out_input = True,
        ignor_index = -100,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.prompts = []
        self.outputs = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.zero_out_input = zero_out_input
        self.ignor_index = ignor_index

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, output, chosen, reject, margin = preprocess_data(data, input_template)
            self.prompts.append(prompt)
            self.outputs.append(output)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(margin)

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, output, chosen, reject, margin = self.prompts[idx], self.outputs[idx], self.chosens[idx], self.rejects[idx], self.margins[idx]
        input_ids = []
        labels = []
        chosen_ids = []
        reject_ids = []

        for i, (query, output) in enumerate(zip(prompt, output)):
            if i == len(prompt) - 1:
                query_tokens = self.tokenizer.encode(query)
                chosen_output = query + chosen + self.tokenizer.eos_token
                tokens = self.tokenizer.encode(chosen_output)
                tokens[-1] = self.tokenizer.eos_token_id
                chosen_ids = input_ids + tokens
                if self.zero_out_input:
                    tokens[:len(query_tokens)] = [self.ignor_index] * len(query_tokens)
                labels.extend(tokens)
                reject_output = query + reject + self.tokenizer.eos_token
                tokens = self.tokenizer.encode(reject_output)
                tokens[-1] = self.tokenizer.eos_token_id
                reject_ids = input_ids + tokens
                input_ids.extend(query_tokens)
            else:
                temp = query + output + self.tokenizer.eos_token
                tokens = self.tokenizer.encode(temp)
                tokens[-1] = self.tokenizer.eos_token_id
                input_ids.extend(tokens)
                if self.zero_out_input:
                    prefix_len = len(self.tokenizer.encode(query))
                    tokens[:prefix_len] = [self.ignor_index] * prefix_len
                labels.extend(tokens)
     
        return (input_ids, labels, chosen_ids, reject_ids, margin)

    def collate_fn(self, item_list):
        input_ids = []
        labels = []
        chosen_ids = []
        reject_ids = []
        margins = []
        for input_id, label, chosen_id, reject_id, margin in item_list:
            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))
            chosen_ids.append(torch.tensor(chosen_id, dtype=torch.long))
            reject_ids.append(torch.tensor(reject_id, dtype=torch.long))
            margins.append(margin)
        # zero pad using tokenizer unk_token_id
        pad_token_id = self.tokenizer.unk_token_id
        input_ids = zero_pad_sequences(input_ids, side="right", value=pad_token_id)
        labels = zero_pad_sequences(labels, side="right", value=-100)
        concat_ids = zero_pad_sequences(chosen_ids + reject_ids, side="right", value=pad_token_id)
        chosen_ids = zero_pad_sequences(chosen_ids, side="right", value=pad_token_id)
        reject_ids = zero_pad_sequences(reject_ids, side="right", value=pad_token_id)
        concat_masks = torch.ne(concat_ids, pad_token_id).long()
        return {
            "prompt": input_ids,
            "chosen_labels": labels,
            "chosen_ids": chosen_ids,
            "reject_ids": reject_ids,
            "concat_ids": concat_ids,
            "concat_masks": concat_masks,
            "margin": torch.tensor(margins, dtype=torch.float32)
        }