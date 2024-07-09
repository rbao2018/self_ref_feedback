from typing import Callable
from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.datasets.utils import exist_and_not_none

def preprocess_data(data, input_template):
    prompts = []
    outputs = []
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
    
    assert len(prompts) == len(outputs), "Prompts and outputs should have the same length"
    outputs[-1] = ""
    return prompts, outputs
    
class PromptDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for rlhf reward modeling and RL
        self.tokenizer: self.tokenizer for reward model
        self.strategy: deepspeed strategy for default
        self.zero_out_input: zero out input tokens loss for reward modeling
        self.prompt_max_len: 
        self.input_template: input template for reward model
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.raw_dataset = dataset
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.prompt_max_len = strategy.args.prompt_max_len
        self.input_template = input_template
        self.filter_dataset = self.raw_dataset.map(
            self._filter_data,
            batched=True, 
            keep_in_memory=True,
            num_proc=8,
            remove_columns=self.raw_dataset.column_names,
            desc=f"Filtering data to all less than prompt max len {self.prompt_max_len}"
        )
    
    def __len__(self):
        length = len(self.filter_dataset)
        return length
    
    def __getitem__(self, idx):
        return self.filter_dataset[idx]
    
    def _filter_data(self, examples):
        chunks = []
        from itertools import zip_longest
        padded_lists = zip_longest(*examples.values(), fillvalue=None)
        output_list = [dict(zip(examples.keys(), values)) for values in padded_lists]
        for tmp_data in output_list:
            prompt, output = preprocess_data(tmp_data, self.input_template)
            input_ids = []
            for i, (query, output) in enumerate(zip(prompt, output)):
                if i == len(prompt) - 1:
                    query_tokens = self.tokenizer.encode(query)
                    input_ids.extend(query_tokens)
                else:
                    temp = query + output + self.tokenizer.eos_token
                    tokens = self.tokenizer.encode(temp)
                    tokens[-1] = self.tokenizer.eos_token_id
                    input_ids.extend(tokens)
            if len(input_ids) <= self.prompt_max_len:
                chunks.append(input_ids)
        return {"input_ids": chunks}
    
    def collate_fn(self, item_list):
        return [x["input_ids"] for x in item_list]
    