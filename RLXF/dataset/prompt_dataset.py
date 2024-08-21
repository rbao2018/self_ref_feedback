from torch.utils.data import Dataset

def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.raw_dataset = dataset
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)
        self.prompt_max_len = getattr(self.strategy.args, "prompt_max_len", 2048)

        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        def _filter_data(examples):
            chunks = []
            from itertools import zip_longest
            padded_lists = zip_longest(*examples.values(), fillvalue=None)
            output_list = [dict(zip(examples.keys(), values)) for values in padded_lists]
            for data in output_list:
                prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
                chunks.append(self.tokenizer.encode(prompt))
            return {"input_ids": chunks}
        
        self.prompts = self.raw_dataset.map(
            _filter_data,
            batched=True, 
            keep_in_memory=True,
            num_proc=8,
            remove_columns=self.raw_dataset.column_names,
            desc=f"Filtering data to all less than prompt max len {self.prompt_max_len}"
        )
    
    def __len__(self):
        length = len(self.prompts)
        return length
    
    def __getitem__(self, idx):
        return self.prompts[idx]
    
    def collate_fn(self, item_list):
        return [x["input_ids"] for x in item_list]