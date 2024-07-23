import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from .experience_maker import Experience


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (A)
    returns: (A)
    advatanges: (A)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            batch_kwargs[i]["info"][k] = vv.item()

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


# def make_experience_batch(items: List[BufferItem]) -> Experience:
#     kwargs = {}
#     keys = (
#         "sequences",
#         "action_log_probs",
#         "values",
#         "returns",
#         "advantages",
#         "attention_mask",
#         "action_mask",
#     )
#     for key in keys:
#         vals = [getattr(item, key) for item in items]
#         batch_data = zero_pad_sequences(vals, "left")
#         kwargs[key] = batch_data

#     kwargs["info"] = {}
#     for key in items[0].info.keys():
#         vals = torch.tensor([item.info[key] for item in items])
#         kwargs["info"][key] = vals
#     return Experience(**kwargs)


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to False.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.sample_batch_size = args.micro_train_batch_size
        self.limit = args.buffer_limit 
        # buffer_limit <= 0 means always use the latest experience, otherwise use some history
        self.cpu_offload = args.buffer_cpu_offload
        self.buffer_history_ratio = args.buffer_history_ratio
        self.packing_samples = args.packing_samples
        self.target_device = torch.cuda.current_device()
        self.historys: List[BufferItem] = []
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        self.historys.extend(items)
        if self.limit > 0:
            self.items.extend(random.sample(self.historys, int(len(items) * self.buffer_history_ratio)))
            self.items.extend(random.sample(items, int(len(items) * (1 - self.buffer_history_ratio))))
            samples_to_remove = len(self.historys) - self.limit
            if samples_to_remove > 0:
                self.historys = self.historys[samples_to_remove:]
        else:
            self.items.extend(items)

    def clear(self) -> None:
        self.items.clear()

    # @torch.no_grad()
    # def sample(self) -> Experience:
    #     items = random.sample(self.items, self.sample_batch_size)
    #     experience = make_experience_batch(items)
    #     if self.cpu_offload:
    #         experience.to_device(self.target_device)
    #     return experience

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, items) -> Experience:
        if self.packing_samples:
            raise NotImplementedError
        kwargs = {}
        keys = (
            "sequences",
            "action_log_probs",
            "values",
            "returns",
            "advantages",
            "attention_mask",
            "action_mask",
        )
        for key in keys:
            vals = [getattr(item, key) for item in items]
            batch_data = zero_pad_sequences(vals, "left")
            kwargs[key] = batch_data

        kwargs["info"] = {}
        for key in items[0].info.keys():
            vals = torch.tensor([item.info[key] for item in items])
            kwargs["info"][key] = vals
        return Experience(**kwargs)

    def normalize(self, attribute: str, strategy) -> None:
        assert attribute == "advantages"
        items = []
        action_masks = []
        for item in self:
            items.append(getattr(item, attribute))
            action_masks.append(item.action_mask)

        items_vector = torch.cat([item.flatten() for item in items]).float()
        action_masks_vector = torch.cat([mask.flatten() for mask in action_masks])

        # for DP
        # mean
        sum_and_count = torch.tensor([items_vector.sum(), action_masks_vector.sum()], device=items_vector.device)
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")
        mean = all_sum / all_count
        # std
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
        all_std = strategy.all_reduce(std, "sum")
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()

        for i, item in enumerate(self):
            setattr(item, attribute, (items[i] - mean) * rstd)
