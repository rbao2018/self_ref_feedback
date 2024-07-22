# Copyright 2024 Musab Gultekin, LlamaFactory team, and OpenRLHF team.
#
# This code is based on the Musab Gultekin's functionary library:
# https://github.com/MeetKai/functionary/tree/main/functionary/train/packing
#
# and LlamaFactory:
# https://github.com/hiyouga/LLaMA-Factory/pull/4224/files
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License
#
# Copyright (c) 2023 Musab Gultekin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Tuple

import torch
import torch.nn.functional as F
import transformers.models


def get_unpad_data(attention_mask: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", int]:
    r"""
    Prepares the indices and seqlens for flash attn varlen function.
    Returns:
        indices: indices of non-masked tokens from the flattened sequence.
        cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.
        max_seqlen_in_batch: the largest seqlen in the current batch.
    e.g.
    ```
    [
        [1, 1, 2, 2, 2, 0],
        [1, 2, 2, 3, 3, 3],
    ]
    0 means padding
    ```
    ->
    ```
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    [0, 2, 5, 6, 8, 11] [leftpad 0 + cumsum([2, 3, 1, 2, 3])]
    3
    ```
    """
    seqlens_in_batch = attention_mask.sum(dim=-1)
    indices = torch.arange(0, torch.sum(attention_mask), device=attention_mask.device)
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def patch_for_block_diag_attn(model_type: str) -> None:
    print(f"patch_for_block_diag_attn: {model_type}")
    if model_type == "cohere":
        transformers.models.cohere.modeling_cohere._get_unpad_data = get_unpad_data
    elif model_type == "falcon":
        transformers.models.falcon.modeling_falcon._get_unpad_data = get_unpad_data
    elif model_type == "gemma":
        transformers.models.gemma.modeling_gemma._get_unpad_data = get_unpad_data
    elif model_type == "gemma2":
        transformers.models.gemma2.modeling_gemma2._get_unpad_data = get_unpad_data
    elif model_type == "llama":
        transformers.models.llama.modeling_llama._get_unpad_data = get_unpad_data
    elif model_type == "mistral":
        transformers.models.mistral.modeling_mistral._get_unpad_data = get_unpad_data
    elif model_type == "phi":
        transformers.models.phi.modeling_phi._get_unpad_data = get_unpad_data
    elif model_type == "phi3":
        transformers.models.phi3.modeling_phi3._get_unpad_data = get_unpad_data
    elif model_type == "qwen2":
        transformers.models.qwen2.modeling_qwen2._get_unpad_data = get_unpad_data
    elif model_type == "starcoder2":
        transformers.models.starcoder2.modeling_starcoder2._get_unpad_data = get_unpad_data
    else:
        raise Exception(f"Unsupported mode type {model_type} for patch_for_block_diag_attn!")
