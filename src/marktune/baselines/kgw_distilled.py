# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
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
"""The one-token KGW teacher detector for Gu et al.'s distilled checkpoint.

Reimplements the teacher green-list rule from watermark-learnability commit
f84f70f9fc30eba5e39bf1889376d61de5f2ee59, specifically:
https://github.com/chenchenygu/watermark-learnability/blob/f84f70f9fc30eba5e39bf1889376d61de5f2ee59/watermarks/kgw/kgw_watermark.py

The upstream teacher uses simple_1, CPU permutations, and tokenizer vocabulary
size. BOS, EOS, PAD, and UNK are removed as both prefix and target tokens after
permutation selection. No dense vocabulary-squared table is materialized here.
"""

from math import sqrt

import torch


class KGWDistilled:
    """Response-only detection with the distilled model's exact teacher PRF."""

    def __init__(self, vocab_size, *, gamma=0.25, hash_key=15485863, special_token_ids=()):
        if type(vocab_size) is not int or vocab_size < 2 or not 0 < gamma < 1:
            raise ValueError("vocab_size must be at least two and gamma must lie in (0, 1)")
        if type(hash_key) is not int or not 0 < hash_key < 2**64:
            raise ValueError("hash_key must be a positive 64-bit integer")
        self.vocab_size, self.gamma, self.hash_key = vocab_size, gamma, hash_key
        self.prefix_length = 1
        self.device = torch.device("cpu")
        self.special_token_ids = frozenset(int(token) for token in special_token_ids)
        if any(token < 0 or token >= vocab_size for token in self.special_token_ids):
            raise ValueError("Special token IDs must lie within the tokenizer vocabulary")
        self.rng = torch.Generator(device="cpu")

    def greenlist(self, token_ids):
        ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu").flatten()
        if not len(ids):
            raise ValueError("KGW-D requires a one-token prefix")
        prefix = int(ids[-1])
        if prefix < 0 or prefix >= self.vocab_size:
            raise ValueError("Prefix token is outside the tokenizer vocabulary")
        if prefix in self.special_token_ids:
            return torch.empty(0, dtype=torch.long)
        self.rng.manual_seed((self.hash_key * prefix) % (2**64 - 1))
        selected = torch.randperm(self.vocab_size, generator=self.rng)[: int(self.gamma * self.vocab_size)]
        # The teacher removes these columns after selection, without replacement.
        for special in self.special_token_ids:
            selected = selected[selected != special]
        return selected

    def score(self, token_ids):
        ids = torch.as_tensor(token_ids, dtype=torch.long, device="cpu").flatten()
        count = len(ids) - self.prefix_length
        if count < 1:
            raise ValueError("Sequence must contain a token after the one-token prefix")
        if bool(((ids < 0) | (ids >= self.vocab_size)).any()):
            raise ValueError("Response token is outside the tokenizer vocabulary")
        green = sum(int(ids[index] in self.greenlist(ids[index - 1 : index])) for index in range(1, len(ids)))
        return (green - self.gamma * count) / sqrt(count * self.gamma * (1 - self.gamma))
