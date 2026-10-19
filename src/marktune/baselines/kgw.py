# Copyright 2024 THU-BPM MarkLLM. Licensed under Apache-2.0.
# Adapted from watermark/kgw/kgw.py; see third_party/markllm/NOTICE.
"""KGW with MarkLLM's product-prefix PRF and left green list."""

from math import prod, sqrt

import torch


class KGW:
    def __init__(self, vocab_size, *, gamma=0.5, delta=1.0, prefix_length=4, hash_key=42, device="cpu"):
        if not 0 < gamma < 1 or prefix_length < 1:
            raise ValueError("gamma must lie in (0, 1) and prefix_length must be positive")
        self.vocab_size = vocab_size
        self.gamma, self.delta = gamma, delta
        self.prefix_length, self.hash_key = prefix_length, hash_key
        self.device = torch.device(device)
        self.rng = torch.Generator(device=self.device).manual_seed(hash_key)
        self.prf = torch.randperm(vocab_size, device=self.device, generator=self.rng)

    def greenlist(self, ids):
        prefix = ids[-self.prefix_length :].tolist()
        prefix_hash = int(self.prf[prod(prefix) % self.vocab_size])
        self.rng.manual_seed((self.hash_key * prefix_hash) % self.vocab_size)
        return torch.randperm(self.vocab_size, device=self.device, generator=self.rng)[
            : int(self.vocab_size * self.gamma)
        ]

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] < self.prefix_length:
            return scores
        result = scores.clone()
        for index, ids in enumerate(input_ids):
            result[index, self.greenlist(ids)] += self.delta
        return result

    def reset(self):
        """KGW has no context history across calls."""

    def score(self, token_ids):
        ids = token_ids.flatten().to(self.device)
        count = len(ids) - self.prefix_length
        if count < 1:
            raise ValueError("Sequence must contain a token after the required prefix")
        green = sum(int(ids[index] in self.greenlist(ids[:index])) for index in range(self.prefix_length, len(ids)))
        return (green - self.gamma * count) / sqrt(count * self.gamma * (1 - self.gamma))
