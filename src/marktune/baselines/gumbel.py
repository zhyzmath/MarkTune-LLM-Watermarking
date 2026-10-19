# Copyright 2024 THU-BPM MarkLLM. Licensed under Apache-2.0.
# Adapted from watermark/exp/exp.py; see third_party/markllm/NOTICE.
"""Gumbel-max sampling with the EXP baseline's deterministic prefix seed."""

from math import log, prod

import torch


class GumbelMax:
    def __init__(self, vocab_size, *, prefix_length=4, seed=42, temperature=0.7):
        if prefix_length < 1 or temperature <= 0:
            raise ValueError("prefix_length and temperature must be positive")
        self.vocab_size = vocab_size
        self.prefix_length, self.seed, self.temperature = prefix_length, seed, temperature

    def uniforms(self, ids):
        prefix = ids[-self.prefix_length :].tolist()
        generator = torch.Generator().manual_seed(self.seed * (prod(prefix) % self.vocab_size))
        return torch.rand(self.vocab_size, generator=generator, dtype=torch.float32)

    def __call__(self, input_ids, scores):
        """Choose the exponential-race winner; model.generate uses greedy mode."""
        if input_ids.shape[-1] < self.prefix_length:
            return scores
        result = torch.full_like(scores, -torch.inf)
        probs = torch.softmax(scores[:, : self.vocab_size].float() / self.temperature, dim=-1).clamp_min(1e-9)
        for index, ids in enumerate(input_ids):
            uniforms = self.uniforms(ids).to(scores.device).clamp_min(1e-9)
            # log(u)/p is monotonic in u**(1/p), without underflow.
            winner = torch.argmax(torch.log(uniforms) / probs[index])
            result[index, winner] = 0
        return result

    def reset(self):
        """The prefix determines every random draw independently."""

    def score(self, token_ids):
        """Return -log Gamma survival probability, with higher values stronger."""
        from scipy.special import gammaln, logsumexp
        import numpy as np

        ids = token_ids.flatten().cpu()
        count = len(ids) - self.prefix_length
        if count < 1:
            raise ValueError("Sequence must contain a token after the required prefix")
        total = sum(
            -log(max(1e-12, 1 - float(self.uniforms(ids[:i])[ids[i]]))) for i in range(self.prefix_length, len(ids))
        )
        if total == 0:
            return 0.0
        terms = np.arange(count)
        return float(total - logsumexp(terms * log(total) - gammaln(terms + 1)))
