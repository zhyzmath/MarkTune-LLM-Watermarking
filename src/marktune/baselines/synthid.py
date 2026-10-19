# Copyright 2024 THU-BPM MarkLLM. Licensed under Apache-2.0.
# Adapted from watermark/synthid/synthid.py and detector.py.
# See third_party/markllm/NOTICE.
"""Non-distortionary SynthID tournaments and mean-g detection."""

import torch

DEFAULT_KEYS = [
    654,
    400,
    836,
    123,
    340,
    443,
    597,
    160,
    57,
    29,
    590,
    639,
    13,
    715,
    468,
    990,
    966,
    226,
    324,
    585,
    118,
    504,
    421,
    521,
    129,
    669,
    732,
    225,
    90,
    960,
]


class SynthID:
    def __init__(
        self,
        vocab_size,
        *,
        prefix_length=4,
        keys=None,
        sampling_table_size=65536,
        sampling_table_seed=0,
        context_history_size=1024,
        temperature=0.7,
        eos_token_id=None,
        device="cpu",
    ):
        if prefix_length < 1 or temperature <= 0 or context_history_size < 1:
            raise ValueError("prefix_length, temperature and context_history_size must be positive")
        self.device = torch.device(device)
        self.vocab_size, self.prefix_length = vocab_size, prefix_length
        self.temperature, self.eos_token_id = temperature, eos_token_id
        self.context_history_size = context_history_size
        self.keys = torch.tensor(DEFAULT_KEYS if keys is None else keys, device=self.device)
        if not len(self.keys):
            raise ValueError("SynthID requires at least one tournament key")
        rng = torch.Generator(device=self.device).manual_seed(sampling_table_seed)
        self.table = torch.randint(0, 2, (sampling_table_size,), generator=rng, device=self.device)
        self.reset()

    def reset(self):
        self.history = None

    @staticmethod
    def hash_tokens(tokens, initial=None):
        value = torch.ones(tokens.shape[:-1], dtype=torch.long, device=tokens.device) if initial is None else initial
        for token in tokens.unbind(-1):
            value = (value + token) * 6364136223846793005 + 1
        return value

    def g_values(self, ngrams):
        hashes = self.hash_tokens(ngrams)
        hashes = (hashes[..., None] + self.keys) * 6364136223846793005 + 1
        return self.table[hashes % len(self.table)]

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] < self.prefix_length:
            return scores
        contexts = input_ids[:, -self.prefix_length :].to(self.device)
        context_hashes = self.hash_tokens(contexts)
        if self.history is None or self.history.shape[0] != input_ids.shape[0]:
            self.history = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=self.device)
        repeated = (self.history == context_hashes[:, None]).any(dim=-1, keepdim=True)
        self.history = torch.cat([context_hashes[:, None], self.history], dim=1)[:, : self.context_history_size]
        tokens = torch.arange(scores.shape[-1], device=self.device)
        hashes = (context_hashes[:, None] + tokens) * 6364136223846793005 + 1
        hashes = (hashes[:, :, None] + self.keys) * 6364136223846793005 + 1
        values = self.table[hashes % len(self.table)]
        unmodified = scores.float() / self.temperature
        probabilities = torch.softmax(unmodified, dim=-1)
        for layer in range(values.shape[-1]):
            g = values[:, :, layer]
            # Finite-precision drift must not turn the g=0 multiplier negative.
            mass = (g * probabilities).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            probabilities = probabilities * (1 + g - mass)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        # Exact zeros, including tokens suppressed by earlier processors, stay -inf.
        watermarked = torch.log(probabilities)
        return torch.where(repeated, unmodified, watermarked)

    def context_mask(self, ids):
        contexts = ids[:-1].unfold(0, self.prefix_length, 1)
        hashes = self.hash_tokens(contexts).tolist()
        history, mask = [], []
        for value in hashes:
            mask.append(value not in history)
            history.insert(0, value)
            del history[self.context_history_size :]
        return torch.tensor(mask, device=ids.device, dtype=torch.bool)

    def score(self, token_ids):
        ids = token_ids.flatten().to(self.device)
        if self.eos_token_id is not None:
            eos = torch.where(ids == self.eos_token_id)[0]
            if len(eos):
                ids = ids[: int(eos[0])]
        if len(ids) <= self.prefix_length:
            raise ValueError("Sequence must contain a token after the required prefix")
        ngrams = ids.unfold(0, self.prefix_length + 1, 1)
        values = self.g_values(ngrams)
        mask = self.context_mask(ids)
        return float(values[mask].float().mean())
