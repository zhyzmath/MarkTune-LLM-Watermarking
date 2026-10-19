"""Prompt-agnostic MarkTune rewards colocated with a GPU-owning actor worker."""

import torch

from marktune.watermark import WatermarkDetector, load_key

from .abstract import AbstractRewardManager
from .registry import register


class GradientReward:
    def __init__(self, ref_model_path, key_path, dtype="bfloat16", device="cuda", revision=None):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            ref_model_path, revision=revision, torch_dtype=getattr(torch, dtype), trust_remote_code=False
        ).to(device)
        self.detector = WatermarkDetector(model, load_key(key_path))

    def __call__(self, batch):
        scores = []
        for response, length in zip(batch["responses"], batch["response_lengths"]):
            scores.append(self.detector.score(response[:int(length)]))
        return torch.tensor(scores, dtype=torch.float32)


@register("marktune")
class MarkTuneRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer=None, num_examine=0, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.kwargs = kwargs
        self.worker_group = None

    def bind_actor_worker_group(self, actor_worker_group):
        if actor_worker_group is None:
            raise ValueError("MarkTune reward requires a GPU-owning actor worker group")
        actor_worker_group.init_marktune_reward(self.kwargs)
        self.worker_group = actor_worker_group

    def __call__(self, data, return_dict=False):
        if self.worker_group is None:
            raise RuntimeError("MarkTune reward has not been bound to actor workers")
        responses = data.batch["responses"]
        attention = data.batch["attention_mask"][:, -responses.shape[1]:]
        lengths = attention.sum(-1).long()
        batch = {"responses": responses.cpu(), "response_lengths": lengths.cpu()}
        values = self.worker_group.compute_marktune_reward(batch).to(responses.device)
        rewards = torch.zeros_like(responses, dtype=torch.float32)
        valid = lengths > 0
        rows = torch.arange(responses.shape[0], device=responses.device)[valid]
        rewards[rows, lengths[valid] - 1] = values[valid]
        return {"reward_tensor": rewards} if return_dict else rewards
