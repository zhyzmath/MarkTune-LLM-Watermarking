"""RMS-scaled Gaussian watermarks and prompt-agnostic gradient detection."""

from pathlib import Path

import torch
import torch.nn.functional as F


def select_parameters(model, names):
    """Select exact Hugging Face parameter names, preserving their order."""
    if not names or len(set(names)) != len(names):
        raise ValueError("Watermark parameters must be nonempty and contain no duplicates")
    parameters = dict(model.named_parameters())
    missing = set(names) - parameters.keys()
    if missing:
        raise ValueError(f"Watermark parameters missing from model: {sorted(missing)}")
    return [(name, parameters[name]) for name in names]


def inject_watermark(model, parameters, sigma, seed):
    """Add sigma * RMS(W) * z and return the exact isotropic Gaussian key."""
    if not isinstance(sigma, (float, int)) or not 0 < sigma < float("inf"):
        raise ValueError("sigma must be positive and finite")
    selected = select_parameters(model, parameters)
    device, dtype = selected[0][1].device, selected[0][1].dtype
    if any(p.device != device or p.dtype != dtype for _, p in selected):
        raise ValueError("Watermark parameters must share device and dtype")
    rms_values = {name: parameter.float().square().mean().sqrt().item() for name, parameter in selected}
    if any(not 0 < rms < float("inf") for rms in rms_values.values()):
        raise ValueError("Every watermark parameter must have a positive finite RMS")
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(sum(p.numel() for _, p in selected), generator=generator, device=device, dtype=dtype)
    record = {
        "format_version": 1,
        "param_names": list(parameters),
        "sigma": float(sigma),
        "key": int(seed),
        "per_param_noise_iso": {},
        "rms_norms": {},
    }
    offset = 0
    with torch.no_grad():
        for name, parameter in selected:
            rms = rms_values[name]
            z = noise[offset : offset + parameter.numel()].view_as(parameter)
            record["per_param_noise_iso"][name] = z.detach().cpu().clone()
            record["rms_norms"][name] = rms
            parameter.add_(z * (sigma * rms))
            offset += parameter.numel()
    return record


def load_key(path):
    """Read a tensor-only key. Detection always uses the stored Gaussian draw."""
    record = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"param_names", "per_param_noise_iso", "rms_norms", "sigma"}
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError("Invalid watermark key file")
    names = record["param_names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("Invalid watermark parameter names in key file")
    if not all(name in record["per_param_noise_iso"] and name in record["rms_norms"] for name in names):
        raise ValueError("Watermark key is missing a parameter block")
    return record


class WatermarkDetector:
    """Score response tokens using the fixed clean reference and a persisted key.

    For blockwise RMS covariance, sigma cancels and the score is
    sum_l RMS_l <z_l, g_l> / sqrt(sum_l RMS_l^2 ||g_l||^2).
    Only response tokens enter the forward pass; prompts are never used.
    """

    def __init__(self, model, key_record):
        self.model = model
        self.parameters = select_parameters(model, key_record["param_names"])
        self.noise = []
        self.rms = []
        for name, parameter in self.parameters:
            noise = key_record["per_param_noise_iso"][name]
            if noise.shape != parameter.shape:
                raise ValueError(f"Key shape differs from reference parameter {name}")
            rms = float(key_record["rms_norms"][name])
            if not torch.isfinite(noise).all() or not 0 < rms < float("inf"):
                raise ValueError(f"Non-finite key or invalid RMS for {name}")
            self.noise.append(noise.to(device=parameter.device, dtype=torch.float32))
            self.rms.append(rms)

    def score(self, token_ids):
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.parameters[0][1].device).flatten()
        if ids.numel() < 2:
            return 0.0
        parameters = list(self.model.parameters())
        flags = [p.requires_grad for p in parameters]
        was_training = self.model.training
        try:
            self.model.eval()
            for p in parameters:
                p.requires_grad_(False)
            selected = [p for _, p in self.parameters]
            for p in selected:
                p.requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(input_ids=ids[None, :-1], use_cache=False).logits.float()
                log_likelihood = -F.cross_entropy(logits[0], ids[1:], reduction="sum")
                if not log_likelihood.requires_grad:
                    return 0.0
                gradients = torch.autograd.grad(log_likelihood, selected, allow_unused=True)
            numerator = torch.zeros((), device=ids.device, dtype=torch.float32)
            variance = torch.zeros_like(numerator)
            for gradient, noise, rms in zip(gradients, self.noise, self.rms):
                if gradient is not None:
                    scaled = gradient.float() * rms
                    numerator += (noise * scaled).sum()
                    variance += scaled.square().sum()
            if not torch.isfinite(numerator) or not torch.isfinite(variance):
                raise ValueError("Non-finite detection statistic")
            return (numerator / variance.sqrt()).item() if variance > 0 else 0.0
        finally:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)
            self.model.train(was_training)

    __call__ = score
