"""Paper metrics with explicit estimands and paired bootstrap uncertainty."""

from collections import Counter
import math

import numpy as np


def roc_curve(null, positive):
    values = np.concatenate([null, positive])
    labels = np.concatenate([np.zeros(len(null)), np.ones(len(positive))])
    order = np.argsort(-values, kind="stable")
    values, labels = values[order], labels[order]
    endpoints = np.r_[np.flatnonzero(np.diff(values)), len(values) - 1]
    tp = np.cumsum(labels)[endpoints]
    fp = 1 + endpoints - tp
    return np.r_[0.0, fp / len(null)], np.r_[0.0, tp / len(positive)], np.r_[np.inf, values[endpoints]]


def detection_metrics(null, positive, *, fpr=0.01, bootstrap_samples=500, confidence=0.9, seed=23333):
    null, positive = np.asarray(null, dtype=float), np.asarray(positive, dtype=float)
    if null.ndim != 1 or positive.ndim != 1 or len(null) != len(positive) or len(null) == 0:
        raise ValueError("Detection requires nonempty, equally sized paired score vectors")
    if not np.isfinite(null).all() or not np.isfinite(positive).all():
        raise ValueError("Detection scores must be finite")
    if not 0 <= fpr <= 1 or not 0 < confidence < 1 or bootstrap_samples < 0:
        raise ValueError("Invalid FPR, confidence, or bootstrap count")
    false, true, thresholds = roc_curve(null, positive)
    widths = np.diff(false)
    auc = float(np.sum(widths * (true[1:] + true[:-1]) / 2))
    valid = np.flatnonzero(false <= fpr)
    index = valid[-1]
    result = {
        "auc": auc,
        "tpr_at_fpr": float(np.interp(fpr, false, true)),
        "target_fpr": fpr,
        "tpr_estimator": "linear ROC interpolation",
        "empirical_tpr": float(true[index]),
        "empirical_fpr": float(false[index]),
        "empirical_threshold": float(thresholds[index]) if math.isfinite(thresholds[index]) else None,
        "null_samples": len(null),
        "positive_samples": len(positive),
        "null_fpr_resolution": 1 / len(null),
    }
    if bootstrap_samples:
        rng = np.random.default_rng(seed)
        aucs, tprs = [], []
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, len(null), len(null))
            x, y, _ = roc_curve(null[indices], positive[indices])
            aucs.append(float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2)))
            tprs.append(float(np.interp(fpr, x, y)))
        tail = (1 - confidence) / 2
        result["bootstrap"] = {
            "samples": bootstrap_samples,
            "confidence": confidence,
            "seed": seed,
            "unit": "matched prompt pair",
            "tpr_mean": float(np.mean(tprs)),
            "tpr_ci": np.quantile(tprs, [tail, 1 - tail]).tolist(),
            "auc_mean": float(np.mean(aucs)),
            "auc_ci": np.quantile(aucs, [tail, 1 - tail]).tolist(),
        }
    return result


def repetition_metrics(texts):
    fractions, total, repeated = [], 0, 0
    for text in texts:
        words = text.split()
        counts = Counter(tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2)))
        count = sum(counts.values())
        duplicates = sum(value - 1 for value in counts.values())
        fractions.append(duplicates / count if count else 0.0)
        total += count
        repeated += duplicates
    return {
        "seq_rep_3": float(np.mean(fractions)) if fractions else 0.0,
        "seq_rep_3_micro": repeated / total if total else 0.0,
    }


def response_perplexity(model, prompt_tokens, response_tokens):
    """Oracle PPL of the response, including its first token given the prompt."""
    import torch
    import torch.nn.functional as F

    if not response_tokens:
        raise ValueError("Perplexity requires a nonempty response")
    tokens = list(prompt_tokens) + list(response_tokens)
    if len(tokens) < 2:
        raise ValueError("Perplexity requires at least two total tokens")
    device = next(model.parameters()).device
    ids = torch.tensor([tokens], device=device, dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        start = max(0, len(prompt_tokens) - 1)
        loss = F.cross_entropy(
            logits[:, start:-1].float().reshape(-1, logits.shape[-1]), ids[:, start + 1 :].reshape(-1)
        )
    return math.exp(float(loss))


def text_perplexity(model, tokenizer, prompt, response):
    """Jointly tokenize prompt and response, scoring response-overlapping labels."""
    import torch
    import torch.nn.functional as F

    encoded = tokenizer(
        prompt + response, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True, truncation=False
    )
    offsets = torch.as_tensor(encoded.pop("offset_mapping"))
    ids = torch.as_tensor(encoded["input_ids"])
    attention = torch.as_tensor(encoded["attention_mask"])
    mask = (offsets[..., 1] > len(prompt)) & (offsets[..., 1] > offsets[..., 0]) & attention.bool()
    mask = mask[:, 1:]
    if not mask.any():
        raise ValueError("Oracle tokenization produced no predictable response labels")
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(input_ids=ids.to(device), attention_mask=attention.to(device)).logits
        losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:].to(device), reduction="none")
    return math.exp(float(losses[mask.to(device)].mean()))


def quality_metrics(records, oracle=None, tokenizer=None, *, mauve=False, device="cuda", seed=23333):
    records = list(records)
    if not records:
        raise ValueError("Quality evaluation requires samples")
    texts = [row["text"] for row in records]
    result = repetition_metrics(texts)
    result["samples"] = len(records)
    if oracle is not None:
        values = [text_perplexity(oracle, tokenizer, row["prompt"], row["text"]) for row in records]
        result["perplexity_mean"] = float(np.mean(values))
        result["perplexity_std"] = float(np.std(values, ddof=0))
    if mauve:
        from mauve import compute_mauve

        if any(not row.get("reference") for row in records):
            raise ValueError("MAUVE requires a human reference for every sample")
        if len(records) < 20:
            raise ValueError("MAUVE requires at least 20 samples for the paper bucket rule")
        value = compute_mauve(
            p_text=[row["reference"] for row in records],
            q_text=texts,
            num_buckets=len(records) // 20,
            device_id=0 if device.startswith("cuda") else -1,
            seed=seed,
        )
        result["mauve"] = float(value.mauve)
    return result
