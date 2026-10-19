"""Batch-one generation and prompt-agnostic detection wall-clock benchmark."""

from itertools import islice
import time


def benchmark_records(records, model, tokenizer, detector, config, *, baseline=None, samples=100, warmup=2):
    import numpy as np
    import torch
    from .generation import generate_record

    rows = list(islice(records, samples + warmup))
    if samples < 1 or warmup < 0 or len(rows) != samples + warmup:
        raise ValueError("Benchmark requires measured samples plus additional warmup prompts")
    device = next(model.parameters()).device
    measured = []
    # Additional prompts warm up kernels; the first N prompts remain the measured set.
    schedule = [(row, False) for row in rows[samples:]] + [(row, True) for row in rows[:samples]]
    for row, include in schedule:
        result = generate_record(model, tokenizer, row, config, baseline, timing=True)
        ids = torch.tensor(tokenizer.encode(result["text"], add_special_tokens=False), dtype=torch.long)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        score = float(detector(ids))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start
        if include:
            measured.append(
                {
                    "id": row.get("id"),
                    "generation_seconds": result["generation_seconds"],
                    "detection_seconds": duration,
                    "score": score,
                }
            )
    summary = {
        "samples": samples,
        "warmup_prompts": warmup,
        "batch_size": 1,
        "prompt_tokens": config["evaluation"]["prompt_tokens"],
        "response_tokens": config["evaluation"]["response_tokens"],
        "device": str(device),
        "measurements": measured,
    }
    if device.type == "cuda":
        summary["gpu"] = torch.cuda.get_device_name(device)
    for name in ("generation_seconds", "detection_seconds"):
        values = [row[name] for row in measured]
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_std"] = float(np.std(values, ddof=0))
    return summary


def run(args, config):
    from marktune.baselines import create_baseline
    from .generation import load_model_tokenizer, seed_everything
    from .detection import load_detector
    from .io import read_records, write_json

    seed_everything(config["evaluation"]["seed"])
    model_name = args.model or config["model"]["name"]
    if args.method in ("marktune", "gaussmark") and not args.model:
        raise ValueError("Weight-embedded benchmarking requires a checkpoint via --model")
    model, tokenizer = load_model_tokenizer(model_name, config, args.tokenizer)
    baseline = None
    if args.method in ("kgw-1", "kgw-2", "gumbel", "synthid"):
        vocab_size = model.config.vocab_size if args.method == "synthid" else len(tokenizer)
        baseline = create_baseline(
            args.method, vocab_size, config, device=next(model.parameters()).device, eos_token_id=tokenizer.eos_token_id
        )
        detector = baseline.score
    else:
        _, detector = load_detector(args, config)
    result = benchmark_records(
        read_records(args.input),
        model,
        tokenizer,
        detector,
        config,
        baseline=baseline,
        samples=args.samples,
        warmup=args.warmup,
    )
    result["method"] = args.method
    write_json(args.output, result)
