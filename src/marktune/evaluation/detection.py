"""Prompt-agnostic watermark scoring of visible response text."""


def score_records(records, tokenizer, detector):
    import torch

    for row in records:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        # Visible-text tokenization is shared across methods and text attacks.
        score = float(detector(torch.tensor(ids, dtype=torch.long)))
        yield dict(row, token_ids=ids, score=score)


def reference_identity(key, requested, default):
    identity = key.get("reference_model")
    reference = requested or identity or default
    if identity and reference != identity:
        raise ValueError("The clean reference model must match the identity stored in the watermark key")
    return reference, key.get("reference_revision")


def load_detector(args, config):
    from .generation import load_model_tokenizer
    from marktune.baselines import create_baseline
    from transformers import AutoTokenizer

    if args.method in ("marktune", "gaussmark"):
        from marktune.watermark import WatermarkDetector, load_key

        if not args.key:
            raise ValueError("Weight-embedded detection requires --key")
        key = load_key(args.key)
        reference, revision = reference_identity(key, args.reference_model, config["model"]["name"])
        model, tokenizer = load_model_tokenizer(
            reference,
            config,
            args.tokenizer or reference,
            revision=revision,
            tokenizer_revision=revision if not args.tokenizer else None,
        )
        detector = WatermarkDetector(model, key)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or config["model"]["name"],
            trust_remote_code=False,
            revision=config["model"].get("revision") if not args.tokenizer else None,
        )
        detector = create_baseline(
            args.method,
            len(tokenizer),
            config,
            device=config["runtime"]["device"],
            eos_token_id=tokenizer.eos_token_id,
            special_token_ids=tokenizer.all_special_ids,
        ).score
    return tokenizer, detector


def matched_records(positive, null):
    if len(positive) != len(null):
        raise ValueError("Positive and clean records must have matching sample counts")
    for pos, neg in zip(positive, null):
        if pos["prompt"] != neg["prompt"] or ("id" in pos and pos["id"] != neg.get("id")):
            raise ValueError("Positive and clean records must be paired by id and prompt")
        yield dict(pos, clean_score=neg["score"])


def run(args, config):
    from .io import read_records, write_jsonl

    tokenizer, detector = load_detector(args, config)
    records = score_records(read_records(args.input), tokenizer, detector)
    if args.clean_input:
        positive = list(records)
        clean = list(score_records(read_records(args.clean_input), tokenizer, detector))
        records = matched_records(positive, clean)
    write_jsonl(args.output, records)
