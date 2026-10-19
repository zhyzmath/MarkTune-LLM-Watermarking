"""Single-response generation with a common Hugging Face pipeline."""

import random


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_tokenizer(
    model_name, config, tokenizer_name=None, *, train=False, revision=None, tokenizer_revision=None
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config["runtime"]["device"]
    if revision is None and model_name == config["model"]["name"]:
        revision = config["model"].get("revision")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, config["model"]["dtype"]), trust_remote_code=False, revision=revision
    )
    model.to(device)
    model.train(train)
    if tokenizer_revision is None and not tokenizer_name:
        tokenizer_revision = revision if model_name == config["model"]["name"] else config["model"].get("revision")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or config["model"]["name"], revision=tokenizer_revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generation_options(config, tokenizer, baseline=None):
    from transformers import GenerationConfig

    length = config["evaluation"]["response_tokens"]
    options = {
        "generation_config": GenerationConfig(),
        "num_beams": 1,
        "max_new_tokens": length,
        "min_new_tokens": length,
        "do_sample": True,
        "temperature": config["evaluation"]["temperature"],
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    # Clear checkpoint-specific sampling restrictions and suppress EOS until the budget.
    if tokenizer.eos_token_id is not None:
        options["eos_token_id"] = tokenizer.eos_token_id
    if baseline is not None:
        from marktune.baselines import GumbelMax, SynthID

        options["logits_processor"] = [baseline]
        if isinstance(baseline, (GumbelMax, SynthID)):
            # These processors apply temperature internally, before the keyed transform.
            options["temperature"] = 1.0
        if isinstance(baseline, GumbelMax):
            options["do_sample"] = False
            options.pop("top_k")
    return options


def generate_record(model, tokenizer, record, config, baseline=None, *, timing=False):
    import torch
    import time

    prompt = record["prompt"]
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("Generation prompt must contain at least one token")
    # Prepared C4 rows have exactly 200 prompt tokens; ELI5 questions remain complete.
    if baseline is not None:
        baseline.reset()
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], device=device, dtype=torch.long)
    options = generation_options(config, tokenizer, baseline)
    if timing and device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **options)
    if timing and device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - start
    response = output[0, len(ids) :].tolist()
    result = {
        "prompt": prompt,
        "reference": record.get("reference", ""),
        "text": tokenizer.decode(response, skip_special_tokens=True),
        "token_ids": response,
    }
    if "id" in record:
        result["id"] = record["id"]
    if timing:
        result["generation_seconds"] = duration
    return result


def take_samples(records, count):
    from itertools import islice

    rows = list(islice(records, count))
    if len(rows) != count or count < 1:
        raise ValueError(f"Generation requires {count} prepared samples; found {len(rows)}")
    return rows


def run(args, config):
    from marktune.baselines import create_baseline
    from .io import read_records, write_jsonl

    seed_everything(config["evaluation"]["seed"])
    model_name = args.model or config["model"]["name"]
    if args.method == "kgw-d" and not args.model:
        model_name = config["baselines"].get("kgw_distilled_model")
        if not model_name:
            raise ValueError("KGW-D requires an explicit distilled checkpoint via --model")
    if args.method in ("marktune", "gaussmark") and not args.model:
        raise ValueError(f"{args.method} generation requires its watermarked checkpoint via --model")
    records = take_samples(read_records(args.input), config["evaluation"]["num_samples"])
    model, tokenizer = load_model_tokenizer(model_name, config, args.tokenizer)
    baseline = None
    if args.method in ("kgw-1", "kgw-2", "gumbel", "synthid"):
        vocab_size = model.config.vocab_size if args.method == "synthid" else len(tokenizer)
        baseline = create_baseline(
            args.method, vocab_size, config, device=next(model.parameters()).device, eos_token_id=tokenizer.eos_token_id
        )
    write_jsonl(
        args.output,
        (dict(generate_record(model, tokenizer, row, config, baseline), method=args.method) for row in records),
    )
