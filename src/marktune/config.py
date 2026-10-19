"""Plain YAML recipes with explicit, typed command-line overrides."""

from copy import deepcopy
from pathlib import Path

import yaml

FIELDS = {
    "model": "name revision oracle dtype",
    "watermark": "parameters sigma gaussmark_sigma seed",
    "data": "directory seed min_prompt_tokens max_prompt_tokens",
    "training": "regularization seed total_steps learning_rate warmup_steps cosine_decay_fraction min_learning_rate_ratio betas "
    "weight_decay gradient_clip clip_ratio group_size prompt_batch_size mini_batch_size inner_steps response_tokens "
    "temperature ce_coefficient ce_batch_size ce_sequence_length save_steps",
    "evaluation": "num_samples prompt_tokens response_tokens temperature seed fpr bootstrap_samples confidence",
    "runtime": "device gpus micro_batch_size rollout_tensor_parallel_size rollout_memory_fraction",
}


def load_config(path, overrides=()):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = deepcopy(config)
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected section.field=value, got {override!r}")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"Unknown configuration field: {key}")
            target = target[part]
        if not isinstance(target, dict) or parts[-1] not in target:
            raise ValueError(f"Unknown configuration field: {key}")
        parsed = yaml.safe_load(value)
        if isinstance(target[parts[-1]], float) and isinstance(parsed, str):
            try:
                parsed = float(parsed)
            except ValueError:
                raise ValueError(f"{key} must be a number") from None
        target[parts[-1]] = parsed
    validate_config(config)
    return config


def validate_config(config):
    for section in ("model", "watermark", "data", "training", "evaluation", "baselines", "attacks", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    unknown_sections = config.keys() - set(FIELDS) - {"baselines", "attacks"}
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown_sections)}")
    for section, fields in FIELDS.items():
        expected = set(fields.split())
        unknown = config[section].keys() - expected
        missing = expected - config[section].keys()
        if unknown:
            raise ValueError(f"Unknown fields in {section}: {sorted(unknown)}")
        if missing:
            raise ValueError(f"Missing fields in {section}: {sorted(missing)}")
    positive = {
        "watermark": ["sigma", "gaussmark_sigma"],
        "training": [
            "total_steps",
            "learning_rate",
            "group_size",
            "prompt_batch_size",
            "mini_batch_size",
            "inner_steps",
            "response_tokens",
            "temperature",
            "ce_batch_size",
            "ce_sequence_length",
        ],
        "evaluation": ["num_samples", "prompt_tokens", "response_tokens", "temperature", "bootstrap_samples"],
        "runtime": ["gpus", "micro_batch_size", "rollout_tensor_parallel_size"],
    }
    for section, fields in positive.items():
        for field in fields:
            value = config[section].get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
                raise ValueError(f"{section}.{field} must be a positive finite number")
    integer_fields = {
        "training": [
            "total_steps",
            "group_size",
            "prompt_batch_size",
            "mini_batch_size",
            "inner_steps",
            "response_tokens",
            "ce_batch_size",
            "ce_sequence_length",
            "warmup_steps",
            "seed",
            "save_steps",
        ],
        "evaluation": ["num_samples", "prompt_tokens", "response_tokens", "bootstrap_samples", "seed"],
        "runtime": ["gpus", "micro_batch_size", "rollout_tensor_parallel_size"],
        "watermark": ["seed"],
        "data": ["seed", "min_prompt_tokens", "max_prompt_tokens"],
    }
    for section, fields in integer_fields.items():
        for field in fields:
            value = config[section].get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{section}.{field} must be a nonnegative integer")
    for field in ("fpr", "confidence"):
        if not 0 < config["evaluation"][field] < 1:
            raise ValueError(f"evaluation.{field} must lie in (0, 1)")
    training = config["training"]
    if training["regularization"] != "ce":
        raise ValueError("training.regularization must be ce for the main-body recipes")
    if not 0 < training["cosine_decay_fraction"] <= 1:
        raise ValueError("training.cosine_decay_fraction must lie in (0, 1]")
    if (
        training["warmup_steps"] + round(training["total_steps"] * training["cosine_decay_fraction"])
        > training["total_steps"]
    ):
        raise ValueError("Warmup plus cosine decay must fit within total training steps")
    if training["prompt_batch_size"] % training["mini_batch_size"]:
        raise ValueError("Prompt batch size must be divisible by mini batch size")
    if (
        training["mini_batch_size"]
        * training["group_size"]
        % (config["runtime"]["micro_batch_size"] * config["runtime"]["gpus"])
    ):
        raise ValueError("Expanded mini batch must be divisible by micro batch size times GPU count")
    if config["runtime"]["gpus"] % config["runtime"]["rollout_tensor_parallel_size"]:
        raise ValueError("GPU count must be divisible by rollout tensor parallel size")
    names = config["watermark"].get("parameters")
    if (
        not names
        or not isinstance(names, list)
        or not all(isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("watermark.parameters must list unique full parameter names")
    if training["ce_coefficient"] <= 0:
        raise ValueError("Main-body recipes require a positive human-text CE coefficient")
    if config["model"]["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("model.dtype must be float32, float16 or bfloat16")
    if (
        not 1
        <= config["data"]["min_prompt_tokens"]
        <= config["data"]["max_prompt_tokens"]
        <= training["ce_sequence_length"]
    ):
        raise ValueError("Prompt lengths must satisfy 1 <= min <= max <= CE sequence length")
    if not 0 <= training["min_learning_rate_ratio"] <= 1:
        raise ValueError("training.min_learning_rate_ratio must lie in [0, 1]")
    if training["group_size"] < 2:
        raise ValueError("GRPO requires at least two samples per prompt")


def add_config_arguments(parser):
    parser.add_argument("--config", required=True, help="YAML recipe")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
