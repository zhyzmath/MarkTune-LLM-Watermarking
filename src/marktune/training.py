"""Initialize an actor and launch the CE-regularized verl GRPO trainer."""

import importlib.util
import json
from pathlib import Path

import yaml


def build_verl_config(config, initialized, output):
    """Translate a public recipe to a verl overlay without loading any model."""
    initialized, output = Path(initialized).resolve(), Path(output).resolve()
    data_dir = Path(config["data"]["directory"]).resolve()
    training, runtime = config["training"], config["runtime"]
    reference = config["model"]["name"]
    revision = config["model"]["revision"]
    manifest_path = initialized / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        reference = manifest["reference_model"]
        revision = manifest.get("reference_revision")
    micro = runtime["micro_batch_size"]
    return {
        "data": {
            "train_files": [str(data_dir / "prompts-train.parquet")],
            "val_files": [str(data_dir / "prompts-validation.parquet")],
            "train_batch_size": training["prompt_batch_size"],
            "val_batch_size": training["prompt_batch_size"],
            "max_prompt_length": config["data"]["max_prompt_tokens"],
            "max_response_length": training["response_tokens"],
            "prompt_key": "prompt_text",
            "reward_fn_key": "data_source",
            "custom_cls": {"path": "pkg://verl.utils.dataset.text_dataset", "name": "RLTextDataset"},
            "ce_train_files": [str(data_dir / "train.parquet")],
            "ce_val_files": [str(data_dir / "validation.parquet")],
            "ce_batch_size": training["ce_batch_size"],
            "ce_val_batch_size": training["ce_batch_size"],
            "ce_max_length": training["ce_sequence_length"],
            "ce_eval_batches": 1,
            "shuffle": True,
            "validation_shuffle": False,
            "dataloader_num_workers": 0,
            "seed": training["seed"],
            "filter_overlong_prompts": True,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "model": {
                "path": str(initialized / "actor"),
                "trust_remote_code": False,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "lora_rank": 0,
            },
            "actor": {
                "strategy": "fsdp",
                "ce_coef": training["ce_coefficient"],
                "ppo_mini_batch_size": training["mini_batch_size"],
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": micro,
                "ppo_epochs": training["inner_steps"],
                "clip_ratio": training["clip_ratio"],
                "clip_ratio_low": training["clip_ratio"],
                "clip_ratio_high": training["clip_ratio"],
                "grad_clip": training["gradient_clip"],
                "loss_agg_mode": "token-mean",
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "entropy_coeff": 0.0,
                "fsdp_config": {
                    "model_dtype": "float32",
                    "mixed_precision": {
                        "param_dtype": config["model"]["dtype"],
                        "reduce_dtype": "float32",
                        "buffer_dtype": "float32",
                    },
                },
                "optim": {
                    "lr": training["learning_rate"],
                    "betas": training["betas"],
                    "weight_decay": training["weight_decay"],
                    "lr_warmup_steps": training["warmup_steps"],
                    "lr_warmup_steps_ratio": 0.0,
                    "warmup_style": "cosine",
                    "cosine_decay_steps": round(training["total_steps"] * training["cosine_decay_fraction"]),
                    "min_lr_ratio": training["min_learning_rate_ratio"],
                },
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",
                "n": training["group_size"],
                "seed": training["seed"],
                "temperature": training["temperature"],
                "top_k": -1,
                "top_p": 1.0,
                "response_length": training["response_tokens"],
                "ignore_eos": True,
                "tensor_model_parallel_size": runtime["rollout_tensor_parallel_size"],
                "gpu_memory_utilization": runtime["rollout_memory_fraction"],
                "dtype": config["model"]["dtype"],
                "enforce_eager": True,
                "log_prob_micro_batch_size_per_gpu": micro,
                "val_kwargs": {
                    "n": 1,
                    "temperature": training["temperature"],
                    "top_k": -1,
                    "top_p": 1.0,
                    "do_sample": True,
                },
            },
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
            "kl_ctrl": {"kl_coef": 0.0},
        },
        "critic": {"enable": False},
        "reward_model": {
            "enable": False,
            "reward_manager": "marktune",
            "enable_resource_pool": False,
            "reward_kwargs": {
                "ref_model_path": reference,
                "revision": revision,
                "key_path": str(initialized / "watermark-key.pt"),
                "dtype": config["model"]["dtype"],
                "device": "cuda",
            },
        },
        "trainer": {
            "total_epochs": max(1, training["total_steps"]),
            "total_training_steps": training["total_steps"],
            "nnodes": 1,
            "n_gpus_per_node": runtime["gpus"],
            "device": "cuda",
            "logger": ["console"],
            "project_name": "marktune",
            "experiment_name": output.name,
            "default_local_dir": str(output),
            "default_hdfs_dir": None,
            "save_freq": training["save_steps"],
            "test_freq": -1,
            "val_before_train": False,
            "resume_mode": "disable",
            "remove_previous_ckpt_in_save": False,
            "max_actor_ckpt_to_keep": None,
            "max_critic_ckpt_to_keep": None,
        },
    }


def compose_verl_config(overlay):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    spec = importlib.util.find_spec("verl")
    if spec is None:
        raise RuntimeError("Install the bundled training backend: pip install -e third_party/verl")
    directory = Path(spec.origin).parent / "trainer/config"
    if not directory.is_dir():
        raise RuntimeError(
            "The imported verl is not the bundled runtime; install third_party/verl and run from this project"
        )
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        base = compose(config_name="ppo_trainer")
    OmegaConf.set_struct(base, False)
    with open_dict(base):
        result = OmegaConf.merge(base, overlay)
    return result


def initialize_actor(config, output, method="marktune"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .watermark import inject_watermark

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], revision=model_cfg["revision"], trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg["revision"],
        torch_dtype=getattr(torch, model_cfg["dtype"]),
        trust_remote_code=False,
    ).to(config["runtime"]["device"])
    watermark = config["watermark"]
    sigma = watermark["gaussmark_sigma"] if method == "gaussmark" else watermark["sigma"]
    key = inject_watermark(model, watermark["parameters"], sigma, watermark["seed"])
    revision = getattr(model.config, "_commit_hash", None) or model_cfg["revision"]
    key["reference_model"] = model_cfg["name"]
    key["reference_revision"] = revision
    torch.save(key, output / "watermark-key.pt")
    model.save_pretrained(output / "actor", safe_serialization=True)
    tokenizer.save_pretrained(output / "actor")
    manifest = {
        "method": method,
        "reference_model": model_cfg["name"],
        "reference_revision": revision,
        "sigma": sigma,
        "watermark_parameters": watermark["parameters"],
        "key_seed": watermark["seed"],
        "dtype": model_cfg["dtype"],
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    with (output / "recipe.yaml").open("x") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def run_training(config, initialized, output, dry_run=False):
    overlay = build_verl_config(config, initialized, output)
    if dry_run:
        print(yaml.safe_dump(overlay, sort_keys=False))
        return
    if not config["runtime"]["device"].startswith("cuda"):
        raise ValueError("Distributed MarkTune training requires CUDA; CPU supports initialization and evaluation")
    initialized, output = Path(initialized), Path(output)
    manifest = json.loads((initialized / "manifest.json").read_text())
    if manifest["method"] != "marktune":
        raise ValueError("Training requires a MarkTune initialization")
    if manifest["reference_model"] != config["model"]["name"]:
        raise ValueError("Initialization and training reference models differ")
    if manifest["key_seed"] != config["watermark"]["seed"]:
        raise ValueError("Initialization and training key seeds differ")
    if manifest["dtype"] != config["model"]["dtype"]:
        raise ValueError("Initialization and training dtypes differ")
    if config["model"]["revision"] is not None and config["model"]["revision"] != manifest["reference_revision"]:
        raise ValueError("Initialization and training model revisions differ; use the resolved immutable revision")
    if (
        manifest["sigma"] != config["watermark"]["sigma"]
        or manifest["watermark_parameters"] != config["watermark"]["parameters"]
    ):
        raise ValueError("Initialization and training watermark settings differ")
    for files in ("train_files", "val_files", "ce_train_files", "ce_val_files"):
        for path in overlay["data"][files]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    composed = compose_verl_config(overlay)
    from omegaconf import OmegaConf
    from transformers import set_seed
    from verl.trainer.main_ppo import run_ppo

    set_seed(config["training"]["seed"])
    with (output / "recipe.yaml").open("x") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with (output / "verl.yaml").open("x") as handle:
        handle.write(OmegaConf.to_yaml(composed, resolve=True))
    run_ppo(composed)
