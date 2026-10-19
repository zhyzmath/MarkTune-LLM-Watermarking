from pathlib import Path

import pytest

from marktune.config import load_config
from marktune.training import build_verl_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "recipe,steps,ce,sigma,temp,warmup,lr,layer",
    [
        ("qwen3", 200, 0.01, 0.6, 0.7, 20, 5e-6, 28),
        ("llama2", 300, 0.005, 0.8, 1.0, 30, 4e-6, 30),
    ],
)
def test_camera_ready_recipes(recipe, steps, ce, sigma, temp, warmup, lr, layer, tmp_path):
    config = load_config(ROOT / "configs" / f"{recipe}.yaml")
    assert config["training"]["regularization"] == "ce"
    assert config["training"]["total_steps"] == steps
    assert config["training"]["ce_coefficient"] == ce
    assert config["watermark"]["sigma"] == sigma
    assert config["watermark"]["parameters"] == [f"model.layers.{layer}.mlp.up_proj.weight"]
    assert config["evaluation"]["temperature"] == temp
    generated = build_verl_config(config, tmp_path / "initialized", tmp_path / "run")
    actor = generated["actor_rollout_ref"]["actor"]
    assert actor["ce_coef"] == ce
    assert actor["use_kl_loss"] is False
    assert actor["kl_loss_coef"] == 0.0
    assert generated["algorithm"]["use_kl_in_reward"] is False
    assert generated["algorithm"]["kl_ctrl"]["kl_coef"] == 0.0
    assert actor["optim"]["lr"] == lr
    assert actor["optim"]["lr_warmup_steps"] == warmup
    assert actor["optim"]["cosine_decay_steps"] == round(steps * 0.4)
    assert actor["ppo_epochs"] == 3
    assert actor["ppo_mini_batch_size"] == 64
    assert generated["data"]["ce_batch_size"] == 128
    assert generated["data"]["train_batch_size"] == 64
    assert generated["trainer"]["n_gpus_per_node"] == 4
    assert generated["trainer"]["total_training_steps"] == steps
    assert generated["reward_model"]["reward_manager"] == "marktune"


def test_overrides_are_typed_and_unknown_fields_rejected():
    path = ROOT / "configs/qwen3.yaml"
    config = load_config(path, ["training.total_steps=50", "runtime.device=cpu"])
    assert config["training"]["total_steps"] == 50
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path, ["training.ce_coeffcient=1"])
    with pytest.raises(ValueError):
        load_config(path, ["training.total_steps=oops"])
    with pytest.raises(ValueError):
        load_config(path, ["watermark.sigma=0"])
    with pytest.raises(ValueError, match="regularization"):
        load_config(path, ["training.regularization=kl"])


def test_scientific_notation_override_and_yaml_typo(tmp_path):
    import yaml

    path = ROOT / "configs/qwen3.yaml"
    config = load_config(path, ["training.learning_rate=1e-5"])
    assert config["training"]["learning_rate"] == 1e-5
    config["training"]["ce_coeffcient"] = 0.1
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(invalid)


@pytest.mark.parametrize("recipe", ["qwen3", "llama2"])
def test_complete_training_config_resolves(recipe, monkeypatch, tmp_path):
    pytest.importorskip("hydra")
    from marktune.training import compose_verl_config
    from omegaconf import OmegaConf

    monkeypatch.syspath_prepend(str(ROOT / "third_party/verl"))
    config = load_config(ROOT / "configs" / f"{recipe}.yaml", ["training.clip_ratio=0.1"])
    result = compose_verl_config(build_verl_config(config, tmp_path / "init", tmp_path / "run"))
    complete = OmegaConf.to_container(result, resolve=True, throw_on_missing=True)
    assert complete["actor_rollout_ref"]["rollout"]["seed"] == config["training"]["seed"]
    assert complete["actor_rollout_ref"]["rollout"]["prompt_length"] == 256
    assert complete["data"]["custom_cls"]["path"] == "pkg://verl.utils.dataset.text_dataset"
    assert complete["actor_rollout_ref"]["actor"]["ce_coef"] == config["training"]["ce_coefficient"]
    assert complete["actor_rollout_ref"]["actor"]["use_kl_loss"] is False
    assert complete["actor_rollout_ref"]["actor"]["kl_loss_coef"] == 0.0
    assert complete["algorithm"]["use_kl_in_reward"] is False
    assert complete["algorithm"]["kl_ctrl"]["kl_coef"] == 0.0
    assert complete["actor_rollout_ref"]["actor"]["clip_ratio_low"] == 0.1
    assert complete["actor_rollout_ref"]["actor"]["clip_ratio_high"] == 0.1


@pytest.mark.parametrize("recipe", ["qwen3", "llama2"])
def test_paper_learning_rate_updates_master_parameters(recipe, tmp_path):
    import torch

    config = load_config(ROOT / "configs" / f"{recipe}.yaml")
    generated = build_verl_config(config, tmp_path / "init", tmp_path / "run")
    actor = generated["actor_rollout_ref"]["actor"]
    parameter = torch.nn.Parameter(
        torch.tensor([0.03, 0.1, 1.0], dtype=getattr(torch, actor["fsdp_config"]["model_dtype"]))
    )
    before = parameter.detach().clone()
    optimizer = torch.optim.AdamW([parameter], lr=actor["optim"]["lr"], weight_decay=0.0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    expected = torch.full_like(parameter, actor["optim"]["lr"])
    torch.testing.assert_close(before - parameter, expected, rtol=0.02, atol=1e-8)
    assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
