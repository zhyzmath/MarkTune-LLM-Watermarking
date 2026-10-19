import json
import random
from types import SimpleNamespace

import pytest
import torch


def test_roc_groups_ties_and_bootstrap_is_paired_and_reproducible():
    from marktune.evaluation.metrics import detection_metrics

    result = detection_metrics([0, 0], [0, 0], fpr=0.01, bootstrap_samples=20, seed=7)
    assert result["auc"] == pytest.approx(0.5)
    assert result["tpr_at_fpr"] == pytest.approx(0.01)
    assert result["null_samples"] == 2
    assert result["null_fpr_resolution"] == 0.5
    assert result["bootstrap"]["tpr_mean"] == pytest.approx(0.01)
    assert result == detection_metrics([0, 0], [0, 0], fpr=0.01, bootstrap_samples=20, seed=7)
    with pytest.raises(ValueError, match="paired"):
        detection_metrics([0], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        detection_metrics([float("nan")], [1])


def test_detection_separates_empirical_threshold_from_interpolation():
    from marktune.evaluation.metrics import detection_metrics

    result = detection_metrics([0, 0, 1, 1], [0.5, 1.5, 2, 3], fpr=0.25, bootstrap_samples=0)
    assert result["empirical_fpr"] <= 0.25
    assert result["empirical_tpr"] == 0.75
    assert result["auc"] == pytest.approx(0.875)


def test_sequence_repetition_reports_mean_per_sequence():
    from marktune.evaluation.metrics import repetition_metrics

    result = repetition_metrics(["a b c a b c", "x y z"])
    assert result["seq_rep_3"] == pytest.approx(0.125)
    assert result["seq_rep_3_micro"] == pytest.approx(0.2)


def test_word_attacks_are_seeded_and_substitutions_use_wordnet_candidates():
    from marktune.evaluation.attacks import delete_words, substitute_words, edit_counts

    assert delete_words("one two three", 1, random.Random(3)) == ""
    assert delete_words("one two three", 0, random.Random(3)) == "one two three"

    def candidates(word):
        return {"fast": ["quick"], "car": ["vehicle"]}.get(word, [])

    assert substitute_words("a fast car", 1, random.Random(3), candidates) == "a quick vehicle"
    assert edit_counts([1, 2, 3], [1, 4, 3, 5]) == {"insertions": 1, "deletions": 0, "substitutions": 1, "distance": 2}
    with pytest.raises(ValueError, match="ratio"):
        delete_words("x", 1.1, random.Random(1))


def test_outputs_never_overwrite_and_json_is_strict(tmp_path):
    from marktune.evaluation.io import write_json, write_jsonl, read_records

    path = tmp_path / "samples.jsonl"
    write_jsonl(path, [{"text": "ok"}])
    assert list(read_records(path)) == [{"text": "ok"}]
    with pytest.raises(FileExistsError):
        write_jsonl(path, [{"text": "changed"}])
    assert json.loads(path.read_text()) == {"text": "ok"}
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", {"score": float("nan")})


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(x) for x in tokens if x != 9)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.last_kwargs = None

    def generate(self, input_ids, **kwargs):
        self.last_kwargs = kwargs
        return torch.cat([input_ids, torch.tensor([[4, 5, 6]])], dim=1)

    def forward(self, input_ids, attention_mask=None):
        # Uniform ten-token distribution -> perplexity ten.
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 10)))


def test_generation_stores_response_only_and_forces_exact_token_budget():
    from marktune.evaluation.generation import generate_record

    model = TinyModel()
    config = {"evaluation": {"response_tokens": 3, "prompt_tokens": 2, "temperature": 0.7}}
    result = generate_record(model, Tokenizer(), {"prompt": "1 2", "reference": "7 8"}, config)
    assert result["token_ids"] == [4, 5, 6]
    assert result["text"] == "4 5 6"
    assert result["prompt"] == "1 2"
    assert model.last_kwargs["max_new_tokens"] == 3
    assert model.last_kwargs["min_new_tokens"] == 3
    assert model.last_kwargs["top_k"] == 0
    assert model.last_kwargs["top_p"] == 1.0


def test_detection_is_prompt_agnostic_and_retokenizes_visible_text():
    from marktune.evaluation.detection import score_records

    seen = []

    def detector(ids):
        seen.append(ids.flatten().tolist())
        return float(ids.sum())

    result = list(score_records([{"prompt": "7 8", "text": "1 2 3", "token_ids": [8, 8]}], Tokenizer(), detector))
    assert seen == [[1, 2, 3]]
    assert result[0]["score"] == 6
    assert result[0]["token_ids"] == [1, 2, 3]


def test_perplexity_masks_prompt_and_preserves_boundary_token():
    from marktune.evaluation.metrics import response_perplexity

    assert response_perplexity(TinyModel(), [1, 2], [3, 4]) == pytest.approx(10)
    with pytest.raises(ValueError, match="response"):
        response_perplexity(TinyModel(), [1], [])


def test_downstream_command_uses_paper_tasks_and_requires_code_opt_in(tmp_path):
    from marktune.evaluation.downstream import build_command

    with pytest.raises(ValueError, match="code"):
        build_command("checkpoint", "bfloat16", tmp_path, ["mbpp"])
    command = build_command("checkpoint", "bfloat16", tmp_path, ["mmlu"])
    assert "--num_fewshot" in command and command[command.index("--num_fewshot") + 1] == "5"
    assert "trust_remote_code=False" in command[command.index("--model_args") + 1]


def test_lora_config_preserves_effective_batch_size():
    from marktune.evaluation.lora import training_options

    config = {
        "attacks": {
            "lora": {
                "batch_size": 64,
                "steps": 1500,
                "learning_rate": 1e-5,
                "warmup_steps": 300,
                "sequence_length": 512,
                "save_steps": 100,
            }
        }
    }
    result = training_options(config, world_size=4, micro_batch_size=2)
    assert result["gradient_accumulation_steps"] == 8
    assert result["max_steps"] == 1500
    assert result["lr_scheduler_type"] == "cosine"
    with pytest.raises(ValueError, match="divisible"):
        training_options(config, world_size=3, micro_batch_size=2)


def test_generation_requires_full_requested_sample_count(tmp_path):
    from marktune.evaluation.generation import take_samples

    with pytest.raises(ValueError, match="requires 2"):
        take_samples([{"prompt": "x"}], 2)
    assert take_samples([1, 2, 3], 2) == [1, 2]


def test_reference_identity_rejects_a_different_clean_reference():
    from marktune.evaluation.detection import reference_identity

    key = {"reference_model": "base", "reference_revision": "fixed-hash"}
    assert reference_identity(key, None, "base") == ("base", "fixed-hash")
    with pytest.raises(ValueError, match="identity"):
        reference_identity(key, "other-base", "base")


def test_benchmark_uses_additional_warmups_and_exact_measurement_count():
    from marktune.evaluation.benchmark import benchmark_records

    rows = [{"prompt": "1 2", "id": value} for value in range(4)]
    config = {"evaluation": {"response_tokens": 3, "prompt_tokens": 2, "temperature": 0.7}}
    seen = []

    def score(ids):
        seen.append(ids.tolist())
        return float(ids.sum())

    result = benchmark_records(rows, TinyModel(), Tokenizer(), score, config, samples=2, warmup=2)
    assert len(seen) == 4
    assert [row["id"] for row in result["measurements"]] == [0, 1]
    assert result["samples"] == 2
    with pytest.raises(ValueError, match="warmup"):
        benchmark_records(rows[:2], TinyModel(), Tokenizer(), score, config, samples=2, warmup=2)


def test_text_perplexity_uses_joint_tokenization_and_response_overlap():
    from marktune.evaluation.metrics import text_perplexity

    class BoundaryTokenizer:
        def __call__(self, value, **kwargs):
            assert value == "abc"
            return {
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
                "offset_mapping": [[(0, 1), (1, 2), (2, 3)]],
            }

    assert text_perplexity(TinyModel(), BoundaryTokenizer(), "a", "bc") == pytest.approx(10)


def test_real_huggingface_generation_accepts_baseline_processors():
    from transformers import GPT2Config, GPT2LMHeadModel
    from marktune.baselines import KGW, GumbelMax, SynthID
    from marktune.evaluation.generation import generate_record

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=10, n_embd=8, n_layer=1, n_head=1, n_positions=16, bos_token_id=0, eos_token_id=9, pad_token_id=9
        )
    ).eval()
    config = {"evaluation": {"response_tokens": 3, "prompt_tokens": 4, "temperature": 0.7}}
    for baseline in [None, KGW(10), GumbelMax(10), SynthID(10, keys=[1, 2])]:
        result = generate_record(model, Tokenizer(), {"prompt": "1 2 3 4"}, config, baseline)
        assert len(result["token_ids"]) == 3
        assert all(0 <= value < 10 for value in result["token_ids"])


def test_cli_help_does_not_import_model_or_training_libraries():
    import argparse
    from marktune.evaluation.cli import add_subparsers

    parser = argparse.ArgumentParser()
    add_subparsers(parser.add_subparsers())
    parsed = parser.parse_args(["attack-lora", "--config", "recipe.yaml", "--model", "checkpoint", "--output", "new"])
    assert callable(parsed.func)


def test_lora_output_reservation_is_shared_across_workers(tmp_path, monkeypatch):
    from marktune.evaluation.lora import reserve_output

    state = SimpleNamespace(is_main_process=True, num_processes=1)
    output = tmp_path / "attack"
    reserve_output(output, state)
    with pytest.raises(FileExistsError):
        reserve_output(output, state)

    def broadcast_status(status, src):
        status[0] = ("FileExistsError", "already exists")

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_status)
    with pytest.raises(FileExistsError):
        reserve_output(output, SimpleNamespace(is_main_process=False, num_processes=2))
