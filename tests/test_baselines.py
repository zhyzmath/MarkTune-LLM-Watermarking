import math

import pytest
import torch


def test_kgw_matches_markllm_product_prf_and_score():
    from marktune.baselines import KGW

    baseline = KGW(31, gamma=0.5, delta=2, prefix_length=4, hash_key=42)
    ids = torch.tensor([[1, 2, 3, 4]])
    generator = torch.Generator().manual_seed(42)
    prf = torch.randperm(31, generator=generator)
    generator.manual_seed((42 * int(prf[24])) % 31)
    expected = torch.randperm(31, generator=generator)[:15]
    scores = baseline(ids, torch.zeros((1, 31)))
    assert set(torch.where(scores[0] == 2)[0].tolist()) == set(expected.tolist())
    sequence = torch.cat([ids[0], expected[:1]])
    assert baseline.score(sequence) == pytest.approx(1)


def test_gumbel_uses_seeded_exp_race_and_higher_is_stronger_score():
    from marktune.baselines import GumbelMax

    baseline = GumbelMax(20, prefix_length=4, seed=42, temperature=0.7)
    ids = torch.tensor([[1, 2, 3, 4]])
    scores = baseline(ids, torch.zeros((1, 20)))
    generator = torch.Generator().manual_seed(42 * (24 % 20))
    expected = torch.rand(20, generator=generator).argmax()
    assert scores.argmax().item() == expected.item()
    assert torch.isneginf(scores[0]).sum() == 19
    assert baseline.score(torch.cat([ids[0], expected[None]])) > 0
    with pytest.raises(ValueError, match="prefix"):
        baseline.score(torch.tensor([1, 2]))


def test_synthid_uses_four_prefix_tokens_and_resets_repetition_state():
    from marktune.baselines import SynthID

    baseline = SynthID(20, prefix_length=4, keys=[1, 2, 3], sampling_table_size=256)
    ids = torch.tensor([[1, 2, 3, 4]])
    scores = torch.zeros((1, 20))
    first = baseline(ids, scores)
    assert not torch.equal(first, scores)
    assert torch.equal(baseline(ids, scores), scores)
    baseline.reset()
    assert torch.equal(baseline(ids, scores), first)
    assert baseline.context_mask(torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 5])).tolist() == [True, True, True, True, False]
    with pytest.raises(ValueError, match="prefix"):
        baseline.score(torch.tensor([1, 2, 3, 4]))


def test_synthid_score_ignores_eos_and_repeated_context():
    from marktune.baselines import SynthID

    baseline = SynthID(20, prefix_length=2, keys=[1, 2, 3], eos_token_id=19, sampling_table_size=256)
    ids = torch.tensor([1, 2, 3, 19, 4, 5])
    assert baseline.score(ids) == baseline.score(ids[:3])
    assert math.isfinite(baseline.score(ids))


def test_synthid_factory_maps_tournaments_to_original_keys():
    from marktune.baselines import create_baseline

    config = {"evaluation": {"temperature": 0.7}, "baselines": {"synthid": {"tournaments": 3}}}
    baseline = create_baseline("synthid", 20, config)
    assert baseline.keys.tolist() == [654, 400, 836]
    config["baselines"]["synthid"]["tournaments"] = 31
    with pytest.raises(ValueError, match="tournaments"):
        create_baseline("synthid", 20, config)


def test_gumbel_handles_padded_model_vocabulary_without_changing_prf():
    from marktune.baselines import GumbelMax

    baseline = GumbelMax(13)
    result = baseline(torch.tensor([[1, 2, 3, 4]]), torch.zeros((1, 20)))
    assert result.shape == (1, 20)
    assert result.argmax().item() < 13
    assert torch.isneginf(result[:, 13:]).all()


def test_synthid_batched_generation_matches_independent_context_histories():
    from marktune.baselines import SynthID

    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    scores = torch.arange(40, dtype=torch.float32).reshape(2, 20) / 10
    baseline = SynthID(20, keys=[1, 2, 3], sampling_table_size=256)
    combined = baseline(ids, scores)
    for index in range(2):
        independent = SynthID(20, keys=[1, 2, 3], sampling_table_size=256)
        assert torch.equal(combined[index], independent(ids[index : index + 1], scores[index : index + 1])[0])


@pytest.mark.parametrize("scale", [0.1, 3.0, 30.0])
def test_synthid_thirty_tournaments_keep_valid_masked_probability_distribution(scale):
    from marktune.baselines import SynthID

    rng = torch.Generator().manual_seed(7)
    ids = torch.randint(3, 16, (128, 4), generator=rng)
    scores = torch.randn((128, 16), generator=rng) * scale
    scores[:, :3] = -torch.inf
    processed = SynthID(16)(ids, scores)
    assert not torch.isnan(processed).any()
    assert not torch.isposinf(processed).any()
    assert torch.isneginf(processed[:, :3]).all()
    probabilities = torch.softmax(processed, dim=-1)
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(128))
    sampled = torch.multinomial(probabilities, num_samples=20, replacement=True, generator=rng)
    assert (sampled >= 3).all()


def test_synthid_thirty_tournaments_preserve_single_token_support_without_uniform_fallback():
    from marktune.baselines import SynthID

    scores = torch.full((1, 16), -torch.inf)
    scores[0, 7] = 30
    processed = SynthID(16)(torch.tensor([[11, 13, 3, 14]]), scores)
    expected = torch.zeros_like(scores)
    expected[0, 7] = 1
    assert torch.equal(torch.softmax(processed, dim=-1), expected)
    assert torch.isneginf(processed[0, :7]).all()
    assert torch.isneginf(processed[0, 8:]).all()


def test_kgw_distilled_matches_pinned_upstream_teacher_greenlists():
    from marktune.baselines import KGWDistilled

    # KGWWatermark output at watermark-learnability f84f70f9fc30eba5e39bf1889376d61de5f2ee59:
    # V=32, gamma=.25, simple_1, CPU RNG, UNK/BOS/EOS IDs 0/1/2.
    expected = {
        0: [],
        1: [],
        2: [],
        3: [16, 17, 21, 23, 25, 26, 27, 31],
        4: [3, 4, 10, 15, 24, 26, 28],
        7: [4, 12, 14, 18, 22, 24, 25],
        31: [4, 7, 8, 13, 15, 23, 24],
    }
    baseline = KGWDistilled(32, special_token_ids=[0, 1, 2])
    for prefix, green in expected.items():
        assert sorted(baseline.greenlist(torch.tensor([prefix])).tolist()) == green
    # Removing special tokens does not refill the green list or shrink V before randperm.
    assert len(baseline.greenlist(torch.tensor([4]))) == 7
    assert baseline.hash_key == 15485863
    assert baseline.device == torch.device("cpu")


def test_kgw_distilled_factory_is_independent_of_inference_kgw_prf():
    from marktune.baselines import KGWDistilled, create_baseline

    config = {
        "evaluation": {"temperature": 1.0},
        "baselines": {
            "kgw": {"gamma": 0.5, "delta": 2.0, "hash_key": 42, "prefix_length": 4},
            "kgw_distilled": {"gamma": 0.25, "hash_key": 15485863},
        },
    }
    baseline = create_baseline("kgw-d", 32, config, device="cuda", special_token_ids=[0, 1, 2])
    assert isinstance(baseline, KGWDistilled)
    assert sorted(baseline.greenlist(torch.tensor([3])).tolist()) == [16, 17, 21, 23, 25, 26, 27, 31]
    assert baseline.score(torch.tensor([3, 16])) == pytest.approx(math.sqrt(3))
    assert baseline.score(torch.tensor([1, 16])) == pytest.approx(-1 / math.sqrt(3))
    with pytest.raises(ValueError, match="prefix"):
        baseline.score(torch.tensor([3]))
