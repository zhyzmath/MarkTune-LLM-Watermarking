from types import SimpleNamespace

import pytest
import torch

from marktune.watermark import WatermarkDetector, inject_watermark, load_key, select_parameters


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 4)
        self.proj = torch.nn.Linear(4, 11, bias=False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.proj(self.embed(input_ids)))


def test_injection_uses_persisted_exact_noise_and_rms(tmp_path):
    model = TinyLM()
    before = model.proj.weight.detach().clone()
    record = inject_watermark(model, ["proj.weight"], sigma=0.6, seed=133377)
    rms = before.float().square().mean().sqrt()
    expected = torch.randn(before.numel(), generator=torch.Generator().manual_seed(133377))
    torch.testing.assert_close(record["per_param_noise_iso"]["proj.weight"].flatten(), expected)
    torch.testing.assert_close(model.proj.weight, before + expected.reshape_as(before) * (0.6 * rms))
    torch.save(record, tmp_path / "key.pt")
    loaded = load_key(tmp_path / "key.pt")
    torch.testing.assert_close(loaded["per_param_noise_iso"]["proj.weight"], expected.reshape_as(before))


def test_detector_matches_reference_formula_and_restores_flags():
    model = TinyLM().eval()
    flags = [p.requires_grad for p in model.parameters()]
    key = inject_watermark(TinyLM(), ["proj.weight"], sigma=0.6, seed=42)
    ids = torch.tensor([1, 3, 2, 5])
    logits = model(ids[None, :-1]).logits.float()
    logp = torch.log_softmax(logits, -1).gather(-1, ids[None, 1:, None]).sum()
    gradient = torch.autograd.grad(logp, model.proj.weight)[0].float()
    noise = key["per_param_noise_iso"]["proj.weight"].float()
    expected = (gradient * noise).sum() / gradient.norm()
    detector = WatermarkDetector(model, key)
    with torch.no_grad():
        actual = detector.score(ids)
    assert actual == pytest.approx(expected.item(), rel=1e-5)
    assert [p.requires_grad for p in model.parameters()] == flags
    assert all(p.grad is None for p in model.parameters())
    assert detector.score(torch.tensor([1])) == 0


def test_key_shape_and_selectors_fail_closed():
    model = TinyLM()
    with pytest.raises(ValueError, match="missing"):
        select_parameters(model, ["missing.weight"])
    with pytest.raises(ValueError, match="duplicate"):
        select_parameters(model, ["proj.weight", "proj.weight"])
    key = inject_watermark(model, ["proj.weight"], sigma=0.6, seed=42)
    key["per_param_noise_iso"]["proj.weight"] = torch.zeros(2)
    with pytest.raises(ValueError, match="shape"):
        WatermarkDetector(model, key)
