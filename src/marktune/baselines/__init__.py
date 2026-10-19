"""Inference-time baselines used in the MarkTune experiments."""

from .kgw import KGW
from .kgw_distilled import KGWDistilled
from .gumbel import GumbelMax
from .synthid import SynthID


def create_baseline(method, vocab_size, config, *, device="cpu", eos_token_id=None, special_token_ids=()):
    """Construct a response detector and inference processor where applicable."""
    options = config.get("baselines", {})
    temperature = config["evaluation"]["temperature"]
    if method == "kgw-d":
        return KGWDistilled(vocab_size, special_token_ids=special_token_ids, **options.get("kgw_distilled", {}))
    if method in ("kgw-1", "kgw-2"):
        kwargs = dict(options.get("kgw", {}))
        if method == "kgw-2":
            kwargs["delta"] = 2.0
        return KGW(vocab_size, device=device, **kwargs)
    if method == "gumbel":
        kwargs = dict(options.get("gumbel", {}))
        kwargs.setdefault("seed", 15485863)
        return GumbelMax(vocab_size, temperature=temperature, **kwargs)
    if method == "synthid":
        kwargs = dict(options.get("synthid", {}))
        tournaments = kwargs.pop("tournaments", 30)
        if type(tournaments) is not int or not 1 <= tournaments <= 30:
            raise ValueError("SynthID tournaments must be an integer in [1, 30]")
        from .synthid import DEFAULT_KEYS

        kwargs.setdefault("keys", DEFAULT_KEYS[:tournaments])
        return SynthID(vocab_size, device=device, eos_token_id=eos_token_id, temperature=temperature, **kwargs)
    raise ValueError(f"No inference-time baseline for {method!r}")


__all__ = ["KGW", "KGWDistilled", "GumbelMax", "SynthID", "create_baseline"]
