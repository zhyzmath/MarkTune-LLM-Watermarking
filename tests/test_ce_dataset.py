import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "third_party/verl/verl/utils/dataset/ce_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ce_dataset", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ce_training_batches_keep_the_configured_size():
    module = load_module()
    loader = module.create_ce_train_loader([f"document {i}" for i in range(20)], batch_size=8)
    batches = list(loader)
    assert [len(batch) for batch in batches] == [8, 8]
    assert len({text for batch in batches for text in batch}) == 16


def test_ce_training_rejects_a_corpus_smaller_than_one_batch():
    module = load_module()
    with pytest.raises(ValueError, match="at least 8"):
        module.create_ce_train_loader(["one document"], batch_size=8)
