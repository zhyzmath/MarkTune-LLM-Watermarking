import pytest

from marktune.schedule import learning_rate_multiplier


def test_cosine_covers_forty_percent_of_training_after_warmup():
    def schedule(step):
        return learning_rate_multiplier(step, warmup_steps=20, decay_steps=80)

    assert schedule(0) == 0
    assert schedule(10) == 0.5
    assert schedule(20) == 1
    assert schedule(60) == pytest.approx(0.5)
    assert schedule(100) == 0
    assert schedule(200) == 0
    assert schedule(300) == 0
