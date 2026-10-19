"""Linear warmup followed by a finite cosine decay window."""

import math


def learning_rate_multiplier(step, warmup_steps, decay_steps, minimum_ratio=0.0):
    if warmup_steps < 0 or decay_steps <= 0 or not 0 <= minimum_ratio <= 1:
        raise ValueError("Invalid warmup, decay duration, or minimum learning-rate ratio")
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
