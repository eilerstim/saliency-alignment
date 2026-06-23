"""Length-aware warmup + cosine-decay learning-rate schedule.

Implemented as a single ``LambdaLR`` parameterised by ``trainer.max_steps`` so
the schedule stays correct across the training-length sweep regardless of
dataset size or epoch count (training is step-bounded). Lightning advances it
once per optimizer step (``interval='step'`` in
``FineTuner.configure_optimizers``); a bare scheduler return would default to
per-epoch stepping and never advance in a sub-one-epoch run.
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    *,
    max_steps: int,
    warmup_ratio: float = 0.03,
    eta_min_ratio: float = 0.0,
) -> LambdaLR:
    """Linear warmup over ``warmup_ratio * max_steps`` steps, then cosine decay.

    Args:
        optimizer: Optimizer whose base learning rate(s) are scaled.
        max_steps: Total optimizer steps the run will take (``trainer.max_steps``).
        warmup_ratio: Fraction of ``max_steps`` spent linearly warming up.
        eta_min_ratio: Floor of the cosine, as a fraction of the base LR.

    Returns:
        A ``LambdaLR`` returning a multiplier in ``[eta_min_ratio, 1]``.
    """
    warmup_steps = max(1, round(max_steps * warmup_ratio))
    decay_steps = max(1, max_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
