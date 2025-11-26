from collections.abc import Sequence
from typing import Any

import torch

from .base import Criterion


class ZeroCriterion(Criterion):
    """A criterion that returns zero loss (equivalent to no auxiliary loss)."""

    def compute_loss(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        return 0.0
