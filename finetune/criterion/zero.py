from collections.abc import Sequence
from typing import Any

import torch

from .base import Criterion


class ZeroCriterion(Criterion):
    def compute_loss(
        self,
        labels: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        annotation_ids: torch.Tensor,
        masks: torch.Tensor,
        segment_infos: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        return 0.0
