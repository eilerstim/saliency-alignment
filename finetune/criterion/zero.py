from typing import Any

from jaxtyping import Float, Int
from torch import Tensor

from vl_saliency.core.grid import SaliencyGrid

from .base import Criterion


class ZeroCriterion(Criterion):
    """A criterion that returns zero loss (equivalent to no auxiliary loss)."""

    def compute_loss(
        self,
        labels: Int[Tensor, "B S"],
        input_ids: Int[Tensor, "B S"],
        segment_ids: Int[Tensor, "B S"],
        preds: Float[Tensor, "B T V"],
        saliency: SaliencyGrid,
        masks: list[Tensor],
        **kwargs: Any,
    ) -> Float[Tensor, "1"]:
        return preds.new_zeros(1)
