import torch
from jaxtyping import Bool, Float

from .base import Criterion


class ZeroCriterion(Criterion):
    """A criterion that returns zero loss (equivalent to no auxiliary loss)."""

    def compute_loss(
        self,
        attn: Float[torch.Tensor, "S H W"],
        mask: Bool[torch.Tensor, "S H W"],
    ) -> Float[torch.Tensor, "1"]:
        return attn.sum() * 0.0
