from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch


class Criterion(ABC):
    """Abstract base class for auxiliary loss functions.

    Args:
        weight (float): Weighting factor for the loss. Default is 1.0.

    Methods:
        compute_loss(labels, preds, attentions, **kwargs) -> float:
            Abstract method to compute the auxiliary loss. Must be implemented by subclasses.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def __call__(
        self,
        labels: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        return self.weight * self.compute_loss(labels, preds, attentions, **kwargs)

    @abstractmethod
    def compute_loss(
        self,
        labels: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> float: ...
