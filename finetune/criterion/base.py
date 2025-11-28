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
        input_ids: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        loss = self.compute_loss(
            labels=labels,
            input_ids=input_ids,
            preds=preds,
            attentions=attentions,
            masks=masks,
            **kwargs,
        )
        return self.weight * loss

    @abstractmethod
    def compute_loss(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        """Compute the auxiliary loss.

        Args:
            labels (torch.Tensor): Ground truth labels.  [batch_size, seq_len]
            input_ids (torch.Tensor): Input token IDs. [batch_size, seq_len]
            preds (torch.Tensor): The model prediction logits. [batch_size, seq_len, vocab_size]
            attentions (Sequence[torch.Tensor]): The attention weights from the model. [List of tensors with shape [batch_size, num_heads, seq_len, seq_len]]
            masks (list[torch.Tensor]): Binary annotation masks. List of [seq_len, H, W]. Zero tensors indicate no annotation.
            **kwargs (Any): Additional keyword arguments

        Returns:
            float: Computed auxiliary loss.
        """
        ...
