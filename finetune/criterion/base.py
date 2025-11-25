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
        annotation_ids: torch.Tensor,
        masks: torch.Tensor,
        segment_infos: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        loss = self.compute_loss(
            labels, preds, attentions, annotation_ids, masks, segment_infos, **kwargs
        )
        return self.weight * loss

    @abstractmethod
    def compute_loss(
        self,
        labels: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        """Compute the auxiliary loss.

        Args:
            labels (torch.Tensor): Ground truth labels.  [batch_size, seq_len]
            preds (torch.Tensor): The model prediction logits. [batch_size, seq_len, vocab_size]
            attentions (Sequence[torch.Tensor]): The attention weights from the model. [List of tensors with shape [batch_size, num_heads, seq_len, seq_len]]
            masks (torch.Tensor): Segmentation masks with pixel values as segment IDs. [batch_size, seq_len, H, W]
            **kwargs (Any): Additional keyword arguments

        Returns:
            float: Computed auxiliary loss.
        """
        ...
