from abc import ABC, abstractmethod
from typing import Any

from jaxtyping import Float, Int
from torch import Tensor

from vl_saliency.core.grid import SaliencyGrid


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
        labels: Int[Tensor, "B S"],
        input_ids: Int[Tensor, "B S"],
        segment_ids: Int[Tensor, "B S"],
        preds: Float[Tensor, "B T V"],
        saliency: SaliencyGrid,
        masks: list[Tensor],
        **kwargs: Any,
    ) -> Float[Tensor, "1"]:
        loss = self.compute_loss(
            labels=labels,
            input_ids=input_ids,
            segment_ids=segment_ids,
            preds=preds,
            saliency=saliency,
            masks=masks,
            **kwargs,
        )
        return self.weight * loss

    @abstractmethod
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
        """Compute the auxiliary loss.

        Args:
            labels (torch.Tensor): Ground truth labels.  [batch_size, seq_len]
            input_ids (torch.Tensor): Input token IDs. [batch_size, seq_len]
            preds (torch.Tensor): The model prediction logits. [batch_size, seq_len, vocab_size]
            saliency (SaliencyGrid): The attention weights from the model.
            masks (list[torch.Tensor]): Binary annotation masks. List of [H, W]. Zero tensors indicate no annotation.
            **kwargs (Any): Additional keyword arguments

        Returns:
            Tensor: Computed auxiliary loss.
        """
        ...
