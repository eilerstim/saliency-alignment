from collections.abc import Sequence
from typing import Any

import torch
from vl_saliency import Trace

from finetune.criterion import Criterion


# TODO:
# - Extract image token attention for text tokens
# - Calculate saliency maps from attention weights
# - Compute alignment loss with provided annotations
# - Mask out non-annotated tokens
# - Mean over token and batch dimensions
class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion.

    This criterion computes the saliency alignment loss between the model's
    saliency maps and the provided annotation.
    """

    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight (float): Weight for the saliency alignment loss.
            image_token_id (int): Token ID representing the image token in the input.
        """
        super().__init__(weight)

    def compute_loss(
        self,
        labels: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        # Compute saliency maps for annotated tokens
        pass


# TODO: Move to utils
def trace_attentions(
    attentions: Sequence[torch.Tensor],
    image_tokens: torch.Tensor,  # (batch_size, num_image_tokens)
    annotated_tokens: torch.Tensor,  # (batch_size, num_annotated_tokens)
) -> list[Trace]:
    pass
