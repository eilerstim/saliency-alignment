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
        annotation_ids: torch.Tensor,
        masks: torch.Tensor,
        segment_infos: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        # TODO assert annotations are as expected (i.e. aligned with caption tokens)
        # Extract annotations
        annotations = load_annotations(
            annotation_ids, masks
        )  # (batch_size, seq_len, H, W)

        # Mask out non-annotated tokens
        annotated = (annotation_ids != -1).any(dim=2)  # (batch_size, seq_len)

        # Return zero loss if no tokens are annotated
        if not annotated.any():
            return 0.0

        # Compute saliency maps for annotated tokens


# TODO: Move to utils
def trace_attentions(
    attentions: Sequence[torch.Tensor],
    image_tokens: torch.Tensor,  # (batch_size, num_image_tokens)
    annotated_tokens: torch.Tensor,  # (batch_size, num_annotated_tokens)
) -> list[Trace]:
    pass
