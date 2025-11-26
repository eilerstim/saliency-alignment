from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
import vl_saliency.transforms as T

from finetune.criterion import Criterion
from finetune.criterion.utils import trace_attentions


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion.

    This criterion computes the saliency alignment loss between the model's
    saliency maps and the provided annotation.

    Currently:
    - Uses Mean Squared Error (MSE) loss for alignment.
    - Localization Heads > Mean Aggregation > Normalization
    """

    def __init__(self, weight: float = 1.0, image_token_id: int = 1) -> None:
        """
        Args:
            weight (float): Weight for the saliency alignment loss.
            image_token_id (int): Token ID representing the image token in the input.
        """
        super().__init__(weight)
        self.image_token_id = image_token_id

    def compute_loss(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: torch.Tensor,
        **kwargs: Any,
    ) -> float:
        # Extract attention traces for image tokens
        traces = trace_attentions(
            input_ids=input_ids,
            attentions=attentions,
            image_token_id=self.image_token_id,
            image_shapes=[mask.shape[-2:] for mask in masks],
        )

        # Compute saliency alignment loss
        scores = total = 0.0
        for b, trace in enumerate(traces):
            _, _, gen_len, h, w = trace.attns[0].shape

            # Saliency map computation pipeline
            pipe = (
                T.LocalizationHeads()
                >> T.Aggregate()
                >> T.normalize()  # Or: T.Binarize()
                >> T.Upscale(h, w, mode="bilinear")
            )

            for i in range(gen_len):
                annotation = masks[b, i]  # (H, W)
                if annotation.sum() == 0:
                    continue

                saliency_map = trace.map(token=i) >> pipe  # (H, W)

                # Compute alignment loss
                loss = F.mse_loss(saliency_map, annotation.float())

                scores += loss.item()
                total += 1

        if total == 0:
            return 0.0

        return scores / total
