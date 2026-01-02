from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from finetune.criterion import Criterion


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion.

    This criterion computes the saliency alignment loss between the model's
    saliency maps and the provided annotation.

    Currently:
    - Uses Mean Squared Error (MSE) loss for alignment.
    - Localization Heads > Mean Aggregation > Normalization
    """

    def compute_loss(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        # Compute saliency alignment loss
        scores = []
        for trace, annotations in zip(attentions, masks, strict=True):
            # Normalize per generated token (gen_len, patch_H, patch_W)
            trace = trace / (trace.sum(dim=(1, 2), keepdim=True) + 1e-8)

            # Upsample trace to match annotation size (gen_len, H, W)
            trace = F.interpolate(
                trace.unsqueeze(1),
                size=annotations.shape[1:],
                mode="bilinear",
                align_corners=False,  # type: ignore
            ).squeeze(1)

            # Identify valid tokens with annotations
            valid = annotations.flatten(1).sum(dim=1) > 0  # (gen_len,)

            if valid.any():
                # Select only valid tokens
                trace = trace[valid]  # (num_valid, h, w)
                annotations = annotations[valid]  # (num_valid, h, w)

                # Compute loss for each valid token
                loss = F.mse_loss(
                    trace, annotations, reduction="none"
                )  # (num_valid, h, w)
                loss = loss.mean(dim=(1, 2))  # (num_valid,)
                scores.append(loss)

        if len(scores) == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.cat(scores).mean()
