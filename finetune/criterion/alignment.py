from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from finetune.criterion import Criterion


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion.

    This criterion computes the saliency alignment loss between the model's
    saliency maps and the provided annotation.

    - Uses Mean Squared Error (MSE) loss for alignment.
    """

    def compute_loss(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> float:
        # Compute saliency alignment loss
        scores = []
        for b in range(preds.shape[0]):
            mask = masks[b]  # (H, W)
            seg_id = segment_ids[b]  # (seq_len, max_segments)
            attn = attentions[b]  # (gen_len, patch_H, patch_W)

            # Normalize per generated token
            attn = attn / (attn.sum(dim=(1, 2), keepdim=True) + 1e-8)

            # Upsample trace to match annotation size (gen_len, H, W)
            trace = F.interpolate(
                attn.unsqueeze(1),
                size=mask.shape,
                mode="bilinear",
                align_corners=False,  # type: ignore
            ).squeeze(1)

            seg_mask = (
                mask[None, None] == seg_id[:, :, None, None]
            )  # (gen_len, max_segments, H, W)

            # merge annotations per token -> (gen_len, H, W)
            token_mask = seg_mask.any(dim=1)
            valid = token_mask.sum(dim=(1, 2)) > 0  # (gen_len,)

            if not valid.any():
                continue

            # Compute loss for each token with annotations
            loss = F.mse_loss(
                trace[valid], token_mask[valid].float(), reduction="none"
            )  # (num_valid, H, W)

            loss = loss.mean(dim=(1, 2))  # (num_valid,)
            scores.append(loss)

        if len(scores) == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.cat(scores).mean()
