from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from finetune.criterion import Criterion


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion using KL divergence."""

    def compute_loss(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        preds: torch.Tensor,
        attentions: Sequence[torch.Tensor],
        masks: list[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        scores: list[torch.Tensor] = []

        for b in range(preds.shape[0]):
            mask = masks[b]  # (H, W)
            attn = attentions[b]  # (gen_len, patch_H, patch_W)

            gen_ids = labels[b] != -100
            seg_ids = segment_ids[b][gen_ids, :]  # (gen_len, max_segments)

            # Normalize attention per token -> probability distribution
            attn = attn / (attn.sum(dim=(1, 2), keepdim=True) + 1e-8)

            # Upsample to annotation resolution
            attn = F.interpolate(
                attn.unsqueeze(1),
                size=mask.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (gen_len, H, W)

            # Build target mask
            seg_mask = (
                mask[None, None] == seg_ids[:, :, None, None]
            )  # (gen_len, max_segments, H, W)

            token_mask = seg_mask.any(dim=1)  # (gen_len, H, W)

            pixel_counts = token_mask.sum(dim=(1, 2))
            valid = pixel_counts > 0

            if not valid.any():
                continue

            # Target distribution: uniform over annotated pixels
            target = token_mask.float()
            target = target / pixel_counts[:, None, None].clamp(min=1)

            # KL(target || attn)
            kl = F.kl_div(
                (attn + 1e-8).log(),
                target,
                reduction="none",
            ).sum(dim=(1, 2))

            scores.append(kl[valid])

        if len(scores) == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.cat(scores).mean()
