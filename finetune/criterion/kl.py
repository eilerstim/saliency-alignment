from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from finetune.criterion import Criterion
from vl_saliency.core.grid import SaliencyGrid


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion using KL divergence."""

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
        scores: list[torch.Tensor] = []

        for b in range(saliency.batch_size):
            mask = masks[b]  # (H, W)
            attn = saliency.maps_for_image(
                batch_idx=b, image_idx=0
            )  # (gen_len, patch_H, patch_W)

            gen_ids = labels[b] != -100
            seg_ids = segment_ids[b][gen_ids, :]  # (gen_len, max_segments)

            # Build target mask
            seg_mask = (
                mask[None, None] == seg_ids[:, :, None, None]
            )  # (gen_len, max_segments, H, W)

            token_mask = seg_mask.any(dim=1)  # (gen_len, H, W)

            pixel_counts = token_mask.sum(dim=(1, 2))
            valid = pixel_counts > 0

            if not valid.any():
                continue
                
            # Upsample to annotation resolution
            attn = F.interpolate(
                attn.unsqueeze(1),
                size=mask.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (gen_len, H, W)


            # Target distribution: uniform over annotated pixels
            target = token_mask.float()

            # Select only valid distributions
            attn = attn[valid]
            target = target[valid]
            
            # Normalize attention per token -> probability distribution
            attn = attn.clamp(min=0)
            attn = attn / attn.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            
            target = target / pixel_counts[:, None, None].clamp(min=1)
            
            # Log attention
            log_attn = torch.log(attn.clamp(min=1e-8))

            # KL(target || attn)
            kl = F.kl_div(log_attn, target, reduction="none").sum(dim=(1, 2))

            scores.append(kl)

        if len(scores) == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.cat(scores).mean()
