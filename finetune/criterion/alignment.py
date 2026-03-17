from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from finetune.criterion import Criterion
from vl_saliency.maps import SaliencyGrid


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion.

    This criterion computes the saliency alignment loss between the model's
    saliency maps and the provided annotation.

    - Uses Mean Squared Error (MSE) loss for alignment.
    """

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
        # Compute saliency alignment loss
        scores: list[torch.Tensor] = []
        for b in range(saliency.batch_size):
            mask = masks[b]  # (H, W)
            attn = saliency.maps_for_image(
                batch_idx=b, image_idx=0
            )  # (gen_len, patch_H, patch_W)

            gen_ids = labels[b] != -100
            seg_ids = segment_ids[b][gen_ids, :]  # (gen_len, max_segments)
            
            gen_len = gen_ids.sum().item()
            attn = attn[-gen_len:]

            # Upsample attn to match annotation size (gen_len, H, W)
            attn = F.interpolate(
                attn.unsqueeze(1),
                size=mask.shape,
                mode="bilinear",
                align_corners=False,  # type: ignore
            ).squeeze(1)

            # Normalize per generated token
            attn = attn / (attn.sum(dim=(1, 2), keepdim=True) + 1e-8)

            seg_mask = (
                mask[None, None] == seg_ids[:, :, None, None]
            )  # (gen_len, max_segments, H, W)

            # merge annotations per token -> (gen_len, H, W)
            token_mask = seg_mask.any(dim=1)

            pixel_counts = token_mask.sum(dim=(1, 2))
            valid = token_mask.sum(dim=(1, 2)) > 0  # (gen_len,)

            diff = (attn - 1.0) ** 2
            loss_per_token = (diff * token_mask).sum(dim=(1, 2)) / pixel_counts.clamp(
                min=1
            )

            scores.append(loss_per_token[valid].flatten())

        if len(scores) == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.cat(scores).mean()
