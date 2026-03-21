from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from vl_saliency.maps import SaliencyGrid


class Criterion(ABC):
    """Abstract base class for auxiliary loss functions.

    Args:
        weight (float): Weighting factor for the loss. Default is 1.0.

    Methods:
        compute_loss(attn, mask) -> Float[Tensor, "1"]:
            Abstract method to compute the auxiliary loss. Must be implemented by subclasses.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def __call__(
        self,
        labels: Int[Tensor, "B S"],
        segment_ids: Int[Tensor, "B S M"],
        preds: Float[Tensor, "B T V"],
        saliency: SaliencyGrid,
        masks: list[Tensor],
    ) -> Float[Tensor, "1"]:
        """Compute the auxiliary loss.

        Args:
            labels (torch.Tensor): Ground truth labels.  [batch_size, seq_len]
            segment_ids (torch.Tensor): Corresponding image segments for each item.
            preds (torch.Tensor): The model prediction logits. [batch_size, seq_len, vocab_size]
            saliency (SaliencyGrid): The attention weights from the model.
            masks (list[torch.Tensor]): Binary annotation masks. List of [H, W]. Zero tensors indicate no annotation.

        Returns:
            Tensor: Computed auxiliary loss.
        """

        losses: list[Tensor] = []
        for b in range(saliency.batch_size):
            mask = masks[b]  # (H, W)
            attn = saliency.maps_for_image(
                batch_idx=b, image_idx=0
            )  # (gen_len, patch_H, patch_W)

            gen_ids = labels[b] != -100
            seg_ids = segment_ids[b][gen_ids]  # (gen_len, max_segments)

            gen_len = seg_ids.shape[0]
            attn = attn[-gen_len:]

            # Filter tokens with segments only
            has_segments = (seg_ids != -1).any(dim=1)
            if not has_segments.any():
                continue

            seg_ids = seg_ids[has_segments]
            attn = attn[has_segments]

            # Upsample attn to match annotation size (gen_len, H, W)
            attn = F.interpolate(
                attn.unsqueeze(1),
                size=mask.shape,
                mode="bilinear",
                align_corners=False,  # type: ignore
            ).squeeze(1)

            # Normalize per generated token
            denom = attn.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            attn = attn / denom

            # Build mask without -1
            valid_seg = seg_ids != -1  # (N, max_segments)
            seg_mask = (mask[None, None] == seg_ids[:, :, None, None]) & valid_seg[
                :, :, None, None
            ]  # (gen_len, max_segments, H, W)

            # merge annotations per token -> (gen_len, H, W)
            token_mask = seg_mask.any(dim=1)

            loss = self.compute_loss(attn, token_mask)
            losses.append(losses)

        if len(losses) == 0:
            return torch.tensor(0.0, device=preds.device)

        loss = torch.cat(losses).mean()
        return self.weight * loss

    @abstractmethod
    def compute_loss(
        self, attn: Float[Tensor, "S H W"], mask: Bool[Tensor, "S H W"]
    ) -> Float[Tensor, "1"]:
        """Compute the auxiliary loss. Passed in only
        attention and masks with valid segments.

        Args:
            attn (torch.Tensor): Normalized attention scores.
            mask (list[torch.Tensor]): Binary annotation mask.

        Returns:
            Tensor: Computed auxiliary loss.
        """
        ...
