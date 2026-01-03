import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


class SaliencyAccumulator:
    """
    Accumulates saliency maps over multiple steps,
    preventing memory overflow from output_attentions.

    Args:
        batch_size (int): The size of the batch for which saliency maps are accumulated.
        values (list[torch.Tensor | None]): List to hold accumulated saliency maps for each batch item.
    """

    def __init__(self, batch_size: int, patch_shape: tuple[int, int]):
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.values: list[torch.Tensor | None] = [None] * batch_size

    def reset(self):
        """Reset the accumulated saliency map."""
        self.values = [None] * self.batch_size

    def accumulate(self, xs: list[torch.Tensor]):
        """
        Accumulate a new saliency map.

        Args:
            xs (list[torch.Tensor]): List of saliency maps to accumulate, one per batch item.
        """
        for i, x in enumerate(xs):
            self.values[i] = x if self.values[i] is None else self.values[i] + x

    def get_maps(self) -> list[torch.Tensor]:
        """
        Retrieve the accumulated saliency maps.

        Returns:
            list[torch.Tensor]: List of accumulated saliency maps for each batch item.
        """
        if any(v is None for v in self.values):
            raise ValueError("Saliency maps have not been fully accumulated.")

        return [v.view(v.contiguous().size(0), *self.patch_shape) for v in self.values]


@torch.compile(mode="reduce-overhead")
def _saliency_from_attentions(
    q: torch.Tensor,
    k: torch.Tensor,
    img_start: torch.Tensor,
    text_start: torch.Tensor,
    scale: float,
):
    Hq = q.shape[0]
    Hkv = k.shape[0]

    # Handle GQA: expand KV heads to match Q heads
    if Hq != Hkv:
        assert Hq % Hkv == 0
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=0)  # [Hq, S, D]

    q_txt = q[:, text_start:, :]  # [Hq, T_text, D]
    k_img = k[:, img_start:text_start, :]  # [Hq, S_img, D]
    scores = (q_txt @ k_img.transpose(-2, -1)) * scale  # [Hq, T_text, S_img]
    return scores.mean(0)  # [T_text, S_img]


orig = F.scaled_dot_product_attention


@contextmanager
def sdpa_saliency(
    accum: SaliencyAccumulator,
    img_start: torch.Tensor,
    text_start: torch.Tensor,
    head_dim: int,
):
    """
    Augments the standard scaled dot-product attention with accumulated saliency maps.

    Args:
        accum (SaliencyAccumulator): The accumulator for saliency maps.
        img_start (torch.Tensor): Tensor of image token start indices per batch item.
        text_start (torch.Tensor): Tensor of text token start indices per batch item.
        head_dim (int): Dimension of each attention head.
    """

    scale = 1.0 / math.sqrt(head_dim)

    def wrapped_sdpa(q, k, v, *args, **kwargs):
        # q, k: [batch_size, num_heads, seq_len, head_dim]

        if not kwargs.get("is_causal", False):  # e.g. ViT
            return orig(q, k, v, *args, **kwargs)

        saliency_maps: list[torch.Tensor] = [
            _saliency_from_attentions(q[i], k[i], img_start[i], text_start[i], scale)
            for i in range(q.size(0))
        ]
        accum.accumulate(saliency_maps)
        return orig(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = wrapped_sdpa
    try:
        yield
    finally:
        pass
        # F.scaled_dot_product_attention = orig
