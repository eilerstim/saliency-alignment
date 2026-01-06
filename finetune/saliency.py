from __future__ import annotations

import math
from functools import partial, wraps

import torch
from transformers import AttentionInterface, PreTrainedModel
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from .transformer_utils import _get_image_token_id


def trace_model(model: PreTrainedModel, patch_shape: tuple[int, int]):
    """
    Trace the model to optimize saliency computation.

    Args:
        model (PreTrainedModel): The model to trace.
        patch_shape (tuple[int, int]): Shape of the image patches (height, width).
    """
    accum = SaliencyAccumulator(patch_shape)
    model._accum = accum
    model._scale = 1.0 / math.sqrt(
        model.config.text_config.hidden_size
        // model.config.text_config.num_attention_heads
    )
    model._image_token_id = _get_image_token_id(model.config)

    partial_spda_saliency = partial(
        spda_saliency,
        model=model,
    )

    # Register a unique attention implementation to avoid conflicts
    unique_name = f"sdpa_saliency_{id(model)}"
    AttentionInterface.register(unique_name, partial_spda_saliency)
    model.set_attn_implementation({"text_config": unique_name})

    model.forward_original = model.forward

    @wraps(model.forward_original)
    def wrapped_forward(self, *args, **kwargs):
        accum.reset()

        input_ids = kwargs.get("input_ids")
        labels = kwargs.get("labels", None)

        self._img_tokens = input_ids == self._image_token_id

        if labels is not None:
            self._gen_tokens = labels != -100
        else:
            # indices [0, 1, ..., L-1]
            idx = torch.arange(input_ids.shape[1], device=input_ids.device)

            # set non-image positions to -1, then take max
            last_img_idx = torch.where(self._img_tokens, idx, -1).amax(dim=1)

            self._gen_tokens = torch.zeros_like(input_ids, dtype=torch.bool)
            self._gen_tokens = idx.unsqueeze(0) > last_img_idx.unsqueeze(1)

            # handle batches with no image tokens
            self._gen_tokens[last_img_idx == -1] = False

        return self.forward_original(*args, **kwargs)

    # bind the method back to the model
    model.forward = wrapped_forward.__get__(model, type(model))


def get_maps(model: PreTrainedModel) -> list[torch.Tensor]:
    """
    Retrieve the accumulated saliency maps from the model.

    Args:
        model (PreTrainedModel): The model with accumulated saliency maps.
    Returns:
        list[torch.Tensor]: List of accumulated saliency maps for each batch item.
    """
    return model._accum.get_maps()


class SaliencyAccumulator:
    """
    Accumulates saliency maps over multiple steps,
    preventing memory overflow from output_attentions.

    Args:
        patch_shapes (tuple[int, int] | list[tuple[int, int]]): Shape of the image patches (height, width).
    """

    def __init__(self, patch_shapes: tuple[int, int] | list[tuple[int, int]]):
        self.patch_shapes = patch_shapes
        self.maps: list[torch.Tensor] | None = None

    def accumulate(self, xs: list[torch.Tensor]):
        """
        Accumulate a new saliency map.

        Args:
            xs (list[torch.Tensor]): List of saliency maps to accumulate, one per batch item.
        """
        if self.maps is None:
            if isinstance(self.patch_shapes, list) and len(self.patch_shapes) != len(
                xs
            ):
                raise ValueError(
                    "Length of patch_shapes must match number of saliency maps."
                )
            self.maps = xs
        else:
            if len(self.maps) != len(xs):
                raise ValueError(
                    "Number of saliency maps to accumulate does not match existing maps."
                )
            for i, x in enumerate(xs):
                self.maps[i] = self.maps[i] + x

    def get_maps(self) -> list[torch.Tensor]:
        """
        Retrieve the accumulated saliency maps.

        Returns:
            list[torch.Tensor]: List of accumulated saliency maps for each batch item.
        """
        if self.maps is None:
            return []  # For now for the initial val check
            # raise ValueError("Saliency maps have not been accumulated.")

        if isinstance(self.patch_shapes, tuple):
            patch_shapes = [self.patch_shapes] * len(self.maps)
        else:
            patch_shapes = self.patch_shapes

        return [
            v.reshape(v.size(0), *patch)
            for v, patch in zip(self.maps, patch_shapes, strict=True)
        ]

    def reset(self):
        """Reset the accumulated saliency maps."""
        self.maps = None


# @torch.compile(mode="reduce-overhead")
def _saliency_from_attentions(
    q: torch.Tensor,
    k: torch.Tensor,
    img_tokens: torch.Tensor,
    gen_tokens: torch.Tensor,
    scale: float,
):
    Hq = q.shape[0]
    Hkv = k.shape[0]

    # Handle GQA: expand KV heads to match Q heads
    if Hq != Hkv:
        assert Hq % Hkv == 0
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=0)  # [Hq, S, D]

    q_txt = q[:, gen_tokens, :]  # [Hq, T_text, D]
    k_img = k[:, img_tokens, :]  # [Hq, S_img, D]
    scores = (q_txt @ k_img.transpose(-2, -1)) * scale  # [Hq, T_text, S_img]
    return scores.mean(0)  # [T_text, S_img]


def spda_saliency(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    model: PreTrainedModel = None,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Drop-in replacement for F.scaled_dot_product_attention that will record
    saliency *if* a saliency_context is active.

    q,k,v expected: [B, H, S, D]
    """
    if model._accum is None:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        )
    maps: list[torch.Tensor] = [
        _saliency_from_attentions(
            query[i],
            key[i],
            model._img_tokens[i],
            model._gen_tokens[i],
            model._scale,
        ).clone()
        for i in range(query.size(0))
    ]

    model._accum.accumulate(maps)
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        *args,
        **kwargs,
    )
