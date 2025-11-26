from collections.abc import Sequence

import torch
from vl_saliency import Trace


def trace_attentions(
    input_ids: torch.Tensor,
    attentions: Sequence[torch.Tensor],
    image_token_id: int,
    image_shapes: list[tuple[int, int]],
) -> list[Trace]:
    """Trace attentions from output tokens back to input tokens.

    Args:
        input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
        attentions (Sequence[torch.Tensor]): List of attention tensors from each layer,
            each of shape (batch_size, num_heads, seq_len, seq_len).
        image_token_id (int): Token ID representing the image token in the input.
        image_shapes (list[tuple[int, int]]): List of (height, width) tuples for each image
            in the batch.
    Returns:
        List[Trace]: List of Trace objects for each example in the batch.
    """

    # Extract image token positions
    mask = input_ids == image_token_id  # (batch_size, seq_len)
    image_tokens_per_sample = [row[row_mask] for row, row_mask in zip(input_ids, mask)]

    # Get generated token positions
    image_ends = (mask.size(1) - 1) - mask.flip(1).float().argmax(dim=1)
    gen_tokens = [input_ids[i, image_ends[i] + 1 :] for i in range(input_ids.size(0))]

    # Split attentions per sample
    stacked = torch.stack(
        [a.detach().cpu() for a in attentions], dim=0
    )  # (num_layers, batch_size, num_heads, seq_len, seq_len)
    attns = torch.unbind(
        stacked, dim=1
    )  # batch_size * (num_layers, num_heads, seq_len, seq_len)

    # Collect traces
    traces = []
    for i, attn in enumerate(attns):
        # Get attention weights to image tokens
        attn_to_image_tokens = [  # (num_layers, num_heads, gen_len, image_len)
            layer_attn[:, :, gen_tokens[i], image_tokens_per_sample[i]]
            for layer_attn in attn
        ]

        # Reshape attention weights to spatial dimensions
        h, w = image_shapes[i]
        attn_to_image_tokens = [  # (num_layers, num_heads, gen_len, h, w)
            layer_attn.contiguous().view(*layer_attn.shape[:-1], h, w)
            for layer_attn in attn_to_image_tokens
        ]

        # Create Trace object
        traces.append(Trace(attn_to_image_tokens))

    return traces
