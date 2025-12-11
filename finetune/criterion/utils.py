from collections.abc import Generator, Sequence

import torch


def image_attentions(
    input_ids: torch.Tensor,
    attentions: Sequence[torch.Tensor],
    image_token_id: int,
    image_shape: tuple[int, int],
) -> Generator[torch.Tensor, None, None]:
    """Generates attention maps from generated tokens to image tokens in the input.

    Args:
        input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
        attentions (Sequence[torch.Tensor]): List of attention tensors from each layer,
            each of shape (batch_size, num_heads, seq_len, seq_len).
        image_token_id (int): Token ID representing the image token in the input.
        image_shapes (tuple[int, int]): Shape of image patches (height, width).

    Returns:
        Generator[torch.Tensor, None, None]: Yields attention maps of shape (gen_len, H, W)
            for each sample in the batch.
    """

    # Extract image token positions
    mask = input_ids == image_token_id  # (batch_size, seq_len)
    image_tokens_per_sample = [
        row[row_mask] for row, row_mask in zip(input_ids, mask, strict=True)
    ]

    # Get generated token positions
    image_ends = (mask.size(1) - 1) - mask.flip(1).float().argmax(dim=1)
    gen_tokens = [input_ids[i, image_ends[i] + 1 :] for i in range(input_ids.size(0))]

    # Average attentions over heads and layers (batch_size, seq_len, seq_len)
    stacked = torch.stack(attentions, dim=0)
    attns = torch.mean(stacked, dim=(0, 2))

    # Build attention maps for each sample
    H, W = image_shape
    for i in range(input_ids.size(0)):
        ai = attns[i, gen_tokens[i], image_tokens_per_sample[i]]  # (gen_len, image_len)
        ai = ai.contiguous().view(len(gen_tokens[i]), H, W)  # (gen_len, H, W)
        yield ai
