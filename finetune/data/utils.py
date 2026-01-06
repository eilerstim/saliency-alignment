"""Shared utilities for data collation."""

import torch


def find_sequence(tensor: torch.Tensor, sequence: list[int]) -> int:
    """Find the starting index of a token sequence in a 1D tensor.
    
    Args:
        tensor: 1D tensor of token IDs to search in.
        sequence: List of token IDs to search for.
    
    Returns:
        Starting index of the sequence, or -1 if not found.
    """
    seq_len = len(sequence)
    seq_tensor = torch.tensor(sequence, dtype=tensor.dtype, device=tensor.device)
    for i in range(len(tensor) - seq_len + 1):
        if torch.equal(tensor[i:i + seq_len], seq_tensor):
            return i
    return -1
