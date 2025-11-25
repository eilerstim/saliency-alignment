from torch import Tensor


def annotation_map_per_token(annotation_ids: Tensor, mask: Tensor) -> Tensor:
    """
    For each token, returns a bitmask of shape [H, W] indicating which annotations
    are associated with that token.

    Args:
        annotation_ids (Tensor): A tensor of shape [batch_size, seq_len, max_annotations] containing
            the annotation IDs for each token in a batch.
        mask (Tensor): A tensor of shape [H, W] where each entry specifies which annotation ID that pixel belongs to.

    Returns:
        Tensor: A tensor of shape [batch_size, seq_len, H, W] where each entry is 1 if the pixel
            belongs to any of the annotations associated with the token, and 0 otherwise.
    """
    raise NotImplementedError("This function needs to be implemented.")
