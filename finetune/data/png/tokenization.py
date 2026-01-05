"""Tokenization utilities for PNG-COCO captions with segment ID tracking.

This module provides functions to tokenize captions from the PNG-COCO dataset
while tracking which tokens correspond to which segment IDs. Unlike the COCONut
tokenization which uses annotation markers like <id: text>, PNG-COCO provides
pre-parsed segments with associated segment_ids.

Note: PNG-COCO utterances don't include inter-word spaces, so we add them
when concatenating (except before punctuation).
"""

import re

from transformers import PreTrainedTokenizer

# Punctuation that shouldn't have a space before it
_PUNCT_NO_SPACE_BEFORE = re.compile(r'^[.,;:!?\'")\]}]+')


def _needs_space_before(utterance: str) -> bool:
    """Check if we need to add a space before this utterance."""
    return not _PUNCT_NO_SPACE_BEFORE.match(utterance)


def tokenize_png_caption(
    segments: list[dict],
    tokenizer: PreTrainedTokenizer,
    add_special_tokens: bool = False,
) -> tuple[list[int], list[list[int]]]:
    """Tokenize a PNG-COCO caption while preserving segment ID information.

    PNG-COCO provides captions as a list of segments, where each segment has:
    - utterance: The text of this segment
    - segment_ids: List of segment IDs (as strings) that this text refers to
    - plural: Whether the noun is plural
    - noun: Whether this segment contains a noun

    Note: Spaces are automatically added between utterances since PNG-COCO
    doesn't include inter-word spacing. Spaces are not added before punctuation.

    Args:
        segments: List of segment dictionaries from PNG-COCO annotation.
        tokenizer: HuggingFace tokenizer to use.
        add_special_tokens: Whether to add special tokens (BOS/EOS).

    Returns:
        Tuple of (token_ids, segment_ids_per_token) where:
        - token_ids: List of token IDs for the full caption
        - segment_ids_per_token: List of lists of segment IDs (as integers)
          corresponding to each token. Empty list for tokens without segments.
    """
    all_token_ids = []
    all_segment_ids = []

    for i, segment in enumerate(segments):
        utterance = segment["utterance"]
        segment_ids = segment.get("segment_ids", [])

        # Convert segment IDs from strings to integers
        segment_id_ints = [int(sid) for sid in segment_ids if sid]

        # Add space before utterance if needed (not first, not punctuation)
        if i > 0 and _needs_space_before(utterance):
            # Tokenize the space separately (no segment IDs)
            space_tokens = tokenizer.encode(" ", add_special_tokens=False)
            all_token_ids.extend(space_tokens)
            all_segment_ids.extend([[] for _ in space_tokens])

        # Tokenize the utterance
        token_ids = tokenizer.encode(utterance, add_special_tokens=False)

        # Each token in this segment gets the same segment IDs
        segment_ids_for_tokens = [segment_id_ints.copy() for _ in token_ids]

        all_token_ids.extend(token_ids)
        all_segment_ids.extend(segment_ids_for_tokens)

    # Add special tokens if requested
    if add_special_tokens:
        if tokenizer.bos_token_id is not None:
            all_token_ids = [tokenizer.bos_token_id] + all_token_ids
            all_segment_ids = [[]] + all_segment_ids
        if tokenizer.eos_token_id is not None:
            all_token_ids = all_token_ids + [tokenizer.eos_token_id]
            all_segment_ids = all_segment_ids + [[]]

    return all_token_ids, all_segment_ids


def build_clean_caption(segments: list[dict]) -> str:
    """Build a clean caption string from PNG-COCO segments.

    Adds spaces between utterances (except before punctuation) since
    PNG-COCO utterances don't include inter-word spacing.

    Args:
        segments: List of segment dictionaries from PNG-COCO annotation.

    Returns:
        The full caption as a single string with proper spacing.
    """
    parts = []
    for i, segment in enumerate(segments):
        utterance = segment["utterance"]
        if i > 0 and _needs_space_before(utterance):
            parts.append(" ")
        parts.append(utterance)
    return "".join(parts)


def batch_tokenize_png_captions(
    batch_segments: list[list[dict]],
    tokenizer: PreTrainedTokenizer,
    padding: bool = True,
    max_length: int | None = None,
    add_special_tokens: bool = False,
) -> tuple[list[list[int]], list[list[list[int]]]]:
    """Tokenize a batch of PNG-COCO captions with segment ID tracking.

    Args:
        batch_segments: List of segment lists, one per caption.
        tokenizer: HuggingFace tokenizer to use.
        padding: Whether to pad sequences to the same length.
        max_length: Maximum sequence length (truncates if exceeded).
        add_special_tokens: Whether to add special tokens (BOS/EOS).

    Returns:
        Tuple of (batch_token_ids, batch_segment_ids) where:
        - batch_token_ids: List of token ID lists [batch_size, seq_len]
        - batch_segment_ids: List of segment ID lists [batch_size, seq_len, num_segments]
    """
    batch_token_ids = []
    batch_segment_ids = []

    for segments in batch_segments:
        token_ids, segment_ids = tokenize_png_caption(
            segments, tokenizer, add_special_tokens=add_special_tokens
        )

        # Apply max_length truncation if specified
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            segment_ids = segment_ids[:max_length]

        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    # Apply padding if requested
    if padding and batch_token_ids:
        max_len = max(len(seq) for seq in batch_token_ids)
        if max_length is not None:
            max_len = min(max_len, max_length)

        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        for i in range(len(batch_token_ids)):
            padding_length = max_len - len(batch_token_ids[i])
            if padding_length > 0:
                batch_token_ids[i] = batch_token_ids[i] + [pad_token_id] * padding_length
                batch_segment_ids[i] = batch_segment_ids[i] + [[]] * padding_length

    return batch_token_ids, batch_segment_ids
