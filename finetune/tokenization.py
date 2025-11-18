"""Annotation-aware tokenization for COCONut captions with segment references."""

import re

from transformers import PreTrainedTokenizer


def parse_annotated_caption(caption: str) -> list[tuple[int, str]]:
    """Parse a caption with segment annotations into (annotation_id, text) pairs.

    Captions contain references to segments in the format "<id: description>".
    For example: "I saw <1: a dog> and <2: a cat>" becomes:
    [(0, "I saw "), (1, "a dog"), (0, " and "), (2, "a cat")]

    Args:
        caption: Caption string with annotations in format "<id: text>".

    Returns:
        List of (annotation_id, text) tuples where annotation_id is 0 for
        non-annotated text and the segment ID for annotated spans.
    """
    # Pattern to match annotations like "<1: a dog>"
    pattern = r"<(\d+):\s*([^>]+)>"

    result = []
    last_end = 0

    for match in re.finditer(pattern, caption):
        # Add any text before this annotation (with annotation_id=0)
        if match.start() > last_end:
            text_before = caption[last_end : match.start()]
            if text_before:  # Only add if non-empty
                result.append((0, text_before))

        # Add the annotated text with its segment ID
        segment_id = int(match.group(1))
        annotated_text = match.group(2)
        result.append((segment_id, annotated_text))

        last_end = match.end()

    # Add any remaining text after the last annotation
    if last_end < len(caption):
        text_after = caption[last_end:]
        if text_after:
            result.append((0, text_after))

    return result


def tokenize_with_annotations(
    caption: str, tokenizer: PreTrainedTokenizer, add_special_tokens: bool = False
) -> tuple[list[int], list[int]]:
    """Tokenize a caption while preserving segment annotation information.

    Args:
        caption: Caption string with annotations in format "<id: text>".
        tokenizer: HuggingFace tokenizer to use.
        add_special_tokens: Whether to add special tokens (BOS/EOS).

    Returns:
        Tuple of (token_ids, annotation_ids) where:
        - token_ids: List of token IDs for the full caption
        - annotation_ids: List of segment IDs (0 for non-annotated text)
          corresponding to each token
    """
    # Parse caption into annotated segments
    segments = parse_annotated_caption(caption)

    all_token_ids = []
    all_annotation_ids = []

    # Tokenize each segment and track which annotation it belongs to
    for annotation_id, text in segments:
        # Tokenize without special tokens - we'll add them once at the end if needed
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Record which annotation each token belongs to
        annotation_ids = [annotation_id] * len(token_ids)

        all_token_ids.extend(token_ids)
        all_annotation_ids.extend(annotation_ids)

    # Add special tokens if requested
    if add_special_tokens:
        if tokenizer.bos_token_id is not None:
            all_token_ids = [tokenizer.bos_token_id] + all_token_ids
            all_annotation_ids = [0] + all_annotation_ids  # BOS gets annotation 0
        if tokenizer.eos_token_id is not None:
            all_token_ids = all_token_ids + [tokenizer.eos_token_id]
            all_annotation_ids = all_annotation_ids + [0]  # EOS gets annotation 0

    return all_token_ids, all_annotation_ids


def batch_tokenize_with_annotations(
    captions: list[str],
    tokenizer: PreTrainedTokenizer,
    padding: bool = True,
    max_length: int = None,
    add_special_tokens: bool = False,
) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenize a batch of captions with annotation tracking.

    Args:
        captions: List of caption strings with annotations.
        tokenizer: HuggingFace tokenizer to use.
        padding: Whether to pad sequences to the same length.
        max_length: Maximum sequence length (truncates if exceeded).
        add_special_tokens: Whether to add special tokens (BOS/EOS).

    Returns:
        Tuple of (batch_token_ids, batch_annotation_ids) where each is a list
        of lists with the same shape.
    """
    batch_token_ids = []
    batch_annotation_ids = []

    for caption in captions:
        token_ids, annotation_ids = tokenize_with_annotations(
            caption, tokenizer, add_special_tokens=add_special_tokens
        )

        # Apply max_length truncation if specified
        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            annotation_ids = annotation_ids[:max_length]

        batch_token_ids.append(token_ids)
        batch_annotation_ids.append(annotation_ids)

    # Apply padding if requested
    if padding:
        max_len = max(len(seq) for seq in batch_token_ids)
        if max_length is not None:
            max_len = min(max_len, max_length)

        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        for i in range(len(batch_token_ids)):
            padding_length = max_len - len(batch_token_ids[i])
            if padding_length > 0:
                batch_token_ids[i] = (
                    batch_token_ids[i] + [pad_token_id] * padding_length
                )
                batch_annotation_ids[i] = batch_annotation_ids[i] + [0] * padding_length

    return batch_token_ids, batch_annotation_ids


def map_annotations_to_segments(
    annotation_ids: list[int], segments_info: list[tuple[int, int]]
) -> list[int]:
    """Map annotation IDs to segment category IDs.

    This converts the annotation tracking (which uses the reference numbers from
    the caption like <1: ...>, <2: ...>) to the actual category IDs from the
    segmentation mask.

    Args:
        annotation_ids: List of annotation IDs (0 for non-annotated, 1, 2, ... for refs).
        segments_info: List of (segment_id, category_id) tuples from COCONut.

    Returns:
        List of category IDs corresponding to each annotation ID.
        Returns 0 for non-annotated tokens and for annotations without matches.
    """
    # Build mapping from annotation number to category_id
    # segments_info contains (id, category_id) tuples
    # We assume annotation <1: ...> refers to the first segment, <2: ...> to second, etc.
    annotation_to_category = {0: 0}  # Non-annotated text maps to category 0

    for idx, (_seg_id, category_id) in enumerate(segments_info, start=1):
        annotation_to_category[idx] = category_id

    # Map each annotation ID to its category
    category_ids = [annotation_to_category.get(ann_id, 0) for ann_id in annotation_ids]

    return category_ids
