"""Annotation-aware tokenization for COCONut captions with segment references."""

import re

from transformers import PreTrainedTokenizer

# Precompile — avoids re.compile on every call to parse_annotated_caption
_ANNOTATION_RE = re.compile(r"<\s*([\d,\s]+)\s*:\s*([^>]+)>")


def parse_annotated_caption(caption: str) -> list[tuple[list[int], str]]:
    """Parse a caption with segment annotations into (annotation_ids, text) pairs.

    Captions contain references to segments in the format "<id: description>" or
    "<id1,id2,id3: description>" for multiple segments.
    For example: "I saw <1: a dog> and <2: a cat>" becomes:
    [([],"I saw "), ([1], "a dog"), ([], " and "), ([2], "a cat")]

    And "< 62,63,48: Additional people>" becomes:
    [([62, 63, 48], "Additional people")]

    Args:
        caption: Caption string with annotations in format "<id: text>" or
            "<id1,id2,...: text>".

    Returns:
        List of (annotation_ids, text) tuples where annotation_ids is an empty
        list for non-annotated text and a list of segment IDs for annotated spans.
    """
    result: list[tuple[list[int], str]] = []
    last_end = 0

    for match in _ANNOTATION_RE.finditer(caption):
        # Add any text before this annotation (with empty annotation list)
        if match.start() > last_end:
            text_before = caption[last_end : match.start()]
            if text_before:
                result.append(([], text_before))

        # Parse the comma-separated segment IDs
        raw_ids = match.group(1)
        annotated_text = match.group(2)

        # Split by comma, strip spaces, filter valid integers
        # We add 1 to each ID to match the IDs in segment_infos and masks
        id_list = [
            int(x) + 1
            for x in (part.strip() for part in raw_ids.split(","))
            if x and x.isdigit()
        ]

        result.append((id_list, annotated_text))
        last_end = match.end()

    # Add any remaining text after the last annotation
    if last_end < len(caption):
        text_after = caption[last_end:]
        if text_after:
            result.append(([], text_after))

    return result


def tokenize_from_parsed(
    parsed_segments: list[tuple[list[int], str]],
    tokenizer: PreTrainedTokenizer,
    add_special_tokens: bool = False,
) -> tuple[list[int], list[list[int]]]:
    """Tokenize already-parsed segments while preserving annotation info.

    This avoids re-parsing when the caller has already called
    ``parse_annotated_caption``.

    Args:
        parsed_segments: Output of ``parse_annotated_caption``.
        tokenizer: HuggingFace tokenizer to use.
        add_special_tokens: Whether to add special tokens (BOS/EOS).

    Returns:
        Tuple of (token_ids, annotation_ids) where:
        - token_ids: List of token IDs for the full caption.
        - annotation_ids: List of annotation-ID lists, one per token.
    """
    all_token_ids: list[int] = []
    all_annotation_ids: list[list[int]] = []

    for annotation_id_list, text in parsed_segments:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        # All tokens in this segment share the same annotation list
        all_token_ids.extend(token_ids)
        all_annotation_ids.extend([annotation_id_list] * len(token_ids))

    if add_special_tokens:
        if tokenizer.bos_token_id is not None:
            all_token_ids.insert(0, tokenizer.bos_token_id)
            all_annotation_ids.insert(0, [])
        if tokenizer.eos_token_id is not None:
            all_token_ids.append(tokenizer.eos_token_id)
            all_annotation_ids.append([])

    return all_token_ids, all_annotation_ids


def tokenize_with_annotations(
    caption: str,
    tokenizer: PreTrainedTokenizer,
    add_special_tokens: bool = False,
) -> tuple[list[int], list[list[int]]]:
    """Tokenize a caption while preserving segment annotation information.

    Convenience wrapper that parses *and* tokenizes in one call.

    Args:
        caption: Caption string with annotations in format "<id: text>" or
            "<id1,id2,...: text>".
        tokenizer: HuggingFace tokenizer to use.
        add_special_tokens: Whether to add special tokens (BOS/EOS).

    Returns:
        Tuple of (token_ids, annotation_ids) where:
        - token_ids: List of token IDs for the full caption.
        - annotation_ids: List of annotation-ID lists, one per token.
    """
    segments = parse_annotated_caption(caption)
    return tokenize_from_parsed(segments, tokenizer, add_special_tokens)


def batch_tokenize_with_annotations(
    captions: list[str],
    tokenizer: PreTrainedTokenizer,
    padding: bool = True,
    max_length: int | None = None,
    add_special_tokens: bool = False,
) -> tuple[list[list[int]], list[list[list[int]]]]:
    """Tokenize a batch of captions with annotation tracking.

    Args:
        captions: List of caption strings with annotations.
        tokenizer: HuggingFace tokenizer to use.
        padding: Whether to pad sequences to the same length.
        max_length: Maximum sequence length (truncates if exceeded).
        add_special_tokens: Whether to add special tokens (BOS/EOS).

    Returns:
        Tuple of (batch_token_ids, batch_annotation_ids) where:
        - batch_token_ids: List of token ID lists [batch_size, seq_len]
        - batch_annotation_ids: List of annotation ID lists
          [batch_size, seq_len, num_regions]
    """
    batch_token_ids = []
    batch_annotation_ids = []

    for caption in captions:
        token_ids, annotation_ids = tokenize_with_annotations(
            caption, tokenizer, add_special_tokens=add_special_tokens
        )

        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            annotation_ids = annotation_ids[:max_length]

        batch_token_ids.append(token_ids)
        batch_annotation_ids.append(annotation_ids)

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
                batch_token_ids[i] += [pad_token_id] * padding_length
                batch_annotation_ids[i] += [[]] * padding_length

    return batch_token_ids, batch_annotation_ids


def map_annotations_to_segments(
    annotation_ids: list[list[int]], segments_info: list[tuple[int, int]]
) -> list[list[int]]:
    """Map annotation IDs to segment category IDs.

    Converts annotation tracking (reference numbers from the caption like
    ``<1: ...>``, ``<2,3: ...>``) to category IDs from the segmentation mask.

    Args:
        annotation_ids: List of annotation ID lists per token.
        segments_info: List of (segment_id, category_id) tuples from COCONut.

    Returns:
        List of category ID lists corresponding to each token's annotations.
    """
    segment_to_category = {seg_id: cat_id for seg_id, cat_id in segments_info}

    return [
        [segment_to_category[aid] for aid in ann_ids if aid in segment_to_category]
        for ann_ids in annotation_ids
    ]
