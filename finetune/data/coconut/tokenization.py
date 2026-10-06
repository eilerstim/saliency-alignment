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

        # Split by comma, strip spaces, filter valid integers.
        # Caption IDs already match the values stored in segment_infos and
        # the panoptic mask (mask value 0 = void, 1..N = segments).
        id_list = [
            int(x)
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


def annotation_spans(
    parsed_segments: list[tuple[list[int], str]],
) -> tuple[str, list[tuple[int, int, list[int]]]]:
    """Clean caption text and the character span of every annotated piece.

    Args:
        parsed_segments: Output of ``parse_annotated_caption``.

    Returns:
        Tuple of (clean_caption, spans) where ``spans`` holds
        ``(start, end, annotation_ids)`` character ranges into
        ``clean_caption`` for the annotated pieces only.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, list[int]]] = []
    pos = 0
    for annotation_ids, text in parsed_segments:
        if annotation_ids:
            spans.append((pos, pos + len(text), annotation_ids))
        parts.append(text)
        pos += len(text)
    return "".join(parts), spans


def align_annotations_to_offsets(
    offsets: list[tuple[int, int]],
    text: str,
    spans: list[tuple[int, int, list[int]]],
) -> list[list[int]]:
    """Assign annotation IDs to tokens of an *existing* tokenization.

    Tokenizing each annotated piece on its own (``tokenize_from_parsed``)
    does not reproduce the tokens the model sees: SentencePiece-style
    tokenizers prepend a word boundary to every call, so a piece such as
    ``" and "`` encodes to ``['▁', '▁and', '▁']`` in isolation but to
    ``['▁and']`` inside the caption. Every extra token shifts all later
    annotations onto the wrong positions. This function instead reads the
    character offsets of the tokens that are actually in the sequence and
    assigns each token the IDs of every annotated span it overlaps.

    A token's leading whitespace is ignored for the overlap test, so a token
    that only touches a span through the space in front of it, or that is
    whitespace only, receives no annotation.

    Args:
        offsets: ``(start, end)`` character offsets of each token into ``text``.
        text: The clean caption the offsets index into.
        spans: Output of ``annotation_spans``.

    Returns:
        One (possibly empty) sorted list of annotation IDs per token.
    """
    per_token: list[list[int]] = []
    for start, end in offsets:
        # Tokens outside ``text`` (e.g. the chat template's trailing space
        # after the caption) are clamped away and receive no annotation.
        start, end = max(start, 0), min(end, len(text))
        while start < end and text[start].isspace():
            start += 1
        ids: set[int] = set()
        if start < end:
            for span_start, span_end, annotation_ids in spans:
                if start < span_end and span_start < end:
                    ids.update(annotation_ids)
        per_token.append(sorted(ids))
    return per_token


def tokenize_from_parsed(
    parsed_segments: list[tuple[list[int], str]],
    tokenizer: PreTrainedTokenizer,
    add_special_tokens: bool = False,
) -> tuple[list[int], list[list[int]]]:
    """Tokenize already-parsed segments while preserving annotation info.

    .. warning::
        The per-piece encoding does **not** match the tokens of the full
        caption in context (see ``align_annotations_to_offsets``). Do not use
        the returned annotation list to index into a sequence that was
        tokenized as a whole.

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
