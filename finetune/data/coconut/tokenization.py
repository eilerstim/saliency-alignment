"""Annotation-aware tokenization for COCONut captions with segment references."""

import re

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


def annotation_char_spans(
    parsed_segments: list[tuple[list[int], str]],
) -> list[tuple[int, int, list[int]]]:
    """Character spans of the annotated text within the clean caption.

    The clean caption is ``"".join(text for _, text in parsed_segments)``.
    Surrounding whitespace is excluded from each span so that a token made
    only of the space before or after an annotation is not attributed to it.

    Args:
        parsed_segments: Output of ``parse_annotated_caption``.

    Returns:
        List of ``(start, end, annotation_ids)`` for each annotated span, with
        ``end`` exclusive.
    """
    spans: list[tuple[int, int, list[int]]] = []
    pos = 0
    for annotation_id_list, text in parsed_segments:
        if annotation_id_list:
            lead = len(text) - len(text.lstrip())
            stripped = text.strip()
            if stripped:
                start = pos + lead
                spans.append((start, start + len(stripped), annotation_id_list))
        pos += len(text)
    return spans


def annotations_for_offsets(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, list[int]]],
) -> list[list[int]]:
    """Assign annotation IDs to tokens from their character offsets.

    A token receives the IDs of every annotated span its characters overlap,
    so tokenization is taken from the full text as the model sees it rather
    than from each span in isolation.

    Args:
        offsets: ``(start, end)`` character offsets per token, in the same
            coordinates as ``spans``.
        spans: Output of ``annotation_char_spans`` (shifted into the offsets'
            coordinates).

    Returns:
        List of annotation-ID lists, one per token (empty if unannotated).
    """
    result: list[list[int]] = []
    for tok_start, tok_end in offsets:
        ids: list[int] = []
        for span_start, span_end, annotation_id_list in spans:
            if tok_start < span_end and tok_end > span_start:
                ids.extend(i for i in annotation_id_list if i not in ids)
        result.append(ids)
    return result


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
