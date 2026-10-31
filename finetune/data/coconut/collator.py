import logging
from collections.abc import Callable

import torch
from transformers import ProcessorMixin

from finetune.data.coconut.tokenization import (
    annotation_char_spans,
    annotations_for_offsets,
    parse_annotated_caption,
)
from finetune.data.utils import crop_to_processor_view, find_sequence

logger = logging.getLogger(__name__)


def _caption_annotations(
    processor: ProcessorMixin,
    text: str,
    parsed_segments: list[tuple[list[int], str]],
    input_ids: torch.Tensor,
) -> tuple[int, list[list[int]]]:
    """Locate the caption in ``input_ids`` and annotate its tokens.

    Re-tokenizes the full chat-formatted ``text`` with character offsets so
    that tokens are split exactly as in the model input. The caption starts at
    the first token reaching into the caption's characters; this does not rely
    on an assistant header, which some templates (e.g. Mistral) leave empty.
    The caption and everything after it lie past the image placeholder that
    the processor expands, so that token sequence must appear verbatim in
    ``input_ids``.

    Returns:
        Index of the first caption token in ``input_ids`` and the annotation
        IDs for every token from there to the end of the prompt.
    """
    clean_caption = "".join(t for _, t in parsed_segments)
    stripped = clean_caption.strip()
    caption_char_start = text.rfind(stripped)
    if caption_char_start == -1:
        raise ValueError("Could not locate the caption in the chat-formatted prompt.")
    # The chat template may trim the caption; shift spans by the trimmed lead
    char_offset = caption_char_start - (
        len(clean_caption) - len(clean_caption.lstrip())
    )

    encoding = processor.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = [tuple(o) for o in encoding["offset_mapping"]]
    text_start = next(
        i for i, (_, end) in enumerate(offsets) if end > caption_char_start
    )

    caption_ids = encoding["input_ids"][text_start:]
    input_start = find_sequence(input_ids, caption_ids)
    if input_start == -1:
        raise ValueError(
            "Caption tokens differ between the processor and tokenizer; "
            "segment IDs cannot be aligned."
        )

    spans = [
        (start + char_offset, end + char_offset, ids)
        for start, end, ids in annotation_char_spans(parsed_segments)
    ]
    return input_start, annotations_for_offsets(offsets[text_start:], spans)


def make_collate_fn(processor: ProcessorMixin) -> Callable[[list[dict]], dict | None]:
    """Create a training collate function with precomputed processor state.

    Returns a closure that captures ``processor``.

    Args:
        processor: Vision-language model processor.

    Returns:
        Collate function compatible with ``torch.utils.data.DataLoader``.
    """

    def collate_fn(examples: list[dict]) -> dict | None:
        """Collate function for training with annotation-aware tokenization.

        Processes a batch of examples from the COCONut dataset, creating clean
        captions (without annotation markers) for the model while separately
        tracking which tokens correspond to which segment IDs for custom loss
        computation.

        Caption Alignment:
            The caption starts at the first token covering the caption's
            characters in the chat-formatted prompt, found via the tokenizer's
            offset mapping and located in ``input_ids`` by its token sequence.

        Label Masking:
            Labels are set to -100 (ignored in loss computation) for:
            - All prompt tokens (BOS, user message, image tokens, assistant header)
            - Padding tokens

            Only the caption tokens (assistant's response) have valid labels.

        Segment ID Alignment:
            Each prompt is re-tokenized as a whole with character offsets, and a
            caption token receives the segment IDs of every annotated span its
            characters overlap. Tokenizing spans separately would split words
            and spaces differently from the model's input and shift the
            segments onto neighbouring tokens. The re-tokenized caption tokens
            must equal the caption portion of ``input_ids``. Each token can have
            multiple segment IDs (e.g., when a word refers to multiple objects).

        Args:
            examples: List of dicts with keys image, caption, mask, segments_info.

        Returns:
            Batch dict or ``None`` if no valid examples remain.
        """
        images = []
        texts = []
        parsed_segments_list: list[list[tuple[list[int], str]]] = []
        panoptic_masks = []

        for example in examples:
            image = example["image"]
            caption = example["caption"]
            mask = example["mask"]

            # Parse once — reused for both clean caption and annotation alignment
            parsed_segments = parse_annotated_caption(caption)
            clean_caption = "".join(text for _, text in parsed_segments)

            if not clean_caption.strip():
                continue  # skip this example, not the whole batch

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the image in detail."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": clean_caption}],
                },
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            images.append(image)
            texts.append(prompt)
            parsed_segments_list.append(parsed_segments)
            # Same region as the image the vision tower sees (center crop)
            panoptic_masks.append(
                crop_to_processor_view(mask, getattr(processor, "image_processor", None))
            )

        if not images:
            return None

        # Standard processing with clean captions
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=False,
            max_length=None,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape

        # Labels: ignore padding tokens
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # First pass: assign annotations to the caption tokens of each prompt
        caption_starts = []
        tokenized_segments = []
        max_segments = 1

        for i, (text, parsed) in enumerate(
            zip(texts, parsed_segments_list, strict=True)
        ):
            caption_start, cap_ann_ids = _caption_annotations(
                processor, text, parsed, input_ids[i]
            )
            caption_starts.append(caption_start)
            tokenized_segments.append(cap_ann_ids)
            for ann_ids in cap_ann_ids:
                if len(ann_ids) > max_segments:
                    max_segments = len(ann_ids)

        # Create segment_ids tensor, padded with -1
        segment_ids_tensor = torch.full(
            (batch_size, seq_len, max_segments), -1, dtype=torch.long
        )

        # Collect indices and values for vectorized assignment
        batch_idx, token_idx, seg_idx, values = [], [], [], []

        for i, (caption_start, cap_ann_ids) in enumerate(
            zip(caption_starts, tokenized_segments, strict=True)
        ):
            # Mask prompt tokens
            labels[i, :caption_start] = -100

            for j, ann_ids in enumerate(cap_ann_ids):
                for k, ann_id in enumerate(ann_ids):
                    batch_idx.append(i)
                    token_idx.append(caption_start + j)
                    seg_idx.append(k)
                    values.append(ann_id)

        if values:
            segment_ids_tensor[batch_idx, token_idx, seg_idx] = torch.tensor(
                values, dtype=torch.long
            )

        return {
            **batch,
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids_tensor,
            "masks": panoptic_masks,
        }

    return collate_fn
