import logging
from collections.abc import Callable

import torch
from transformers import ProcessorMixin

from finetune.data.coconut.tokenization import (
    align_annotations_to_offsets,
    annotation_spans,
    parse_annotated_caption,
)
from finetune.data.utils import find_sequence, visible_crop_box

logger = logging.getLogger(__name__)

PROMPT = "Describe the image in detail."


def _compute_suffix_tokens(processor: ProcessorMixin) -> list[int]:
    """Compute the assistant-header token IDs once.

    Derives the token sequence that separates the user prompt from the
    assistant response by diffing the chat template with and without
    ``add_generation_prompt=True``.
    """
    user_messages = [
        {"role": "user", "content": [{"type": "text", "text": "X"}]},
    ]
    without_gen = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=False
    )
    with_gen = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    assistant_header = with_gen[len(without_gen) :]
    return processor.tokenizer.encode(assistant_header, add_special_tokens=False)


def make_collate_fn(processor: ProcessorMixin) -> Callable[[list[dict]], dict | None]:
    """Create a training collate function with precomputed processor state.

    Returns a closure that captures ``processor`` and the assistant-header
    token IDs so they are computed once rather than every batch.

    Args:
        processor: Vision-language model processor.

    Returns:
        Collate function compatible with ``torch.utils.data.DataLoader``.
    """
    suffix_tokens = _compute_suffix_tokens(processor)
    tokenizer = processor.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError(
            "The COCONut collator needs a fast tokenizer: annotation IDs are "
            "assigned to tokens through their character offsets."
        )
    if tokenizer.padding_side != "left":
        # The criterion and the alignment metrics pair the LAST gen_len
        # saliency rows with the labelled positions, which is only right when
        # the caption is the tail of the sequence.
        raise ValueError(
            f"padding_side must be 'left' (got {tokenizer.padding_side!r}); "
            "the loss/metrics assume the caption ends the sequence."
        )

    def collate_fn(examples: list[dict]) -> dict | None:
        """Collate function for training with annotation-aware tokenization.

        Processes a batch of examples from the COCONut dataset, creating clean
        captions (without annotation markers) for the model while separately
        tracking which tokens correspond to which segment IDs for custom loss
        computation.

        Caption Alignment:
            The function identifies where the caption starts in the tokenized
            sequence by searching for the precomputed assistant-header tokens.
            The caption tokens begin immediately after the header.

        Label Masking:
            Labels are set to -100 (ignored in loss computation) for:
            - All prompt tokens (BOS, user message, image tokens, assistant header)
            - Padding tokens

            Only the caption tokens (assistant's response) and the closing
            EOS token have valid labels.

        Segment ID Alignment:
            The prompt is tokenized a second time with character offsets, and
            every caption token receives the segment IDs of the annotated
            span(s) its characters fall into. The caption tokens of this
            tokenization are checked to be identical to the ones the processor
            produced, so segment IDs always refer to the tokens the model sees.
            Each token can have multiple segment IDs (e.g., when a word refers
            to multiple objects).

        Mask Geometry:
            The image processor center-crops non-square images, so the
            panoptic mask is cropped to the same region
            (:func:`finetune.data.utils.visible_crop_box`) before it is
            returned. The loss and the alignment metrics upsample the patch
            grid to the mask's shape, which is only valid when both cover
            the same part of the image.

        Args:
            examples: List of dicts with keys image, caption, mask, segments_info.

        Returns:
            Batch dict or ``None`` if no valid examples remain.
        """
        images = []
        texts = []
        clean_captions: list[str] = []
        spans_list: list[list[tuple[int, int, list[int]]]] = []
        panoptic_masks = []

        for example in examples:
            image = example["image"]
            caption = example["caption"]
            mask = example["mask"]

            # Parse once — reused for both clean caption and annotation alignment
            parsed_segments = parse_annotated_caption(caption)
            clean_caption, spans = annotation_spans(parsed_segments)

            if not clean_caption.strip():
                continue  # skip this example, not the whole batch

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT},
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
            # Close the assistant turn the way LLaVA-1.5 was trained: the HF
            # chat template ends the turn with a bare space and no EOS, which
            # would train the model to emit a space and never to stop. Drop
            # that space and make </s> the last supervised token.
            if prompt.endswith(clean_caption + " "):
                prompt = prompt[:-1]
            prompt += tokenizer.eos_token

            if tuple(mask.shape) != (image.height, image.width):
                logger.warning(
                    "Skipping example: mask shape %s does not match image size "
                    "(H, W) = %s.",
                    tuple(mask.shape),
                    (image.height, image.width),
                )
                continue
            left, top, right, bottom = visible_crop_box(
                image.size, processor.image_processor
            )
            mask = mask[top:bottom, left:right].contiguous()

            images.append(image)
            texts.append(prompt)
            clean_captions.append(clean_caption)
            spans_list.append(spans)
            panoptic_masks.append(mask)

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
        attention_mask = batch["attention_mask"]
        batch_size, seq_len = input_ids.shape

        # Labels: ignore padding tokens
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # First pass: per-token segment IDs from character offsets, and max_segments
        tokenized_segments: list[list[list[int]]] = []
        caption_starts: list[int] = []
        max_segments = 1

        for i, (prompt, clean_caption, spans) in enumerate(
            zip(texts, clean_captions, spans_list, strict=True)
        ):
            caption_start = find_sequence(input_ids[i], suffix_tokens)
            if caption_start < 0:
                raise ValueError("Assistant header not found in tokenized prompt.")
            caption_start += len(suffix_tokens)
            caption_len = int(attention_mask[i, caption_start:].sum())
            caption_ids = input_ids[i, caption_start : caption_start + caption_len]

            # Same text, tokenized once more to get character offsets. The
            # processor only rewrites the ``<image>`` placeholder, which lies
            # before the caption, so the caption tokens must coincide.
            encoding = tokenizer(
                prompt, add_special_tokens=True, return_offsets_mapping=True
            )
            ref_ids = encoding["input_ids"]
            ref_start = find_sequence(torch.tensor(ref_ids), suffix_tokens)
            if ref_start < 0:
                raise ValueError("Assistant header not found in offset tokenization.")
            ref_start += len(suffix_tokens)
            if ref_ids[ref_start:] != caption_ids.tolist():
                raise ValueError(
                    "Caption tokens differ between the processor and the offset "
                    "tokenization; cannot align annotations."
                )

            caption_char0 = prompt.rfind(clean_caption)
            if caption_char0 < 0:
                raise ValueError(
                    "The chat template did not insert the caption verbatim; "
                    "cannot map character offsets back to the caption."
                )
            offsets = [
                (start - caption_char0, end - caption_char0)
                for start, end in encoding["offset_mapping"][ref_start:]
            ]
            # Offsets must be real character spans covering the caption;
            # guards against tokenizer versions that return empty offsets.
            if any(end <= start for start, end in offsets) or (
                offsets and offsets[-1][1] < len(clean_caption.rstrip())
            ):
                raise ValueError(
                    "Tokenizer returned empty or incomplete offsets for the "
                    "caption; cannot assign annotations."
                )
            cap_ann_ids = align_annotations_to_offsets(offsets, clean_caption, spans)

            tokenized_segments.append(cap_ann_ids)
            caption_starts.append(caption_start)
            for ann_ids in cap_ann_ids:
                if len(ann_ids) > max_segments:
                    max_segments = len(ann_ids)

        # Create segment_ids tensor, padded with -1
        segment_ids_tensor = torch.full(
            (batch_size, seq_len, max_segments), -1, dtype=torch.long
        )

        # Collect indices and values for vectorized assignment
        batch_idx, token_idx, seg_idx, values = [], [], [], []

        for i, (cap_ann_ids, caption_start) in enumerate(
            zip(tokenized_segments, caption_starts, strict=True)
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
