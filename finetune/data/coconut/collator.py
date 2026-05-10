import logging
from collections.abc import Callable

import torch
from transformers import ProcessorMixin

from finetune.data.coconut.tokenization import (
    parse_annotated_caption,
    tokenize_from_parsed,
)
from finetune.data.utils import find_sequence

logger = logging.getLogger(__name__)


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

            Only the caption tokens (assistant's response) have valid labels.

        Segment ID Alignment:
            Segment IDs from COCONut annotations are aligned token-by-token with
            the caption portion of ``input_ids``. Each token can have multiple
            segment IDs (e.g., when a word refers to multiple objects).

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
        batch_size, seq_len = input_ids.shape

        # Labels: ignore padding tokens
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # First pass: tokenize annotations (reusing parsed segments) and find max_segments
        tokenized_segments = []
        max_segments = 1

        for parsed in parsed_segments_list:
            _, cap_ann_ids = tokenize_from_parsed(
                parsed, processor.tokenizer, add_special_tokens=False
            )
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

        for i, cap_ann_ids in enumerate(tokenized_segments):
            caption_start = find_sequence(input_ids[i], suffix_tokens) + len(
                suffix_tokens
            )

            # Mask prompt tokens
            labels[i, :caption_start] = -100

            caption_len = min(len(cap_ann_ids), seq_len - caption_start)

            for j, ann_ids in enumerate(cap_ann_ids[:caption_len]):
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
