"""Collate functions for PNG-COCO dataset with segment ID tracking."""

import torch
from transformers import ProcessorMixin

from finetune.data.png.tokenization import build_clean_caption, tokenize_png_caption
from finetune.data.utils import find_sequence


def collate_fn(examples: list[dict], processor: ProcessorMixin) -> dict:
    """Collate function for training with PNG-COCO segment ID tracking.

    This function processes a batch of examples from the PNG-COCO dataset, creating
    clean captions for the model while separately tracking which tokens correspond
    to which segment IDs for custom loss computation.

    Caption Alignment:
        The function identifies where the caption starts in the tokenized sequence by:
        1. Extracting the assistant header (e.g., " ASSISTANT:") by comparing chat
           templates with and without `add_generation_prompt=True`
        2. Searching for this header in the processed `input_ids`
        3. The caption tokens begin immediately after the header

        This approach is robust to variable image token counts (e.g., dynamic resolution
        models) since it searches for the header in the final tokenized sequence.

    Label Masking:
        Labels are set to -100 (ignored in loss computation) for:
        - All prompt tokens (BOS, user message, image tokens, assistant header)
        - Padding tokens
        
        Only the caption tokens (assistant's response) have valid labels, ensuring
        the model is trained only to generate the caption, not predict the prompt.

    Segment ID Alignment:
        Segment IDs from PNG-COCO annotations are aligned token-by-token with the
        caption portion of `input_ids`. Each token can have multiple segment IDs
        (e.g., when a word refers to multiple objects).

    Args:
        examples: List of dictionaries, each containing:
            - image: PIL Image object
            - caption: Full caption string
            - segments: List of segment dicts with utterance and segment_ids
            - panoptic_mask: Panoptic segmentation mask tensor
        processor: Vision-language model processor
            that handles both image and text processing.

    Returns:
        dict:
            - input_ids (torch.Tensor): Token IDs [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - labels (torch.Tensor): Training targets [batch_size, seq_len]
                -100 for prompt/padding tokens, token IDs for caption tokens
            - segment_ids (torch.Tensor): Per-token segment IDs 
                [batch_size, seq_len, max_segments], padded with -1
            - masks (list[torch.Tensor]): Panoptic masks [H, W] containing segment IDs
            - **batch: Additional fields provided by the processor
    """
    images = []
    texts = []
    all_segments = []
    panoptic_masks = []

    # Get the assistant header tokens by comparing with/without generation prompt
    user_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "X"}],
        },
    ]
    without_gen = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=False
    )
    with_gen = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    assistant_header = with_gen[len(without_gen):]
    suffix_tokens = processor.tokenizer.encode(assistant_header, add_special_tokens=False)

    for example in examples:
        image = example["image"]
        segments = example["segments"]
        panoptic_mask = example["panoptic_mask"]

        # Build clean caption from segments
        clean_caption = build_clean_caption(segments)

        if not clean_caption.strip():
            return None  # Skip empty captions

        # Build chat messages with clean caption
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
                "content": [
                    {"type": "text", "text": clean_caption},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        images.append(image)
        texts.append(prompt)
        all_segments.append(segments)
        panoptic_masks.append(panoptic_mask)

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

    # First pass: tokenize all captions and find max_segments
    tokenized_segments = []
    max_segments = 1  # At least 1 to avoid empty dimension

    for segments in all_segments:
        _, cap_segment_ids = tokenize_png_caption(
            segments, processor.tokenizer, add_special_tokens=False
        )
        tokenized_segments.append(cap_segment_ids)
        for seg_ids in cap_segment_ids:
            if len(seg_ids) > max_segments:
                max_segments = len(seg_ids)

    # Create segment_ids tensor, padded with -1
    segment_ids_tensor = torch.full(
        (batch_size, seq_len, max_segments), -1, dtype=torch.long
    )

    # Collect indices and values for vectorized assignment
    batch_idx, token_idx, seg_idx, values = [], [], [], []

    for i, cap_segment_ids in enumerate(tokenized_segments):
        # Find caption start by searching for suffix tokens in input_ids
        # (more efficient than processing prompt with each image)
        caption_start = find_sequence(input_ids[i], suffix_tokens) + len(suffix_tokens)

        # Mask prompt tokens for this example
        labels[i, :caption_start] = -100

        caption_len = min(len(cap_segment_ids), seq_len - caption_start)

        for j, seg_ids in enumerate(cap_segment_ids[:caption_len]):
            for k, seg_id in enumerate(seg_ids):
                batch_idx.append(i)
                token_idx.append(caption_start + j)
                seg_idx.append(k)
                values.append(seg_id)

    # Single vectorized assignment
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


def eval_collate_fn(examples: list[dict], processor: ProcessorMixin) -> dict:
    """Collate function for evaluation and inference with PNG-COCO.

    This function processes a batch of examples for evaluation by feeding only
    the user prompt to the model (without the assistant response), allowing the
    model to generate captions. Returns ground truth data for comparison.

    Args:
        examples: List of dictionaries, each containing:
            - image: PIL Image object
            - caption: Ground truth caption string
            - segments: List of segment dicts with utterance and segment_ids
            - panoptic_mask: Panoptic segmentation mask tensor
        processor: Vision-language model processor
            that handles both image and text processing.

    Returns:
        dict: A dictionary containing:
            - input_ids (torch.Tensor): Token IDs for prompts [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - answers (list[str]): List of ground truth caption strings
            - segments (list[list[dict]]): Ground truth segments with segment_ids
            - panoptic_masks (list[torch.Tensor]): Panoptic masks [batch_size, H, W]
            - **batch: Additional fields provided by the processor
    """
    images = []
    texts = []
    answers = []
    all_segments = []
    panoptic_masks = []

    for example in examples:
        image = example["image"]
        caption = example["caption"]
        segments = example["segments"]
        panoptic_mask = example["panoptic_mask"]

        images.append(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image in detail."},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(prompt)
        answers.append(caption)
        all_segments.append(segments)
        panoptic_masks.append(panoptic_mask)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]

    # Labels: ignore padding tokens
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        **batch,
        "input_ids": input_ids,
        "labels": labels,
        "answers": answers,
        "segments": all_segments,
        "masks": panoptic_masks,
    }
