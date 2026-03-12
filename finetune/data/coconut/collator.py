import torch
from transformers import ProcessorMixin

from finetune.data.coconut.tokenization import (
    parse_annotated_caption,
    tokenize_with_annotations,
)
from finetune.data.utils import find_sequence


def collate_fn(examples: list[dict], processor: ProcessorMixin) -> dict:
    """Collate function for training with annotation-aware tokenization.

    This function processes a batch of examples from the COCONut dataset, creating
    clean captions (without annotation markers) for the model while separately
    tracking which tokens correspond to which segment IDs for custom loss computation.

    Caption Alignment:
        The function identifies where the caption starts in the tokenized sequence by:
        1. Extracting the assistant header (e.g., " ASSISTANT:") by comparing chat
           templates with and without `add_generation_prompt=True`
        2. Searching for this header in the processed `input_ids`
        3. The caption tokens begin immediately after the header

    Label Masking:
        Labels are set to -100 (ignored in loss computation) for:
        - All prompt tokens (BOS, user message, image tokens, assistant header)
        - Padding tokens

        Only the caption tokens (assistant's response) have valid labels.

    Segment ID Alignment:
        Segment IDs from COCONut annotations are aligned token-by-token with the
        caption portion of `input_ids`. Each token can have multiple segment IDs
        (e.g., when a word refers to multiple objects).

    Args:
        examples: List of dictionaries, each containing:
            - image: PIL Image object
            - caption: Annotated caption string with <id: text> markers
            - mask: Segmentation mask tensor
            - segments_info: List of (segment_id, category_id) tuples
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
    captions_annotated = []
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
    assistant_header = with_gen[len(without_gen) :]
    suffix_tokens = processor.tokenizer.encode(
        assistant_header, add_special_tokens=False
    )

    for example in examples:
        image = example["image"]
        caption = example["caption"]  # annotated: contains <id: text>
        mask = example["mask"]

        # Parse caption and build a clean version without markers
        parsed_segments = parse_annotated_caption(caption)
        clean_caption = "".join(text for _, text in parsed_segments)

        if not clean_caption.strip():
            return None  # Skip empty captions

        # Build chat messages with *clean* caption only
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
        captions_annotated.append(caption)
        panoptic_masks.append(mask)

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

    for caption in captions_annotated:
        _, cap_ann_ids = tokenize_with_annotations(
            caption, processor.tokenizer, add_special_tokens=False
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
        # Find caption start by searching for suffix tokens in input_ids
        caption_start = find_sequence(input_ids[i], suffix_tokens) + len(suffix_tokens)

        # Mask prompt tokens for this example
        labels[i, :caption_start] = -100

        caption_len = min(len(cap_ann_ids), seq_len - caption_start)

        for j, ann_ids in enumerate(cap_ann_ids[:caption_len]):
            for k, ann_id in enumerate(ann_ids):
                batch_idx.append(i)
                token_idx.append(caption_start + j)
                seg_idx.append(k)
                values.append(ann_id)

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
    """Collate function for evaluation and inference.

    This function processes a batch of examples for evaluation by feeding only
    the user prompt to the model (without the assistant response), allowing the
    model to generate captions. Returns ground truth data for comparison.

    Args:
        examples: List of dictionaries, each containing:
            - image: PIL Image object
            - caption: Ground truth caption string (with annotations)
            - mask: Segmentation mask tensor
            - segments_info: List of (segment_id, category_id) tuples
        processor: Vision-language model processor
            that handles both image and text processing.

    Returns:
        dict: A dictionary containing:
            - input_ids (torch.Tensor): Token IDs for prompts [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - answers (list[str]): List of ground truth caption strings
            - segments (list[list[tuple[list[int], str]]]): Parsed annotation segments
            - masks (list[torch.Tensor]): Panoptic masks [H, W] containing segment IDs
            - **batch: Additional fields provided by the processor
    """
    images = []
    texts = []
    answers = []
    panoptic_masks = []
    all_segments = []

    for example in examples:
        image = example["image"]
        caption = example["caption"]
        mask = example["mask"]

        # Parse annotated caption to get segments
        parsed_segments = parse_annotated_caption(caption)
        clean_caption = "".join(text for _, text in parsed_segments)

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
        answers.append(clean_caption)
        panoptic_masks.append(mask)
        all_segments.append(parsed_segments)

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
