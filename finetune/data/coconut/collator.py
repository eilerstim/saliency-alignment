import torch
from transformers import ProcessorMixin

from finetune.data.coconut.tokenization import (
    parse_annotated_caption,
    tokenize_with_annotations,
)


def train_collate_fn(examples: list[dict], processor: ProcessorMixin) -> dict:
    """Collate function for training with annotation-aware tokenization.

    This function processes a batch of examples from the COCONut dataset, creating
    clean captions (without annotation markers) for the model while separately
    tracking which tokens correspond to which segments for custom loss computation.

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
            - input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - labels (torch.Tensor): Label IDs for training (same as input_ids with -100 for padding)
            - masks (list of torch.Tensor): Per-token annotation masks of shape [seq_len, height, width]
            - **batch: Additional fields provided by the processor
    Note:
        The function creates clean captions by removing <id: text> markers before
        passing to the model, but separately tokenizes the annotated captions to
        maintain alignment between tokens and their corresponding segments.
    """
    images = []
    texts = []
    captions_annotated = []
    masks = []
    segments_infos = []

    # User-only messages template (for finding caption start)
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image in detail."},
            ],
        },
    ]
    prompt_only = processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )

    for example in examples:
        image = example["image"]
        caption = example["caption"]  # annotated: contains <id: text>
        mask = example["mask"]
        segments_info = example["segments_info"]

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
        masks.append(mask)
        segments_infos.append(segments_info)

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

    # Build annotation_ids aligned 1:1 with input_ids
    annotation_ids_lists: list[list[list[int]]] = [
        [[] for _ in range(seq_len)] for _ in range(batch_size)
    ]

    for i, caption in enumerate(captions_annotated):
        # Process prompt-only WITH this example's image to find where caption starts
        # (image placeholder expands to variable tokens depending on image size)
        prompt_only_batch = processor(
            text=[prompt_only],
            images=[images[i]],
            padding=False,
            return_tensors="pt",
        )
        caption_start = prompt_only_batch["input_ids"].shape[1]

        # Mask prompt tokens for this example
        labels[i, :caption_start] = -100

        # Tokenize annotated caption to get per-token annotation ids (list of lists) in caption space
        cap_token_ids, cap_ann_ids = tokenize_with_annotations(
            caption, processor.tokenizer, add_special_tokens=False
        )

        caption_len = min(len(cap_ann_ids), seq_len - caption_start)

        # Truncate/pad annotation ids to match caption_len
        if len(cap_ann_ids) >= caption_len:
            aligned_ann_ids = cap_ann_ids[:caption_len]
        else:
            pad_len = caption_len - len(cap_ann_ids)
            aligned_ann_ids = cap_ann_ids + [[]] * pad_len

        # Copy the annotation ID lists into the batch structure
        for j, ann_id_list in enumerate(aligned_ann_ids):
            if caption_start + j < seq_len:
                annotation_ids_lists[i][caption_start + j] = ann_id_list.copy()

    # Create per-token annotation masks [batch_size, seq_len, H, W]
    batch_size, seq_len = input_ids.shape

    # Create annotation masks for each token (list of (seq_len, H, W) tensors)
    annotation_masks = []
    for i, img_mask in enumerate(masks):
        height, width = img_mask.shape[-2:]
        annotation_masks.append(torch.zeros((seq_len, height, width), dtype=torch.bool))
        for j, ann_ids in enumerate(annotation_ids_lists[i]):
            token_mask = annotation_masks[i][j]
            for ann_id in ann_ids:
                token_mask |= img_mask == ann_id

    # Segment infos left out for now, can be added if needed
    return {
        **batch,
        "input_ids": input_ids,
        "labels": labels,
        "masks": annotation_masks,
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
        processor: Vision-language model processor (e.g., Qwen2VLProcessor)
            that handles both image and text processing.

    Returns:
        dict: A dictionary containing:
            - input_ids (torch.Tensor): Token IDs for prompts [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - answers (list): List of ground truth caption strings
            - masks (list of torch.Tensor): Per-token annotation masks of shape [seq_len, height, width]
            - segments_infos (torch.Tensor): Tensor of shape [batch_size, max_segments, 2]
              containing (segment_id, category_id) pairs, padded with -1
            - **batch: Additional fields provided by the processor
    Note:
        Unlike train_collate_fn, this uses add_generation_prompt=True to prepare
        the input for text generation during evaluation.
    """
    images = []
    texts = []
    answers = []
    masks = []
    segments_infos = []

    for example in examples:
        image = example["image"]
        ground_truth = example["caption"]
        mask = example["mask"]
        segments_info = example["segments_info"]

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
        answers.append(ground_truth)
        masks.append(mask)
        segments_infos.append(segments_info)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]

    # Labels: ignore padding tokens
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Convert segments_infos to tensor
    # Find max number of segments across the batch
    batch_size = len(segments_infos)
    max_segments = max(len(si) for si in segments_infos) if segments_infos else 1
    max_segments = max(max_segments, 1)  # Ensure at least 1

    # Create tensor and fill with -1 (padding)
    segments_infos_tensor = torch.full(
        (batch_size, max_segments, 2), -1, dtype=torch.long
    )
    for i, seg_info in enumerate(segments_infos):
        if seg_info:
            segments_infos_tensor[i, : len(seg_info)] = torch.tensor(
                seg_info, dtype=torch.long
            )
    segments_infos = segments_infos_tensor

    return {
        **batch,
        "input_ids": input_ids,
        "labels": labels,
        "answers": answers,
        "masks": masks,
        "segments_infos": segments_infos,
    }
