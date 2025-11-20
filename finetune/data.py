import torch

from finetune.tokenization import (
    parse_annotated_caption,
    tokenize_with_annotations,
)


def train_collate_fn(examples, processor):
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
        processor: Vision-language model processor (e.g., Qwen2VLProcessor)
            that handles both image and text processing.

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - labels (torch.Tensor): Label IDs for training (same as input_ids with -100 for padding)
            - annotation_ids (list[list[list[int]]]): Nested list structure [batch_size][seq_len][num_regions]
              where each token has a list of all region IDs it refers to (empty list for non-annotated tokens)
            - masks (torch.Tensor): Stacked segmentation masks [batch_size, height, width]
            - segments_infos (list): List of segments_info for each example in batch

    Note:
        The function creates clean captions by removing <id: text> markers before
        passing to the model, but separately tokenizes the annotated captions to
        maintain alignment between tokens and their corresponding segments.
    """
    import torch

    images = []
    texts = []
    captions_annotated = []
    masks = []
    segments_infos = []

    for example in examples:
        image = example["image"]
        caption = example["caption"]  # annotated: contains <id: text>
        mask = example["mask"]
        segments_info = example["segments_info"]

        # Parse caption and build a clean version without markers
        parsed_segments = parse_annotated_caption(caption)
        clean_caption = "".join(text for _, text in parsed_segments)

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
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    # Labels: ignore padding tokens
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Build annotation_ids aligned 1:1 with input_ids
    # annotation_ids will be a list of lists of lists: [batch_size][seq_len][num_regions]
    batch_size, seq_len = input_ids.shape
    annotation_ids = [[[] for _ in range(seq_len)] for _ in range(batch_size)]

    for i, caption in enumerate(captions_annotated):
        # Tokenize annotated caption to get per-token annotation ids (list of lists) in caption space
        cap_token_ids, cap_ann_ids = tokenize_with_annotations(
            caption, processor.tokenizer, add_special_tokens=False
        )

        # Find where the assistant tokens (caption) start in labels
        valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue

        caption_start = valid_positions[0].item()
        caption_len = len(valid_positions)

        # Truncate/pad annotation ids to match caption_len
        if len(cap_ann_ids) >= caption_len:
            aligned_ann_ids = cap_ann_ids[:caption_len]
        else:
            pad_len = caption_len - len(cap_ann_ids)
            aligned_ann_ids = cap_ann_ids + [[]] * pad_len

        # Copy the annotation ID lists into the batch structure
        for j, ann_id_list in enumerate(aligned_ann_ids):
            if caption_start + j < seq_len:
                annotation_ids[i][caption_start + j] = ann_id_list.copy()

    # Stack masks into a batch tensor
    masks = torch.stack(masks)

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    batch["annotation_ids"] = annotation_ids
    batch["masks"] = masks
    batch["segments_infos"] = segments_infos

    return (
        input_ids,
        attention_mask,
        pixel_values,
        labels,
        annotation_ids,
        masks,
        segments_infos,
    )


def eval_collate_fn(examples, processor):
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
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Token IDs for prompts [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - answers (list): List of ground truth caption strings
            - masks (torch.Tensor): Stacked segmentation masks [batch_size, height, width]
            - segments_infos (list): List of segments_info for each example in batch

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
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    # Stack masks into a batch tensor
    masks = torch.stack(masks)

    return input_ids, attention_mask, pixel_values, answers, masks, segments_infos
