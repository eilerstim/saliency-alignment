import torch
import torch.nn.functional as F
from pycocotools import mask as maskUtils
from transformers import ProcessorMixin


def train_collate_fn(examples: list[dict], processor: ProcessorMixin) -> dict:
    """Collate function for training with phrase-based token-to-region mapping.

    This function processes a batch of examples from the GranD dataset, tokenizing
    captions and mapping tokens to object regions based on phrase matching.

    Args:
        examples: List of dictionaries, each containing:
            - image: PIL Image object
            - caption: Dense caption string (plain text)
            - rle_masks: Dictionary mapping object_id to RLE segmentation dict
            - phrase_positions: List of (start_pos, end_pos, object_ids) tuples
        processor: Vision-language model processor
            that handles both image and text processing.

    Returns:
        dict: A dictionary containing:
            - input_ids (torch.Tensor): Token IDs of shape [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - labels (torch.Tensor): Label IDs for training (same as input_ids with -100 for padding)
            - masks (torch.Tensor): Per-token annotation masks of shape [batch_size, seq_len, height, width]
            - **batch: Additional fields provided by the processor
    Note:
        RLE masks are decoded on-the-fly only for objects referenced by tokens.
        Token-to-object mapping is determined using character positions from
        GranD's tokens_positive annotations.
    """
    images = []
    texts = []
    captions_list = []
    rle_masks_list = []
    phrase_positions_list = []

    for example in examples:
        image = example["image"]
        caption = example["caption"]
        rle_masks = example["rle_masks"]
        phrase_positions = example["phrase_positions"]

        # Build chat messages with caption
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
                    {"type": "text", "text": caption},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        images.append(image)
        texts.append(prompt)
        captions_list.append(caption)
        rle_masks_list.append(rle_masks)
        phrase_positions_list.append(phrase_positions)

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

    # Labels: ignore padding tokens
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Build token-to-object mappings by matching phrases
    batch_size, seq_len = input_ids.shape
    annotation_ids_lists: list[list[list[int]]] = [
        [[] for _ in range(seq_len)] for _ in range(batch_size)
    ]

    for i, caption in enumerate(captions_list):
        phrase_positions = phrase_positions_list[i]

        # Find where the assistant tokens (caption) start in labels
        valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue

        caption_start = valid_positions[0].item()

        # Tokenize just the caption to get token offsets
        caption_encoding = processor.tokenizer(
            caption, add_special_tokens=False, return_offsets_mapping=True
        )
        token_offsets = caption_encoding["offset_mapping"]

        # For each phrase position from GranD, find which tokens it covers
        for phrase_start, phrase_end, obj_ids in phrase_positions:
            # Find tokens that overlap with this phrase using character positions
            for tok_idx, (tok_start, tok_end) in enumerate(token_offsets):
                if tok_idx >= len(valid_positions):
                    break
                # Check if token overlaps with phrase
                if tok_start < phrase_end and tok_end > phrase_start:
                    global_tok_idx = caption_start + tok_idx
                    if global_tok_idx < seq_len:
                        annotation_ids_lists[i][global_tok_idx].extend(obj_ids)

    # Create per-token annotation masks [batch_size, seq_len, H, W]
    # Decode RLE masks on-the-fly only for objects referenced by tokens
    batch_size, seq_len = input_ids.shape

    # Use standard resolution based on GranD dataset average
    target_height, target_width = 1500, 2250

    annotation_masks = torch.zeros((batch_size, seq_len, target_height, target_width))

    for i in range(batch_size):
        rle_masks = rle_masks_list[i]

        # Get original dimensions from first RLE mask
        if rle_masks:
            first_rle = next(iter(rle_masks.values()))
            orig_height, orig_width = first_rle["size"]
        else:
            continue

        for j, ann_ids in enumerate(annotation_ids_lists[i]):
            if ann_ids:
                # Create binary mask for this token by decoding and combining RLE masks
                token_mask = torch.zeros((orig_height, orig_width), dtype=torch.bool)
                for ann_id in ann_ids:
                    if ann_id in rle_masks:
                        # Decode RLE mask for this object
                        binary_mask = maskUtils.decode(rle_masks[ann_id])
                        token_mask |= torch.from_numpy(binary_mask > 0)

                # Resize mask to target resolution if needed
                if (orig_height, orig_width) != (target_height, target_width):
                    # Convert to float tensor and add batch/channel dims for interpolate
                    token_mask_float = token_mask.float().unsqueeze(0).unsqueeze(0)
                    # Resize using bilinear interpolation
                    resized = F.interpolate(
                        token_mask_float,
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    # Threshold back to binary and remove extra dims
                    annotation_masks[i, j] = (
                        resized.squeeze(0).squeeze(0) > 0.5
                    ).float()
                else:
                    annotation_masks[i, j] = token_mask.float()

    return {
        **batch,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
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
            - caption: Ground truth caption string (plain text)
            - rle_masks: Dictionary mapping object_id to RLE segmentation dict
            - phrase_positions: List of (start_pos, end_pos, object_ids) tuples
        processor: Vision-language model processor (e.g., Qwen2VLProcessor)
            that handles both image and text processing.

    Returns:
        dict: A dictionary containing:
            - input_ids (torch.Tensor): Token IDs for prompts [batch_size, seq_len]
            - attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            - pixel_values (torch.Tensor): Processed image tensor
            - answers (list): List of ground truth caption strings
            - rle_masks_list (list): List of RLE mask dictionaries for each example
            - phrase_positions_list (list): List of phrase position mappings for each example
            - **batch: Additional fields provided by the processor
    Note:
        Unlike train_collate_fn, this uses add_generation_prompt=True to prepare
        the input for text generation during evaluation.
        RLE masks and phrase positions are returned for potential use in evaluation metrics.
    """
    images = []
    texts = []
    answers = []
    rle_masks_list = []
    phrase_positions_list = []

    for example in examples:
        image = example["image"]
        ground_truth = example["caption"]
        rle_masks = example["rle_masks"]
        phrase_positions = example["phrase_positions"]

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
        rle_masks_list.append(rle_masks)
        phrase_positions_list.append(phrase_positions)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    return {
        **batch,
        "answers": answers,
        "rle_masks_list": rle_masks_list,
        "phrase_positions_list": phrase_positions_list,
    }
