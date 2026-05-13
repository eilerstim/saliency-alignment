from transformers import ProcessorMixin

from finetune.data.coconut.collator import _compute_suffix_tokens
from finetune.data.coconut.collator import make_collate_fn as make_coconut_collate_fn
from finetune.data.utils import find_sequence


def llava_to_chat_messages(conversations: list[dict]) -> list[dict]:
    role_map = {
        "human": "user",
        "gpt": "assistant",
    }

    messages = []

    for turn in conversations:
        role = role_map[turn["from"]]
        value = turn["value"]

        if role == "user" and "<image>" in value:
            text = value.replace("<image>", "").strip()

            content = [{"type": "image"}]
            if text:
                content.append({"type": "text", "text": text})
        else:
            content = value

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return messages


def make_collate_fn(processor: ProcessorMixin, image_path: str):
    """Create a collate function for LLaVA-Instruct-150K training.

    Since LLaVA-Instruct-150K is built on top of the COCO dataset, we can
    reuse the annotation-aware tokenization logic from COCONut by creating a
    collate function that captures the processor state.

    Args:
        processor: Vision-language model processor.
        image_path: Path to the image directory.
    """
    coconut_collate_fn = make_coconut_collate_fn(processor)
    suffix_tokens = _compute_suffix_tokens(processor)

    def collate_fn(examples: list[dict]) -> tuple[dict | None, dict | None]:
        """Collate function for LLaVA-Instruct-150K training.

        This function simply delegates to the COCONut collate function, which
        handles the annotation-aware tokenization and segment ID tracking. The
        LLaVA-specific logic is encapsulated in the dataset and processor, so
        we can reuse the same collate function for both datasets.

        Args:
            examples: List of examples from the dataset.

        Returns:
            - Batch with annotations processed for training (input_ids, attention_mask, segment_ids, etc.)
            - Batch without annotations (input_ids, etc)
        """

        annotated = []
        non_annotated_images = []
        non_annotated_conversations = []
        for example in examples:
            if "mask" in example:
                annotated.append(example)
            else:
                path_to_image = f"{image_path}/COCO_train2014_{example['image']}"
                non_annotated_images.append(path_to_image)
                conversation = processor.apply_chat_template(
                    llava_to_chat_messages(example["conversations"]),
                    add_generation_prompt=False,
                    tokenize=False,
                )
                non_annotated_conversations.append(conversation)

        if non_annotated_images and non_annotated_conversations:
            non_annotated_batch = processor(
                text=non_annotated_conversations,
                images=non_annotated_images,
                padding=True,
                truncation=False,
                max_length=None,
                return_tensors="pt",
            )

            input_ids = non_annotated_batch["input_ids"]

            labels = input_ids.clone()
            batch_size, seq_len = input_ids.shape
            labels[labels == processor.tokenizer.pad_token_id] = -100

            for i in range(batch_size):
                caption_start = find_sequence(input_ids[i], suffix_tokens) + len(
                    suffix_tokens
                )

                # Mask prompt tokens
                labels[i, :caption_start] = -100

            non_annotated_batch["labels"] = labels
        else:
            non_annotated_batch = None

        return (
            coconut_collate_fn(annotated) if annotated else None,
            non_annotated_batch,
        )

    return collate_fn
