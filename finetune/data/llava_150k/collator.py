from transformers import ProcessorMixin

from finetune.data.coconut.collator import make_collate_fn as make_coconut_collate_fn


def make_collate_fn(processor: ProcessorMixin, image_path: str):
    """Create a collate function for LLaVA-Instruct-150K training.

    Since LLaVA-Instruct-150K is built on top of the COCO dataset, we can
    reuse the annotation-aware tokenization logic from COCONut by creating a
    collate function that captures the processor state.

    Args:
        processor: Vision-language model processor.
        image_path: Path to the image directory.
    """

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
                path_to_image = f"{image_path}/{example['image']}"
                non_annotated_images.append(path_to_image)
                non_annotated_conversations.append(example["conversations"])

        if non_annotated_images and non_annotated_conversations:
            non_annotated_batch = processor(
                text=non_annotated_conversations,
                images=non_annotated_images,
                padding=True,
                truncation=False,
                max_length=None,
                return_tensors="pt",
            )
        else:
            non_annotated_batch = None

        return (
            make_coconut_collate_fn(processor)(annotated) if annotated else None,
            non_annotated_batch,
        )

    return collate_fn
