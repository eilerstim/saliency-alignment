"""Helper functions for dealing with HF Transformers models. Source: vl_saliency."""


def _get_image_token_id(config) -> int:
    """
    Get the image token id from a multimodal config. If not found, return -1.
    """
    return getattr(config, "image_token_id", getattr(config, "image_token_index", -1))


def _get_vision_patch_shape(config) -> tuple[int, int] | None:
    """
    Get the number of height and width tokens from a multimodal config.
    """
    # If explicit count is given, prefer that
    if "mm_tokens_per_image" in config:
        side = int(config.mm_tokens_per_image**0.5)
        return side, side  # Assume Square Tokens

    # Otherwise, check vision_config
    if "vision_config" in config:
        vision_cfg = config.vision_config
        if "image_size" in vision_cfg and "patch_size" in vision_cfg:
            image_size = vision_cfg.image_size
            patch_size = vision_cfg.patch_size
            side = image_size // patch_size
            return side, side  # Assume Square Tokens

    else:
        raise ValueError("Cannot determine vision patch shape from model config.")
