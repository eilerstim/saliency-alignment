import torch

from vl_saliency._core.accum.base import SaliencyAccumulator
from vl_saliency.config import SaliencyConfig


class AnnotatedAccumulator(SaliencyAccumulator):
    """
    Saliency accumulator subclass that first filters out batch items with no annotations.

    Args:
        config (SaliencyConfig): The configuration object containing parameters for saliency extraction.
        input_ids (torch.Tensor): The input token IDs for the batch, used to identify image and generated tokens.
        masks (list[torch.Tensor | None]): A list of attention masks for each batch item, with None indicating no mask (i.e. skip).
        **kwargs: Additional keyword arguments from the forward pass, passed to the patch layout function.
    """

    def __init__(
        self,
        config: SaliencyConfig,
        input_ids: torch.Tensor,
        masks: list[torch.Tensor | None],
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        # Filter out batch items with no annotations (mask is None)
        annotated_indices = [i for i, mask in enumerate(masks) if mask is not None]

        self.annotated_indices = annotated_indices

        if self.annotated_indices:
            annotated_input_ids = input_ids[annotated_indices]
            annotated_pixel_values = pixel_values[annotated_indices]
            super().__init__(
                config,
                input_ids=annotated_input_ids,
                pixel_values=annotated_pixel_values,
                **kwargs,
            )

    def accumulate_qk(self, q: torch.Tensor, k: torch.Tensor):
        if not self.annotated_indices:
            return  # No annotated items, skip accumulation
        q_annotated = q[self.annotated_indices]
        k_annotated = k[self.annotated_indices]
        super().accumulate_qk(q_annotated, k_annotated)

    @property
    def saliency(self):
        if not self.annotated_indices:
            return None  # Return dummy saliency for empty case
        return super().saliency
