from jaxtyping import Bool, Float
from torch import Tensor

from finetune.criterion import Criterion


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion.

    This criterion computes the saliency alignment loss between the model's
    saliency maps and the provided annotation.

    - Uses Mean Squared Error (MSE) loss for alignment.
    """

    def compute_loss(
        self,
        attn: Float[Tensor, "S H W"],
        mask: Bool[Tensor, "S H W"],
    ) -> Float[Tensor, "1"]:
        pixel_counts = mask.sum(dim=(1, 2)).clamp(min=1)
        target = mask.float() / pixel_counts[:, None, None]

        loss = ((attn - target) ** 2).sum(dim=(1, 2))
        return loss.mean()
