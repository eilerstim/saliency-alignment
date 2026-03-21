import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor

from finetune.criterion import Criterion


class SaliencyAlignment(Criterion):
    """Saliency Alignment Criterion using KL divergence."""

    def compute_loss(
        self, attn: Float[Tensor, "S H W"], mask: Bool[Tensor, "S H W"]
    ) -> Float[Tensor, "1"]:
        pixel_counts = mask.sum(dim=(1, 2)).clamp(min=1)
        target = mask.float() / pixel_counts[:, None, None]

        log_attn = torch.log(attn.clamp(min=1e-8))

        # KL(target || attn)
        kl = F.kl_div(log_attn, target, reduction="none").sum(dim=(1, 2))
        return kl.mean()
