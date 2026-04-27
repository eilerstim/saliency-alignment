import logging
from collections import defaultdict

import lightning as L
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from vl_saliency.maps import SaliencyGrid

from .criterion import Criterion
from .metrics import ALIGNMENT_METRICS, format_alignment_table, summarize_alignment

logger = logging.getLogger(__name__)


class FineTuner(L.LightningModule):
    """Fine-tuning module for a pre-trained model."""

    def __init__(
        self, cfg: DictConfig, model: PreTrainedModel, processor: ProcessorMixin
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.processor = processor

        # Instantiate auxiliary loss function
        self.auxiliary_loss: Criterion = instantiate(self.cfg.loss)

        # Per-image alignment scores accumulated within ``run_alignment_eval``.
        # Keys are metric names from ``ALIGNMENT_METRICS``; values are lists of
        # per-image floats (one entry per batch item that contains grounded tokens).
        self._val_alignment_scores: dict[str, list[float]] = defaultdict(list)

    def forward(self, **batch):
        return self.model(**batch, return_dict=True)

    def training_step(self, batch: dict, batch_idx: int):
        # Forward pass with saliency accumulation
        outputs = self(**batch)
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            segment_ids=batch["segment_ids"],
            preds=outputs.logits,
            saliency=outputs.saliency,
            masks=batch["masks"],
        )

        # Log relevant metrics
        log_dict = {
            "train/ce_loss": loss.detach(),
            "train/auxiliary_loss": auxiliary_loss.detach(),
            "train/loss": loss.detach() + auxiliary_loss.detach(),
        }
        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].size(0),
        )

        return loss + auxiliary_loss

    def validation_step(self, batch: dict, batch_idx: int):
        # Forward pass with saliency accumulation
        outputs = self(**batch)
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            segment_ids=batch["segment_ids"],
            preds=outputs.logits,
            saliency=outputs.saliency,
            masks=batch["masks"],
        )

        preds = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Shift for next-token prediction
        shift_preds = preds[:, :-1]
        shift_labels = labels[:, 1:]

        # Ignore padding (-100)
        valid_token_mask = shift_labels != -100

        if valid_token_mask.any():
            accuracy = (
                (shift_preds[valid_token_mask] == shift_labels[valid_token_mask])
                .float()
                .mean()
            )
        else:
            accuracy = torch.tensor(0.0, device=labels.device)

        # Log relevant metrics
        log_dict = {
            "val/ce_loss": loss,
            "val/auxiliary_loss": auxiliary_loss,
            "val/loss": loss + auxiliary_loss,
            "val/accuracy": accuracy,
        }
        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].size(0),
        )

    @torch.no_grad()
    def run_alignment_eval(self) -> None:
        """Run a full-validation pass dedicated to attention alignment metrics.

        We deliberately bypass ``trainer.validate(...)`` here. Re-entering an
        FSDP-wrapped model with ``use_orig_params=True`` after ``trainer.fit``
        has returned can leave inner units in their sharded state, which
        surfaces as a ``[0]``-shape ``pre_layrnorm.weight`` inside the CLIP
        vision tower the moment the saliency-patched forward fires.

        Driving the loop ourselves inside the active ``Saliency`` context
        keeps the FSDP forward-hook chain that already worked during fit.
        """
        if self.trainer.is_global_zero:
            logger.info("Starting full-validation alignment metrics pass.")

        self._val_alignment_scores = defaultdict(list)
        self.eval()

        # ``trainer.val_dataloaders`` is populated and DistSampler-wrapped by
        # Lightning's val phase during fit; fall back to wrapping fresh if
        # something cleared it.
        val_loader = self.trainer.val_dataloaders or self.trainer.strategy.process_dataloader(
            self.val_dataloader()
        )

        for batch in val_loader:
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            outputs = self(**batch)
            per_image = self._per_image_alignment_metrics(
                saliency=outputs.saliency,
                masks=batch["masks"],
                segment_ids=batch["segment_ids"],
                labels=batch["labels"],
            )
            for name, scores in per_image.items():
                self._val_alignment_scores[name].extend(scores)

        merged = self._gather_alignment_scores()
        summary = summarize_alignment(merged)
        if not summary or not self.trainer.is_global_zero:
            return

        logger.info(
            "Validation attention alignment metrics:\n%s",
            format_alignment_table(summary),
        )

    def _gather_alignment_scores(self) -> dict[str, list[float]]:
        """All-gather variable-length per-rank score lists into one dict."""
        local = {k: list(v) for k, v in self._val_alignment_scores.items()}

        if self.trainer.world_size <= 1 or not (
            dist.is_available() and dist.is_initialized()
        ):
            return local

        gathered: list[dict[str, list[float]] | None] = [None] * self.trainer.world_size
        dist.all_gather_object(gathered, local)

        merged: dict[str, list[float]] = defaultdict(list)
        for d in gathered:
            if d is None:
                continue
            for name, scores in d.items():
                merged[name].extend(scores)
        return merged

    @torch.no_grad()
    def _per_image_alignment_metrics(
        self,
        saliency: SaliencyGrid,
        masks: list[Tensor],
        segment_ids: Tensor,
        labels: Tensor,
    ) -> dict[str, list[float]]:
        """Compute per-image alignment scores by averaging per-token metrics.

        For each batch item, gathers attention maps of tokens with at least
        one ground-truth segment, upsamples them to annotation resolution,
        and runs each metric in a single batched call over the ``(T, H, W)``
        token stack. Per-image score = NaN-ignoring mean across tokens.
        """
        per_image: dict[str, list[float]] = {name: [] for name in ALIGNMENT_METRICS}

        for b in range(saliency.batch_size):
            mask_seg = masks[b]
            attn = saliency.maps_for_image(batch_idx=b, image_idx=0)

            gen_ids = labels[b] != -100
            seg_ids = segment_ids[b][gen_ids]

            gen_len = seg_ids.shape[0]
            attn = attn[-gen_len:]

            has_segments = (seg_ids != -1).any(dim=1)
            if not has_segments.any():
                continue

            seg_ids = seg_ids[has_segments]
            attn = attn[has_segments]

            # Upsample raw attention to annotation resolution. No softmax:
            # AMR depends on the unnormalised mass distribution; AP and NSS
            # are invariant under monotonic / affine rescaling.
            attn_up = (
                F.interpolate(
                    attn.unsqueeze(1).float(),
                    size=mask_seg.shape,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)
                .detach()
            )

            # Build per-token boolean masks ``(T, H, W)`` in one shot.
            valid = seg_ids != -1
            token_mask = (
                (mask_seg[None, None] == seg_ids[:, :, None, None])
                & valid[:, :, None, None]
            ).any(dim=1)

            # One batched call per metric instead of T × calls; AP's argsort
            # is the dominant cost and benefits most from this.
            for name, fn in ALIGNMENT_METRICS.items():
                token_vals = fn(attn_up, token_mask)
                mean_val = torch.nanmean(token_vals)
                if not torch.isnan(mean_val):
                    per_image[name].append(float(mean_val))

        return per_image

    def configure_optimizers(self) -> tuple:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = instantiate(self.cfg.optim, params=params)

        if "scheduler" not in self.cfg:
            return optimizer

        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        # Instantiate dataset with split="train"
        dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="train")

        # Get collator function and bind processor via partial
        collate_fn = instantiate(self.cfg.data.collator, processor=self.processor)

        dl_kwargs = getattr(self.cfg, "dataloader", {})
        return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Instantiate dataset with split="validation"
        dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="validation")

        # Get eval collator function and bind processor via partial
        collate_fn = instantiate(self.cfg.data.eval_collator, processor=self.processor)

        dl_kwargs = OmegaConf.to_container(
            getattr(self.cfg, "dataloader", {}), resolve=True
        )
        dl_kwargs["shuffle"] = False  # No shuffling for validation

        return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)
