from dataclasses import dataclass
from typing import Literal

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from .criterion import Criterion


@dataclass
class DummySaliency:
    saliency: Literal[None] = None

    def accumulate_qk(self, q, k):
        pass


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

    def forward(self, batch: tuple[dict, dict]):
        annotated, non_annotated = batch
        if annotated is not None:
            annotated_out = self.model(**annotated, return_dict=True)
        else:
            annotated_out = None

        if non_annotated is not None:
            dummy_saliency = DummySaliency()
            non_annotated_out = self.model(
                saliency=dummy_saliency, **non_annotated, return_dict=True
            )
        else:
            non_annotated_out = None

        return annotated_out, non_annotated_out

    def training_step(self, batch: tuple[dict, dict], batch_idx: int):
        # Forward pass with saliency accumulation
        annotated_out, non_annotated_out = self(batch)
        loss = torch.tensor(0.0, device=self.device)
        batch_size = 0
        if annotated_out is not None:
            if annotated_out.loss is not None:
                loss += annotated_out.loss

            # Calculate auxiliary loss
            annotated = batch[0]
            auxiliary_loss = self.auxiliary_loss(
                labels=annotated["labels"],
                segment_ids=annotated["segment_ids"],
                preds=annotated_out.logits,
                saliency=annotated_out.saliency,
                masks=annotated["masks"],
            )
            batch_size += annotated["input_ids"].size(0)
        else:
            auxiliary_loss = torch.tensor(0.0, device=self.device)

        if non_annotated_out is not None:
            if non_annotated_out.loss is not None:
                loss += non_annotated_out.loss
            batch_size += batch[1]["input_ids"].size(0)

        # Log relevant metrics
        log_dict = {
            "train/ce_loss": loss.detach(),
            "train/auxiliary_loss": auxiliary_loss.detach(),
            "train/loss": loss.detach() + auxiliary_loss.detach(),
            "train/annotated_loss": annotated_out.loss.detach()
            if annotated_out is not None and annotated_out.loss is not None
            else torch.tensor(0.0, device=self.device),
            "train/non_annotated_loss": non_annotated_out.loss.detach()
            if non_annotated_out is not None and non_annotated_out.loss is not None
            else torch.tensor(0.0, device=self.device),
        }
        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss + auxiliary_loss

    def validation_step(self, batch: tuple[dict, dict], batch_idx: int):
        # Forward pass with saliency accumulation
        annotated_out, non_annotated_out = self(batch)
        loss = torch.tensor(0.0, device=self.device)
        batch_size = 0
        if annotated_out is not None:
            if annotated_out.loss is not None:
                loss += annotated_out.loss
            batch_size += batch[0]["input_ids"].size(0)
        if non_annotated_out is not None:
            if non_annotated_out.loss is not None:
                loss += non_annotated_out.loss
            batch_size += batch[1]["input_ids"].size(0)

        if annotated_out is not None:
            # Calculate auxiliary loss
            annotated = batch[0]
            auxiliary_loss = self.auxiliary_loss(
                labels=annotated["labels"],
                segment_ids=annotated["segment_ids"],
                preds=annotated_out.logits,
                saliency=annotated_out.saliency,
                masks=annotated["masks"],
            )
        else:
            auxiliary_loss = torch.tensor(0.0, device=self.device)

        # preds = annotated_out.logits.argmax(dim=-1)
        # labels = annotated["labels"]

        # # Shift for next-token prediction
        # shift_preds = preds[:, :-1]
        # shift_labels = labels[:, 1:]

        # # Ignore padding (-100)
        # valid_token_mask = shift_labels != -100

        # if valid_token_mask.any():
        #     accuracy = (
        #         (shift_preds[valid_token_mask] == shift_labels[valid_token_mask])
        #         .float()
        #         .mean()
        #     )
        # else:
        #     accuracy = torch.tensor(0.0, device=labels.device)

        # Log relevant metrics
        log_dict = {
            "val/ce_loss": loss,
            "val/auxiliary_loss": auxiliary_loss,
            "val/loss": loss + auxiliary_loss,
            "val/annotated_loss": annotated_out.loss
            if annotated_out is not None and annotated_out.loss is not None
            else torch.tensor(0.0, device=self.device),
            "val/non_annotated_loss": non_annotated_out.loss
            if non_annotated_out is not None and non_annotated_out.loss is not None
            else torch.tensor(0.0, device=self.device),
            # "val/accuracy": accuracy,
        }

        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

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
