import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from .transformer_utils import _get_image_token_id, _get_vision_patch_shape


class FineTuner(L.LightningModule):
    """Fine-tuning module for a pre-trained model."""

    def __init__(
        self, cfg: DictConfig, model: PreTrainedModel, processor: ProcessorMixin
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.processor = processor

        self.image_token_id = _get_image_token_id(self.model.config)
        self.patch_shape = _get_vision_patch_shape(self.model.config)
        self.auxiliary_loss = instantiate(
            self.cfg.loss, self.image_token_id, self.patch_shape
        )

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch: dict, batch_idx: int):
        # Only pass model-relevant inputs to the backbone

        model_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "labels": batch["labels"],
        }

        # Forward pass
        outputs = self.model(**model_batch, output_attentions=True, return_dict=True)
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            input_ids=batch["input_ids"],
            preds=outputs.logits,
            attentions=outputs.attentions,
            masks=batch.get("masks"),
        )

        # Log relevant metrics
        log_dict = {
            "train/ce_loss": loss,
            "train/auxiliary_loss": auxiliary_loss,
            "train/loss": loss + auxiliary_loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, sync_dist=True, batch_size=len(batch["input_ids"])
        )

        return loss + auxiliary_loss

    def validation_step(self, batch: dict, batch_idx: int):
        # Only pass model-relevant inputs to the backbone
        model_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "labels": batch["labels"],
        }

        # Forward pass
        with torch.inference_mode():
            outputs = self.model(
                **model_batch, output_attentions=True, return_dict=True
            )
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            input_ids=batch["input_ids"],
            preds=outputs.logits,
            attentions=outputs.attentions,
            masks=batch.get("masks"),
        )

        # Optionally ignore padding (-100) when computing accuracy
        preds = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        valid_token_mask = labels != -100
        if valid_token_mask.any():
            accuracy = (
                (preds[valid_token_mask] == labels[valid_token_mask]).float().mean()
            )
        else:
            accuracy = (preds == labels).float().mean()

        # Log relevant metrics
        log_dict = {
            "val/ce_loss": loss,
            "val/auxiliary_loss": auxiliary_loss,
            "val/loss": loss + auxiliary_loss,
            "val/accuracy": accuracy,
        }
        self.log_dict(
            log_dict, prog_bar=True, sync_dist=True, batch_size=len(batch["input_ids"])
        )

    def configure_optimizers(self) -> tuple:
        optimizer = instantiate(self.cfg.optim, params=self.model.parameters())

        if "scheduler" not in self.cfg:
            return optimizer

        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        # Instantiate dataset with split="train"
        dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="train")

        # Get collator function and bind processor via partial
        collate_fn = instantiate(
            self.cfg.data.collator, processor=self.processor, _partial_=True
        )

        dl_kwargs = getattr(self.cfg, "dataloader", {})
        dl_kwargs = OmegaConf.to_container(dl_kwargs)
        return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Instantiate dataset with split="validation"
        dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="validation")

        # Get eval collator function and bind processor via partial
        collate_fn = instantiate(
            self.cfg.data.eval_collator, processor=self.processor, _partial_=True
        )

        dl_kwargs = getattr(self.cfg, "dataloader", {})
        dl_kwargs = OmegaConf.to_container(dl_kwargs)
        dl_kwargs["shuffle"] = False  # No shuffling for validation

        return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)
