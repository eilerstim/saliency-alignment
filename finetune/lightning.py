from functools import partial

import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from finetune.cli.save_data import COCONutPanCapDataset
from finetune.data import eval_collate_fn, train_collate_fn


class FineTuner(L.LightningModule):
    """Fine-tuning module for a pre-trained model."""

    def __init__(self, cfg: DictConfig, model: PreTrainedModel, processor):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.auxiliary_loss = instantiate(self.cfg.loss)

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

        outputs = self.model(**model_batch, output_attentions=True, return_dict=True)
        loss = outputs.loss

        # Auxiliary loss can leverage annotation_ids, masks, and segments_infos
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            preds=outputs.logits,
            attentions=outputs.attentions,
            annotation_ids=batch.get("annotation_ids"),
            masks=batch.get("masks"),
            segments_infos=batch.get("segments_infos"),
        )

        log_dict = {
            "train/ce_loss": loss,
            "train/auxiliary_loss": auxiliary_loss,
            "train/loss": loss + auxiliary_loss,
        }
        self.log_dict(log_dict, prog_bar=True)
        return loss + auxiliary_loss

    def validation_step(self, batch: dict, batch_idx: int):
        model_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "pixel_values": batch["pixel_values"],
            "labels": batch["labels"],
        }

        outputs = self.model(**model_batch, output_attentions=True, return_dict=True)
        loss = outputs.loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            preds=outputs.logits,
            attentions=outputs.attentions,
            annotation_ids=batch.get("annotation_ids"),
            masks=batch.get("masks"),
            segments_infos=batch.get("segments_infos"),
        )
        # Optionally ignore padding (-100) when computing accuracy
        preds = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        valid_token_mask = labels != -100
        if valid_token_mask.any():
            accuracy = (preds[valid_token_mask] == labels[valid_token_mask]).float().mean()
        else:
            accuracy = (preds == labels).float().mean()

        log_dict = {
            "val/ce_loss": loss,
            "val/auxiliary_loss": auxiliary_loss,
            "val/loss": loss + auxiliary_loss,
            "val/accuracy": accuracy,
        }
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, params=self.model.parameters())

        if "scheduler" not in self.cfg:
            return optimizer

        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = COCONutPanCapDataset(self.cfg.data, split="train")
        collate = partial(train_collate_fn, processor=self.processor)

        dl_kwargs = getattr(self.cfg.data, "dataloader_kwargs", None)
        if dl_kwargs is None:
            dl_kwargs = {
                "batch_size": self.cfg.batch_size,
                "num_workers": self.cfg.num_workers,
                "shuffle": True,
            }

        return DataLoader(dataset, collate_fn=collate, **dl_kwargs)

    def val_dataloader(self):
        dataset = COCONutPanCapDataset(self.cfg.data, split="validation")
        collate = partial(eval_collate_fn, processor=self.processor)

        dl_kwargs = getattr(self.cfg.data, "dataloader_kwargs", None)
        if dl_kwargs is None:
            dl_kwargs = {
                "batch_size": self.cfg.batch_size,
                "num_workers": self.cfg.num_workers,
                "shuffle": False,
            }

        # Ensure we don't shuffle in validation even if config says otherwise
        dl_kwargs = {**dl_kwargs, "shuffle": False}

        return DataLoader(dataset, collate_fn=collate, **dl_kwargs)