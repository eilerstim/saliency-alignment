import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from finetune.criterion import Criterion


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

    def forward(self, **batch):
        return self.model(**batch, return_dict=True)

    def training_step(self, batch: tuple[dict, list, list], batch_idx: int):
        # Forward pass with saliency accumulation
        batch, masks, segment_ids = batch
        outputs = self(**batch)
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            segment_ids=segment_ids,
            preds=outputs.logits,
            saliency=outputs.saliency,
            masks=masks,
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

    def validation_step(self, batch: tuple[dict, list, list], batch_idx: int):
        # Forward pass with saliency accumulation
        batch, masks, segment_ids = batch
        outputs = self(**batch)
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            segment_ids=segment_ids,
            preds=outputs.logits,
            saliency=outputs.saliency,
            masks=masks,
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

    def configure_optimizers(self) -> tuple:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = instantiate(self.cfg.optim, params=params)

        if "scheduler" not in self.cfg:
            return optimizer

        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
        
    def train_dataloader(self) -> DataLoader:
        # Get collator function and bind processor via partial
        train_dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="train")
        collate_fn = instantiate(self.cfg.data.collator, processor=self.processor)

        dl_kwargs = getattr(self.cfg, "dataloader", {})
        return DataLoader(train_dataset, collate_fn=collate_fn, **dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Get eval collator function and bind processor via partial
        val_dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="validation")
        collate_fn = instantiate(self.cfg.data.eval_collator, processor=self.processor)

        dl_kwargs = OmegaConf.to_container(
            getattr(self.cfg, "dataloader", {}), resolve=True
        )
        dl_kwargs["shuffle"] = False  # No shuffling for validation

        return DataLoader(val_dataset, collate_fn=collate_fn, **dl_kwargs)
