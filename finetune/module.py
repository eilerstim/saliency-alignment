from typing import Literal

import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from finetune.accumulator import AnnotatedAccumulator
from finetune.criterion import Criterion
from finetune.model import build_model
from vl_saliency.config import SaliencyConfig


class FineTuner(pl.LightningModule):
    """Fine-tuning module for a pre-trained model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        self.model, self.processor = build_model(cfg.model, cfg.lora)
        self.saliency_config = SaliencyConfig.from_model(self.model)
        self.auxiliary_loss: Criterion = instantiate(self.cfg.loss)

        self.save_hyperparameters()

    def forward(
        self, inputs: tuple[dict, list, list], stage: Literal["train", "val"]
    ) -> torch.Tensor:
        batch, masks, segment_ids = inputs
        trace = AnnotatedAccumulator(
            config=self.saliency_config,
            masks=masks,
            **batch,
        )

        outputs = self.model(**batch, saliency=trace, return_dict=True)
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            segment_ids=segment_ids,
            preds=outputs.logits,
            saliency=outputs.saliency,
            masks=masks,
        )

        log_dict = {
            f"{stage}/ce_loss": outputs.loss.detach(),
            f"{stage}/auxiliary_loss": auxiliary_loss.detach(),
            f"{stage}/loss": outputs.loss.detach() + auxiliary_loss.detach(),
        }

        if stage == "val":
            metrics = validation_metrics(outputs.logits, batch["labels"])
            log_dict.update({f"val/{k}": v for k, v in metrics.items()})

        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].size(0),
        )

        return outputs.loss + auxiliary_loss

    def training_step(self, batch: tuple[dict, list, list], batch_idx: int):
        loss = self.forward(batch, stage="train")
        return loss

    def validation_step(self, batch: tuple[dict, list, list], batch_idx: int):
        self.forward(batch, stage="val")

    def configure_optimizers(
        self,
    ) -> Optimizer | tuple[list[Optimizer], list[LRScheduler]]:
        optimizer: Optimizer = instantiate(
            self.cfg.optim, params=self.model.parameters()
        )

        if "scheduler" not in self.cfg:
            return optimizer

        scheduler: LRScheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        # Get collator function and bind processor via partial
        train_dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="train")
        collate_fn = instantiate(self.cfg.data.collator, processor=self.processor)

        dl_kwargs = getattr(self.cfg, "dataloader", {})
        return DataLoader(train_dataset, collate_fn=collate_fn, **dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Get eval collator function and bind processor via partial
        val_dataset = instantiate(
            self.cfg.data.dataset, self.cfg.data, split="validation"
        )
        collate_fn = instantiate(self.cfg.data.eval_collator, processor=self.processor)

        dl_kwargs = OmegaConf.to_container(
            getattr(self.cfg, "dataloader", {}), resolve=True
        )
        dl_kwargs["shuffle"] = False  # No shuffling for validation

        return DataLoader(val_dataset, collate_fn=collate_fn, **dl_kwargs)


def validation_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return {"accuracy": _compute_accuracy(logits, labels)}


def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    shift_preds = preds[:, :-1]
    shift_labels = labels[:, 1:]
    valid_token_mask = shift_labels != -100

    if valid_token_mask.any():
        return (
            (shift_preds[valid_token_mask] == shift_labels[valid_token_mask])
            .float()
            .mean()
        )
    else:
        return torch.tensor(0.0, device=labels.device)
