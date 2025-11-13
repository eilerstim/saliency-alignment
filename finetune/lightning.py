import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import PreTrainedModel


class FineTuner(L.LightningModule):
    """Fine-tuning module for a pre-trained model."""

    def __init__(self, cfg: DictConfig, model: PreTrainedModel):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.auxiliary_loss = instantiate(self.cfg.loss)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch: dict, batch_idx: int):
        outputs = self.model(**batch, output_attentions=True, return_dict=True)
        loss = outputs.loss
        auxiliary_loss = self.auxiliary_loss(
            batch["labels"], outputs.logits, outputs.attentions
        )

        log_dict = {
            "train/ce_loss": loss,
            "train/auxiliary_loss": auxiliary_loss,
            "train/loss": loss + auxiliary_loss,
        }
        self.log_dict(log_dict, prog_bar=True)
        return loss + auxiliary_loss

    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self.model(**batch, output_attentions=True, return_dict=True)
        loss = outputs.loss
        auxiliary_loss = self.auxiliary_loss(
            batch["labels"], outputs.logits, outputs.attentions
        )
        accuracy = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean()

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
