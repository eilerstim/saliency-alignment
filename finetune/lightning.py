import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, ProcessorMixin

from .saliency import get_maps, trace_model
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

        text_config = self.model.config.text_config
        self.head_dim = text_config.hidden_size // text_config.num_attention_heads

        # Instantiate auxiliary loss function
        self.auxiliary_loss = instantiate(self.cfg.loss)

        # Trace model for saliency computation
        trace_model(self.model, self.patch_shape)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch: dict, batch_idx: int):
        # Forward pass with saliency accumulation
        outputs = self.model(**batch, return_dict=True)
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            input_ids=batch["input_ids"],
            segment_ids=batch["segment_ids"],
            preds=outputs.logits,
            attentions=get_maps(self.model),
            masks=batch["masks"],
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
        # Forward pass with saliency accumulation
        outputs = self.model(**batch, return_dict=True)
        loss = outputs.loss

        # Calculate auxiliary loss
        auxiliary_loss = self.auxiliary_loss(
            labels=batch["labels"],
            input_ids=batch["input_ids"],
            segment_ids=batch["segment_ids"],
            preds=outputs.logits,
            attentions=get_maps(self.model),
            masks=batch["masks"],
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
        collate_fn = self._collate_fn(self.cfg.data.collator)

        dl_kwargs = getattr(self.cfg, "dataloader", {})
        dl_kwargs = OmegaConf.to_container(dl_kwargs)
        return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)

    def val_dataloader(self) -> DataLoader:
        # Instantiate dataset with split="validation"
        dataset = instantiate(self.cfg.data.dataset, self.cfg.data, split="validation")

        # Get eval collator function and bind processor via partial
        collate_fn = self._collate_fn(self.cfg.data.eval_collator)

        dl_kwargs = getattr(self.cfg, "dataloader", {})
        dl_kwargs = OmegaConf.to_container(dl_kwargs)
        dl_kwargs["shuffle"] = False  # No shuffling for validation

        return DataLoader(dataset, collate_fn=collate_fn, **dl_kwargs)

    def _collate_fn(self, collator_cfg: DictConfig) -> dict:
        """Wrapper around the collate function to bind the processor."""
        collate_fn = instantiate(collator_cfg, processor=self.processor, _partial_=True)
        return collate_fn

        # def wrapper(batch: dict) -> dict:
        #     batch = collate_fn(batch)

        #     input_ids = batch["input_ids"]
        #     B, S = input_ids.size()

        #     batch_idx, seq_idx = (input_ids == self.image_token_id).nonzero(
        #         as_tuple=True
        #     )

        #     img_starts = torch.full((B,), S, dtype=torch.long, device=input_ids.device)
        #     img_ends = torch.zeros((B,), dtype=torch.long, device=input_ids.device)

        #     img_starts.scatter_reduce_(0, batch_idx, seq_idx, reduce="amin")
        #     img_ends.scatter_reduce_(0, batch_idx, seq_idx + 1, reduce="amax")

        #     if (img_starts == S).any():
        #         raise RuntimeError("Sample without image tokens")

        #     segment_ids = batch["segment_ids"]
        #     B, S, _ = segment_ids.shape

        #     # positions where all segments are valid
        #     batch_idx, seq_idx = (segment_ids != -1).any(dim=-1).nonzero(as_tuple=True)

        #     gen_starts = torch.full((B,), S, dtype=torch.long, device=input_ids.device)
        #     gen_starts.scatter_reduce_(0, batch_idx, seq_idx, reduce="amin")

        #     if (gen_starts == S).any():
        #         raise RuntimeError("Sample without generation tokens")

        #     if (img_starts >= gen_starts).any():
        #         raise RuntimeError("Malformed image spans in input_ids.")

        #     batch["img_starts"] = img_starts
        #     batch["img_ends"] = img_ends
        #     batch["gen_starts"] = gen_starts

        #     return batch

        # return wrapper
