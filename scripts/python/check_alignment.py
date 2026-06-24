"""One-batch diagnostic: do supervised tokens' saliency land on their masks?

Reproduces the exact token<->saliency pairing that finetune/criterion/base.py
and align_eval/metrics.py rely on (attn[-gen_len:] paired with the labels != -100
tokens) on a single real batch, and prints, per supervised token: the decoded
text, its mask size, and the chance-normalised AMR (reusing align_eval.metrics).

Interpretation:
  * Aligned   -> object/noun tokens show AMR > 1 and the decoded text names what
                 the mask covers (e.g. "dog" with AMR 3.x).
  * Misaligned -> AMR ~= 1 (chance) across tokens, or sensible-looking tokens
                 paired with the wrong regions => the attn[-gen_len:] slice /
                 padding side is off and the loss is training the wrong tokens.

Run on 1 GPU in the saliency env (loads the base model by default; pass
+model_path=... to check a trained checkpoint):
    srun --environment=saliency $PROJECT_DIR/.venv/bin/python \
        scripts/python/check_alignment.py
"""

import logging

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from align_eval.metrics import amr
from vl_saliency import Saliency

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = cfg.get("model_path") or cfg.model.name

    # Eager attention is required for vl_saliency's hook-based extractor.
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=cfg.model.dtype, attn_implementation="eager"
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"tokenizer padding_side = {processor.tokenizer.padding_side}")

    dataset = instantiate(cfg.data.dataset, cfg.data, split="validation")
    collate_fn = instantiate(cfg.data.eval_collator, processor=processor)

    bs = int(cfg.dataloader.batch_size)
    batch = collate_fn([dataset[i] for i in range(bs)])
    assert batch is not None, "collator dropped the whole batch; try other indices"
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with Saliency(model, backend="torch_eager"), torch.no_grad():
        out = model(**batch, return_dict=True)
    saliency = out.saliency

    for b in range(saliency.batch_size):
        mask = batch["masks"][b].to(device)
        attn = saliency.maps_for_image(batch_idx=b, image_idx=0)

        gen = batch["labels"][b] != -100
        seg_ids = batch["segment_ids"][b][gen]
        tok_ids = batch["input_ids"][b][gen]
        gen_len = seg_ids.shape[0]
        print(f"\n=== image {b}: {attn.shape[0]} attn rows, {gen_len} supervised tokens ===")
        attn = attn[-gen_len:]  # the exact slice criterion/base.py uses

        has = (seg_ids != -1).any(dim=1)
        if not has.any():
            print("  (no annotated tokens in this caption)")
            continue

        # Match the criterion/metric: upsample -> spatial softmax -> per-token mask.
        a = F.interpolate(
            attn.unsqueeze(1).float(), size=tuple(mask.shape),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        a = torch.softmax(a.flatten(1), dim=1).view(-1, *mask.shape)
        valid = seg_ids != -1
        tmask = (
            (mask[None, None] == seg_ids[:, :, None, None]) & valid[:, :, None, None]
        ).any(dim=1)

        per_tok = amr(a, tmask)  # (gen_len,), NaN where mask empty
        for t in range(gen_len):
            if not has[t]:
                continue
            text = processor.tokenizer.decode(tok_ids[t : t + 1])
            flag = "" if (per_tok[t] > 1) else "  <-- at/below chance"
            print(
                f"  tok={text!r:<18} mask_px={int(tmask[t].sum()):>7} "
                f"AMR={per_tok[t].item():6.2f}{flag}"
            )


if __name__ == "__main__":
    main()
