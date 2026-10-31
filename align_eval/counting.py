"""Does localization predict counting accuracy? (paper Section 5.3, Appendix D)

Counting questions from TallyQA on COCO images, each labelled with the COCO
category it counts (see the cat_bench repo for the labelling pipeline). For
every question the model's answer is constrained to the integers 1-9, and the
saliency of the count is scored against the union of that category's instance
masks with the same AMR / AP / NSS used everywhere else (``align_eval.metrics``).

One forward pass per question gives both: the prompt ends just before the count
digit, so the last position's logits over the nine digit tokens are the
constrained answer, and the last position's saliency row is the query that
predicts the count. That row cannot see the answer, so it is the same for the
ground-truth and the predicted count. Masks are cropped to the region the
vision grid covers (LLaVA-1.5 center-crops).

Usage (one GPU):

    python -m align_eval.counting --model llava-hf/llava-1.5-7b-hf \\
        --data ../cat_bench/data/test.json --out results/counting/base
"""

import argparse
import json
import logging
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from scipy import stats
from transformers import AutoModelForImageTextToText, AutoProcessor

from finetune.data.utils import crop_to_processor_view
from vl_saliency import Saliency

from .metrics import METRIC_NAMES, amr, average_precision, nss

logger = logging.getLogger(__name__)

COUNTS = [str(i) for i in range(1, 10)]
GROUPS = {"1-3": COUNTS[:3], "4-9": COUNTS[3:]}


def load_questions(path: Path, head: int) -> pd.DataFrame:
    """First ``head`` labelled questions on COCO train2014 images with answer 1-9
    (the selection cat_bench's ``preprocess_data`` makes)."""
    df = pd.DataFrame(json.loads(path.read_text()))
    df["answer"] = df["answer"].astype(str)
    df = df.dropna(subset=["label", "answer"])
    df = df[(df["image"].str.split("/").str[0] == "train2014") & df["answer"].isin(COUNTS)]
    return df.head(head).reset_index(drop=True)


def digit_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids of the digits 1-9 and the prefix the tokenizer puts before a
    digit at the start of an answer (LLaMA: the "▁" space token)."""
    encoded = [tokenizer.encode(d, add_special_tokens=False) for d in COUNTS]
    prefix = encoded[0][:-1]
    for ids in encoded:
        if ids[:-1] != prefix:
            raise ValueError(f"Digits do not share one prefix: {encoded}")
    return [ids[-1] for ids in encoded], prefix


def category_mask(coco: COCO, image_id: int, category: str) -> np.ndarray:
    """Union of all instance masks of ``category`` in the image."""
    info = coco.loadImgs(image_id)[0]
    mask = np.zeros((info["height"], info["width"]), dtype=bool)
    cat_ids = coco.getCatIds(catNms=[category])
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=cat_ids)):
        mask |= coco.annToMask(ann).astype(bool)
    return mask


def pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x[ok].astype(float), y[ok].astype(float))
    return float(r), float(p)


def summarise(df: pd.DataFrame) -> dict:
    """Accuracy, error and metric means, plus each metric's Pearson r (and p)
    with correctness."""
    out = {
        "n": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "mean_abs_error": float((df["pred"].astype(int) - df["answer"].astype(int)).abs().mean()),
        "mean_p_correct": float(df["p_correct"].mean()),
    }
    for m in METRIC_NAMES:
        out[f"{m}_mean"] = float(df[m].mean())
        out[f"{m}_median"] = float(df[m].median())
        out[f"{m}_r"], out[f"{m}_p"] = pearson(df[m], df["correct"])
    return out


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    coco = COCO(str(args.instances))
    category_names = [c["name"] for c in coco.loadCats(coco.getCatIds())]
    questions = load_questions(Path(args.data), args.head)
    logger.info("Loaded %d questions from %s", len(questions), args.data)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.tokenizer or args.model)
    digit_ids, prefix = digit_token_ids(processor.tokenizer)

    rows = []
    with Saliency(model, backend="torch_eager"):
        for i, q in questions.iterrows():
            row = {k: q[k] for k in ("question_id", "image_id", "question", "answer", "label")}
            # Closest COCO category, as cat_bench matches it
            match = get_close_matches(q["label"], category_names, n=1, cutoff=0.6)
            row["category"] = match[0] if match else None

            image = Image.open(Path(args.images) / f"{int(q['image_id']):012d}.jpg").convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": q["question"]}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            if prefix:  # end the prompt right before the digit
                extra = torch.tensor([prefix], device=device)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], extra], dim=1)
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], torch.ones_like(extra)], dim=1)

            out = model(**inputs, return_dict=True)
            probs = torch.softmax(out.logits[0, -1, digit_ids].float(), dim=0)
            pred = int(probs.argmax()) + 1
            row["pred"] = str(pred)
            row["correct"] = row["pred"] == row["answer"]
            row["p_correct"] = float(probs[int(q["answer"]) - 1])

            # Saliency of the query that predicts the count (last position)
            sal = out.saliency.maps_for_image(batch_idx=0, image_idx=0)[-1]
            mask = None
            if row["category"] is not None:
                full = torch.from_numpy(category_mask(coco, int(q["image_id"]), row["category"]))
                mask = crop_to_processor_view(full, processor.image_processor).to(device)
            if mask is None or not mask.any():
                row.update({m: float("nan") for m in METRIC_NAMES})
                row["valid"] = False
            else:
                sal = F.interpolate(sal[None, None].float(), size=tuple(mask.shape),
                                    mode="bilinear", align_corners=False)[0]
                sal = torch.softmax(sal.flatten(1), dim=1).view(1, *mask.shape)
                for name, fn in zip(METRIC_NAMES, (amr, average_precision, nss), strict=True):
                    row[name] = float(fn(sal, mask[None])[0])
                row["valid"] = True
            rows.append(row)
            if (i + 1) % 200 == 0:
                logger.info("Evaluated %d / %d", i + 1, len(questions))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_question.csv", index=False)
    valid = df[df["valid"]]

    summary = {
        "model": args.model,
        "data": str(args.data),
        "n_questions": int(len(df)),
        "n_invalid": int((~df["valid"]).sum()),
        "all_questions": {k: v for k, v in summarise(df).items() if "AMR" not in k
                          and "AP" not in k and "NSS" not in k},
        "overall": summarise(valid),
        "by_count": {c: summarise(valid[valid["answer"] == c])
                     for c in COUNTS if (valid["answer"] == c).any()},
        "by_group": {g: summarise(valid[valid["answer"].isin(cs)]) for g, cs in GROUPS.items()},
        "confusion": pd.crosstab(valid["answer"], valid["pred"]).to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    o = summary["overall"]
    logger.info("n=%d (invalid %d) accuracy=%.3f", o["n"], summary["n_invalid"], o["accuracy"])
    for m in METRIC_NAMES:
        logger.info("%s mean=%.3f r=%.3f (p=%.2g)", m, o[f"{m}_mean"], o[f"{m}_r"], o[f"{m}_p"])
    logger.info("Wrote %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or checkpoint directory")
    ap.add_argument("--tokenizer", default=None, help="Processor to load (default: --model)")
    ap.add_argument("--data", type=Path, required=True, help="cat_bench questions JSON")
    ap.add_argument("--instances", type=Path,
                    default=Path("data/coco/annotations/instances_train2017.json"))
    ap.add_argument("--images", type=Path, default=Path("data/coco/images/train2017"))
    ap.add_argument("--head", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
