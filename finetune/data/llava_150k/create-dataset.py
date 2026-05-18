#!/usr/bin/env python3
"""Replace LLaVA detail_23k answers with matching Coconut PanCap captions."""

# TODO: Unvibecode this to work with the config

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

LLAVA_REPO_ID = "liuhaotian/LLaVA-Instruct-150K"
LLAVA_DETAIL_FILE = "detail_23k.json"
LLAVA_EXTRA_FILES = ("complex_reasoning_77k.json", "conversation_58k.json")
COCONUT_DATASET_ID = "xdeng77/coconut_pancap"

ASSISTANT_ROLES = {"assistant", "gpt"}
ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}
IMAGE_TOKEN = "<image>"


def parse_args() -> argparse.Namespace:
    parser_kwargs = {
        "description": (
            "Build a LLaVA detail_23k variant where assistant responses are "
            "replaced by matching Coconut PanCap captions."
        ),
    }
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--hf-output-dir",
        type=Path,
        default=Path("data/llava_coconut_pancap_150k"),
        help="Optional directory for Dataset.save_to_disk output.",
    )
    parser.add_argument(
        "--coconut-dataset-id",
        default=COCONUT_DATASET_ID,
        help="Coconut PanCap dataset id.",
    )
    return parser.parse_args()


def normalize_image_id(value: Any) -> str | None:
    """Return a 12-digit COCO image id from a filename, key, or id string."""
    if value is None:
        return None

    text = str(value)
    basename = Path(text).name
    stem = Path(basename).stem
    matches = re.findall(r"\d+", stem) or re.findall(r"\d+", text)

    if not matches:
        return None

    return matches[-1].zfill(12)


def load_hub_json(filename: str) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=LLAVA_REPO_ID,
        repo_type="dataset",
        filename=filename,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_coconut_index(
    dataset_id: str,
    split: str,
) -> tuple[dict[str, str], dict[str, int]]:

    dataset = load_dataset(
        dataset_id,
        split=split,
    )

    captions_by_image_id: dict[str, str] = {}
    duplicate_keys = 0
    skipped_rows = 0

    for row in dataset:
        image_id = normalize_image_id(row.get("__key__"))
        caption = row.get("txt")

        if not image_id or not isinstance(caption, str) or not caption.strip():
            skipped_rows += 1
            continue

        if image_id in captions_by_image_id:
            duplicate_keys += 1
            continue

        captions_by_image_id[image_id] = caption.strip()

    stats = {
        "coconut_duplicate_keys": duplicate_keys,
        "coconut_skipped_rows": skipped_rows,
    }
    return captions_by_image_id, stats


def get_llava_image_id(row: dict[str, Any]) -> str | None:
    return normalize_image_id(row.get("image")) or normalize_image_id(row.get("id"))


def replace_first_assistant_response(
    row: dict[str, Any],
    new_response: str,
) -> dict[str, Any] | None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return None

    for turn in row["conversations"]:
        if not isinstance(turn, dict):
            continue

        role = str(turn.get("from", "")).strip().lower()
        if role in ASSISTANT_ROLES and "value" in turn:
            turn["value"] = new_response
            return row

    return None


def replace_detail_rows(
    detail_rows: Iterable[dict[str, Any]],
    captions_by_image_id: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output_rows: list[dict[str, Any]] = []
    stats = {
        "input_detail_rows": 0,
        "matched_and_replaced": 0,
        "skipped_no_match": 0,
        "skipped_no_assistant": 0,
        "extra_rows_added": 0,
        "output_rows": 0,
    }

    for row in detail_rows:
        stats["input_detail_rows"] += 1
        image_id = get_llava_image_id(row)
        replacement = captions_by_image_id.get(image_id) if image_id else None

        if replacement is None:
            stats["skipped_no_match"] += 1
            continue

        replaced_row = replace_first_assistant_response(row, replacement)
        if replaced_row is None:
            stats["skipped_no_assistant"] += 1
            continue

        output_rows.append(replaced_row)
        stats["matched_and_replaced"] += 1

    return output_rows, stats


def chat_content(role: str, value: str) -> list[dict[str, str | None]]:
    if role == "user" and IMAGE_TOKEN in value:
        text = value.replace(IMAGE_TOKEN, "").strip()
        content = [{"type": "image"}]
        if text:
            content.append({"type": "text", "text": text})
        return content

    return [{"type": "text", "text": value}]


def llava_to_chat_messages(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    for turn in conversations:
        source_role = str(turn.get("from", "")).strip().lower()
        role = ROLE_MAP.get(source_role)
        if role is None:
            raise ValueError(f"Unsupported LLaVA conversation role: {source_role!r}")

        value = str(turn.get("value", ""))
        messages.append(
            {
                "role": role,
                "content": chat_content(role, value),
            }
        )

    return messages


def prepare_output_row(
    row: dict[str, Any],
    *,
    segments: bool,
) -> dict[str, Any]:
    conversations = row.get("conversations", [])
    row["conversations"] = llava_to_chat_messages(conversations)

    row["segments"] = segments

    return row


def load_extra_llava_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filename in LLAVA_EXTRA_FILES:
        rows.extend(load_hub_json(filename))
    return rows


def save_hf_dataset(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(output_dir))


def build_summary(
    args: argparse.Namespace,
    stats: dict[str, int],
) -> dict[str, Any]:
    summary: dict[str, Any] = dict(stats)
    summary.update(
        {
            "llava_repo_id": LLAVA_REPO_ID,
            "llava_detail_file": LLAVA_DETAIL_FILE,
            "llava_extra_files": list(LLAVA_EXTRA_FILES),
            "coconut_dataset_id": args.coconut_dataset_id,
            "coconut_split": "train",
            "hf_output_dir": str(args.hf_output_dir),
        }
    )
    return summary


def main() -> None:
    args = parse_args()

    detail_rows = load_hub_json(LLAVA_DETAIL_FILE)
    coconut_captions, coconut_stats = load_coconut_index(
        dataset_id=args.coconut_dataset_id,
        split="train",
    )

    replaced_rows, stats = replace_detail_rows(
        detail_rows=detail_rows,
        captions_by_image_id=coconut_captions,
    )
    stats.update(coconut_stats)

    output_rows = [
        prepare_output_row(
            row,
            segments=True,
        )
        for row in replaced_rows
    ]

    extra_rows = [
        prepare_output_row(
            row,
            segments=False,
        )
        for row in load_extra_llava_rows()
    ]
    output_rows.extend(extra_rows)

    save_hf_dataset(output_rows, args.hf_output_dir)

    summary = build_summary(args, stats)
    summary["total_output_rows"] = len(output_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
