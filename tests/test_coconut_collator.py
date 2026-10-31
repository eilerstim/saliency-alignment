"""Regression tests for COCONut segment-ID alignment in the collator.

The tokenizer below attaches the preceding space to a word ("▁bear") and
splits rare words into subwords ("▁grass" "##y"), like the SentencePiece and
BPE tokenizers of the supported models. Tokenizing each annotated span on its
own would then produce different tokens than the model input ("Prefix " ->
"Prefix" "▁", "a brown bear" -> "a" ...), shifting segments onto neighbours.
"""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from finetune.data.coconut.collator import make_collate_fn

SPECIAL = ["<pad>", "<unk>", "<image>", "<user>", "<assistant>", "<end>"]
WORDS = [
    "Prefix",
    "▁a",
    "a",
    "▁brown",
    "▁bear",
    "▁sits",
    "▁on",
    "▁grass",
    "##y",
    "▁rocks",
    ".",
    "▁",
    "Describe",
    "▁the",
    "▁image",
    "▁in",
    "▁detail",
    "[INST]",
    "▁[/INST]",
    "▁Prefix",
]
N_IMAGE_TOKENS = 3


def make_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(SPECIAL + WORDS)}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never"),
            pre_tokenizers.Punctuation(),
        ]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>"
    )
    fast.add_special_tokens({"additional_special_tokens": SPECIAL[2:]})
    return fast


class FakeProcessor:
    """Chat template plus image-placeholder expansion, like HF VLM processors.

    ``header=True``: ``<user><image>prompt<assistant>caption<end>`` where the
    generation prompt is ``<assistant>``. ``header=False`` mimics Mistral's
    ``[INST] ... [/INST] caption`` where the generation prompt is empty.
    """

    def __init__(self, header: bool, padding_side: str = "right"):
        self.tokenizer = make_tokenizer()
        self.tokenizer.padding_side = padding_side
        self.header = header

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        out = ""
        for m in messages:
            text = "".join(
                "<image>" if c["type"] == "image" else c["text"] for c in m["content"]
            )
            if m["role"] == "user":
                out += f"<user>{text}" if self.header else f"[INST]{text} [/INST]"
            else:
                out += f"<assistant>{text}<end>" if self.header else f" {text}<end>"
        if add_generation_prompt and self.header:
            out += "<assistant>"
        return out

    def __call__(self, text, images, return_tensors, **kwargs):
        expanded = [t.replace("<image>", "<image>" * N_IMAGE_TOKENS) for t in text]
        return self.tokenizer(
            expanded, padding=True, add_special_tokens=False, return_tensors="pt"
        )


CAPTION = "Prefix <1: a brown bear> sits on <2,3: grassy rocks>."
# Expected (token, segment IDs) for the caption as the model sees it
EXPECTED = [
    ("▁Prefix", []),
    ("▁a", [1]),
    ("▁brown", [1]),
    ("▁bear", [1]),
    ("▁sits", []),
    ("▁on", []),
    ("▁grass", [2, 3]),
    ("##y", [2, 3]),
    ("▁rocks", [2, 3]),
    (".", []),
    ("<end>", []),
]


def example(caption: str) -> dict:
    return {"image": None, "caption": caption, "mask": torch.zeros(2, 2)}


def caption_tokens(batch: dict, processor: FakeProcessor, b: int):
    """(token, segment IDs) for every labelled token of item ``b``."""
    labelled = (batch["labels"][b] != -100).nonzero().flatten()
    tokens = processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][b, labelled])
    segs = batch["segment_ids"][b, labelled]
    return [(t, s[s != -1].tolist()) for t, s in zip(tokens, segs, strict=True)]


def strip_space(pairs):
    # The header template has no space before the caption ("<assistant>Prefix")
    return [(t.removeprefix("▁") if i == 0 else t, s) for i, (t, s) in enumerate(pairs)]


def test_segments_follow_in_context_tokenization():
    processor = FakeProcessor(header=True)
    batch = make_collate_fn(processor)([example(CAPTION)])
    assert caption_tokens(batch, processor, 0) == strip_space(EXPECTED)


def test_empty_assistant_header_masks_prompt():
    """With an empty generation prompt (Mistral-style), the prompt and image
    tokens must still be masked and segments must start at the caption."""
    processor = FakeProcessor(header=False)
    batch = make_collate_fn(processor)([example(CAPTION)])
    assert caption_tokens(batch, processor, 0) == EXPECTED
    image_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_pos = batch["input_ids"][0] == image_id
    assert (batch["labels"][0][image_pos] == -100).all()
    assert (batch["segment_ids"][0][image_pos] == -1).all()


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_batch_with_padding(padding_side):
    processor = FakeProcessor(header=True, padding_side=padding_side)
    short = "<4: a bear>."
    batch = make_collate_fn(processor)([example(CAPTION), example(short)])
    assert caption_tokens(batch, processor, 0) == strip_space(EXPECTED)
    assert caption_tokens(batch, processor, 1) == [
        ("a", [4]),
        ("▁bear", [4]),
        (".", []),
        ("<end>", []),
    ]
    pad = batch["input_ids"] == processor.tokenizer.pad_token_id
    assert (batch["labels"][pad] == -100).all()
    assert (batch["segment_ids"][pad] == -1).all()


def test_mask_cropped_like_image():
    """Masks are cropped to the region the vision grid covers (center crop)."""
    from transformers import CLIPImageProcessor

    processor = FakeProcessor(header=True)
    processor.image_processor = CLIPImageProcessor(
        size={"shortest_edge": 336},
        crop_size={"height": 336, "width": 336},
        do_center_crop=True,
        do_resize=True,
    )
    mask = torch.arange(640).repeat(480, 1)  # each pixel = its column index
    batch = make_collate_fn(processor)(
        [{"image": None, "caption": CAPTION, "mask": mask}]
    )
    (out,) = batch["masks"]
    assert out.shape == (480, 480)
    assert out[0, 0] == 80 and out[0, -1] == 559
