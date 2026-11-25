"""Regression tests for the token <-> saliency-row pairing of the alignment loss.

The saliency row at query position ``q`` is built from that query's attention.
The criterion and the alignment metrics pair every supervised target token at
position ``p`` with the row of query ``p - QUERY_SHIFT``:

* ``QUERY_SHIFT = 0``: the target's own position, whose query has the token as
  input. This is the read-out of the paper's main experiments.
* ``QUERY_SHIFT = 1``: the preceding position, whose hidden state predicts the
  token and has not seen it yet (a causal decoder predicts the token at ``p``
  from the state at ``p - 1``). This is the read-out of the paper's Appendix G.

``QUERY_SHIFT`` below states which of the two the code implements; it has to
match the slices in ``finetune/criterion/base.py`` and
``align_eval/metrics.py`` (see "Attention read-out position" in the README).

Each saliency row puts a sharp peak on a patch that encodes its query
position, and each target's segment mask covers exactly the patch of the
query it is paired with. The right pairing therefore gives a near-zero KL; an
off-by-one pairing puts the peak outside the mask.
"""

import pytest
import torch
from torch import Tensor

from finetune.criterion import Criterion
from finetune.criterion.kl import SaliencyAlignment as KLAlignment
from vl_saliency._core.seq_layout import SequenceLayout
from vl_saliency.config import SaliencyConfig, StaticPatchLayout
from vl_saliency.maps import SaliencyGrid

# Rows back from a target's own position that are read for it: 0 in the
# paper's main experiments, 1 at the predicting position (Appendix G).
QUERY_SHIFT = 0

PAD, BOS, EOS, IMG = 0, 1, 2, 3
HEADER = [4, 5]  # assistant header tokens between the image and the caption
CAPTION_TOKEN = 10
SIDE = 4  # 4x4 patch grid, annotation mask at the same resolution
N_PATCHES = SIDE * SIDE
PEAK = 50.0


def patch_of(query_pos: int) -> int:
    return query_pos % N_PATCHES


def seg_of_patch(patch: int) -> int:
    # Annotation mask pixel k carries segment id k + 1 (0 is void)
    return patch + 1


def build_batch(
    caption_lens: list[int],
    supervised: list[dict[int, list[int]]],
    padding_side: str = "right",
):
    """Build a batch laid out as BOS, IMG*16, HEADER, caption..., EOS.

    ``supervised[b]`` maps a caption offset (0-based, EOS is offset
    ``caption_len``) to the list of query positions whose patches the target's
    segment should cover.
    """
    seqs = [
        [BOS] + [IMG] * N_PATCHES + HEADER + [CAPTION_TOKEN] * n + [EOS]
        for n in caption_lens
    ]
    seq_len = max(len(s) for s in seqs)
    B = len(seqs)

    input_ids = torch.full((B, seq_len), PAD, dtype=torch.long)
    labels = torch.full((B, seq_len), -100, dtype=torch.long)
    segment_ids = torch.full((B, seq_len, 2), -1, dtype=torch.long)

    for b, seq in enumerate(seqs):
        start = 0 if padding_side == "right" else seq_len - len(seq)
        input_ids[b, start : start + len(seq)] = torch.tensor(seq)
        caption_start = start + 1 + N_PATCHES + len(HEADER)
        labels[b, caption_start : start + len(seq)] = input_ids[
            b, caption_start : start + len(seq)
        ]
        for offset, query_positions in supervised[b].items():
            p = caption_start + offset
            for k, q in enumerate(query_positions):
                segment_ids[b, p, k] = seg_of_patch(patch_of(q))

    config = SaliencyConfig(
        image_token_id=IMG,
        pad_token_id=PAD,
        patch_layout_fn=StaticPatchLayout(SIDE, SIDE),
        attn_scale=1.0,
        head_op=None,
        head_reduce="mean",
        layer_op=None,
        layer_reduce="mean",
        selection=None,
        backend="torch_eager",
    )
    layout = SequenceLayout(
        config, input_ids=input_ids, pixel_values=torch.zeros(B, 3, 8, 8)
    )

    # Row t of item b belongs to query position gen_token_idx[b, t]
    tensor = torch.zeros(B, layout.T_gen, N_PATCHES)
    for b in range(B):
        for t, q in enumerate(layout.gen_token_idx[b].tolist()):
            if q >= 0:
                tensor[b, t, patch_of(q)] = PEAK
    saliency = SaliencyGrid(tensor, layout)

    mask = (torch.arange(N_PATCHES) + 1).view(SIDE, SIDE)
    masks = [mask.clone() for _ in range(B)]
    preds = torch.zeros(B, seq_len, 1)
    return labels, segment_ids, preds, saliency, masks, caption_start_of(input_ids)


def caption_start_of(input_ids: Tensor) -> list[int]:
    return [int((row == IMG).nonzero().max()) + 1 + len(HEADER) for row in input_ids]


class RecordingCriterion(Criterion):
    """Records which patch each supervised row peaks on and which it targets."""

    def __init__(self):
        super().__init__(weight=1.0)
        self.calls: list[tuple[list[int], list[set[int]]]] = []

    def compute_loss(self, attn, mask):
        peaks = attn.flatten(1).argmax(1).tolist()
        targets = [set(m.flatten().nonzero().flatten().tolist()) for m in mask]
        self.calls.append((peaks, targets))
        return attn.sum() * 0


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_targets_pair_with_supervised_query(padding_side):
    """Every target, including the first caption token and the final EOS, is
    paired with the row of the query ``QUERY_SHIFT`` positions before it."""
    # Item 0: caption "a brown bear" (+EOS) and item 1 shorter, so padding is used.
    # Each target's segment covers the patch of the query it is paired with.
    lens = [3, 2]
    labels, segment_ids, preds, saliency, masks, starts = build_batch(
        lens,
        supervised=[{}, {}],
        padding_side=padding_side,
    )
    supervised = []
    for b, n in enumerate(lens):
        # offsets 0 (first caption token), n - 1 ("bear"), n (EOS, final target)
        supervised.append(
            {o: [starts[b] + o - QUERY_SHIFT] for o in (0, n - 1, n)}
        )
    labels, segment_ids, preds, saliency, masks, _ = build_batch(
        lens, supervised, padding_side=padding_side
    )

    criterion = RecordingCriterion()
    criterion(labels, segment_ids, preds, saliency, masks)

    assert len(criterion.calls) == len(lens)
    for peaks, targets in criterion.calls:
        assert len(peaks) == 3
        for peak, target in zip(peaks, targets, strict=True):
            assert peak in target


def test_neighbouring_row_is_not_supervised():
    """The row one position away (the predicting position when the target's
    own row is read, and vice versa) must not be the one supervised."""
    lens = [3]
    _, _, _, _, _, starts = build_batch(lens, supervised=[{}])
    bear = starts[0] + 2
    supervised_query = bear - QUERY_SHIFT
    other_query = bear - (1 - QUERY_SHIFT)
    labels, segment_ids, preds, saliency, masks, _ = build_batch(
        lens, supervised=[{2: [other_query]}]
    )

    criterion = RecordingCriterion()
    criterion(labels, segment_ids, preds, saliency, masks)

    ((peaks, targets),) = criterion.calls
    assert peaks == [patch_of(supervised_query)]
    assert patch_of(other_query) in targets[0]
    assert peaks[0] not in targets[0]


def test_kl_loss_near_zero_when_aligned():
    """End-to-end KL: masks matched to the supervised rows give ~0 loss."""
    lens = [4]
    _, _, _, _, _, starts = build_batch(lens, supervised=[{}])
    s = starts[0]
    supervised = [{o: [s + o - QUERY_SHIFT] for o in (0, 2, 4)}]
    labels, segment_ids, preds, saliency, masks, _ = build_batch(lens, supervised)

    loss = KLAlignment(weight=1.0)(labels, segment_ids, preds, saliency, masks)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_multi_token_word_subwords_use_their_own_rows():
    """Subwords of one annotated word share segments, but each is supervised
    through its own row (the one ``QUERY_SHIFT`` positions before it)."""
    lens = [4]
    _, _, _, _, _, starts = build_batch(lens, supervised=[{}])
    s = starts[0]
    # offsets 1 and 2 form one two-token word
    word = [s + 1 - QUERY_SHIFT, s + 2 - QUERY_SHIFT]
    labels, segment_ids, preds, saliency, masks, _ = build_batch(
        lens, supervised=[{1: word, 2: word}]
    )

    criterion = RecordingCriterion()
    criterion(labels, segment_ids, preds, saliency, masks)

    ((peaks, targets),) = criterion.calls
    assert peaks == [patch_of(word[0]), patch_of(word[1])]
    assert targets[0] == targets[1] == {patch_of(q) for q in word}


def test_unsupervised_and_padding_ignored():
    """Items without segments contribute nothing and padding rows are unused."""
    lens = [3, 1]
    labels, segment_ids, preds, saliency, masks, starts = build_batch(
        lens, supervised=[{}, {}]
    )
    query = starts[0] + 1 - QUERY_SHIFT
    labels, segment_ids, preds, saliency, masks, _ = build_batch(
        lens, supervised=[{1: [query]}, {}]
    )

    criterion = RecordingCriterion()
    criterion(labels, segment_ids, preds, saliency, masks)

    assert len(criterion.calls) == 1
    ((peaks, targets),) = criterion.calls
    assert peaks == [patch_of(query)]
    assert peaks[0] in targets[0]


def test_eval_metrics_use_same_alignment():
    """align_eval mirrors the criterion: a peak on the supervised row scores
    the maximum chance-normalised AMR (N_PATCHES), not ~0."""
    from align_eval.metrics import per_image_scores

    lens = [3]
    _, _, _, _, _, starts = build_batch(lens, supervised=[{}])
    s = starts[0]
    supervised = [{o: [s + o - QUERY_SHIFT] for o in range(lens[0] + 1)}]
    labels, segment_ids, _, saliency, masks, _ = build_batch(lens, supervised)

    scores = per_image_scores(saliency, masks, segment_ids, labels)
    assert scores[0, 0].item() == pytest.approx(N_PATCHES, rel=1e-3)


def test_supervised_row_belongs_to_the_expected_query():
    """End-to-end through a tiny LLaVA under the real Saliency hook: the row
    the criterion supervises for "bear" is the one of the query
    ``QUERY_SHIFT`` positions before it. Causal attention makes that
    observable: a query's row changes with its own input token and is
    unaffected by any later token. This also guards against a double shift if
    vl_saliency ever realigns rows itself."""
    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
    )

    from vl_saliency import Saliency

    torch.manual_seed(0)
    vision = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    text = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=PAD,
    )
    config = LlavaConfig(
        vision_config=vision,
        text_config=text,
        image_token_index=IMG,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        pad_token_id=PAD,
    )
    config._attn_implementation = "eager"
    model = LlavaForConditionalGeneration(config).eval()

    # BOS IMG*4 HEADER "a" "brown" "bear" EOS PAD
    a, brown, bear = 11, 12, 13
    input_ids = torch.tensor([[BOS] + [IMG] * 4 + HEADER + [a, brown, bear, EOS, PAD]])
    bear_pos = 1 + 4 + len(HEADER) + 2
    labels = input_ids.clone()
    labels[:, : 1 + 4 + len(HEADER)] = -100
    labels[input_ids == PAD] = -100
    segment_ids = torch.full((*input_ids.shape, 1), -1)
    segment_ids[0, bear_pos, 0] = 1
    masks = [torch.ones(2, 2, dtype=torch.long)]
    pixel_values = torch.randn(1, 3, 8, 8)

    def supervised_rows(ids):
        with Saliency(model, backend="torch_eager"), torch.no_grad():
            out = model(
                input_ids=ids,
                attention_mask=(ids != PAD).long(),
                pixel_values=pixel_values,
                return_dict=True,
            )
        criterion = RecordingAttn()
        criterion(labels, segment_ids, out.logits, out.saliency, masks)
        return criterion.attn

    query_pos = bear_pos - QUERY_SHIFT

    # The token after the supervised query cannot reach its row.
    later = input_ids.clone()
    later[0, query_pos + 1] = 30
    torch.testing.assert_close(supervised_rows(input_ids), supervised_rows(later))

    # The query's own input token does, so the check above can fail.
    own = input_ids.clone()
    own[0, query_pos] = 30
    assert not torch.allclose(supervised_rows(input_ids), supervised_rows(own))


class RecordingAttn(Criterion):
    def compute_loss(self, attn, mask):
        self.attn = attn
        return attn.sum() * 0


def test_token_with_empty_mask_is_dropped():
    """A token whose referent lies outside the (cropped) mask carries no
    supervision and must not dilute the loss of the other tokens."""
    lens = [4]
    _, _, _, _, _, starts = build_batch(lens, supervised=[{}])
    s = starts[0]
    labels, segment_ids, preds, saliency, masks, _ = build_batch(
        lens, supervised=[{0: [s - QUERY_SHIFT], 2: [s + 2 - QUERY_SHIFT]}]
    )
    # Point offset 2 at a segment that does not exist in the mask
    segment_ids[0, s + 2, 0] = N_PATCHES + 99

    criterion = RecordingCriterion()
    criterion(labels, segment_ids, preds, saliency, masks)
    ((peaks, targets),) = criterion.calls
    assert len(peaks) == 1 and all(t for t in targets)
