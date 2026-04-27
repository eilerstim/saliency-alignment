"""Compare saved fine-tuned checkpoints against the base HF model.

Version-robust: walks ``model.named_modules()`` and matches components by
suffix, so it works whether ``transformers`` nests LLaVA submodules as
``model.language_model`` (older) or ``model.model.language_model`` (newer).

Usage:
    python scripts/compare_drift.py \\
        --base llava-hf/llava-1.5-7b-hf \\
        models/llava-1.5-7b_kl_w0.5_proj_only \\
        models/llava-1.5-7b_kl_w0.5_lm_only \\
        models/llava-1.5-7b_kl_w0.5_lm_proj \\
        --baseline llava-1.5-7b_kl_w0.5_proj_only

Two-checkpoint usage (e.g., lm+proj vs base only):
    python scripts/compare_drift.py \\
        --base llava-hf/llava-1.5-7b-hf \\
        models/llava-1.5-7b_kl_w0.5_lm_proj

Headline test for Hypothesis 1: if the ``multi_modal_projector``'s
``relative_update`` in the lm+proj run is substantially smaller than in
the proj_only run, the LM is absorbing the saliency-loss gradient and
the projector is being starved. With only a single lm+proj run, compare
its projector rel_update to its language_model rel_update within the
same run: a huge disparity is the same smoking gun.

Runs on CPU; peak RAM ~60 GB for base + one 7B checkpoint in fp32.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

logger = logging.getLogger(__name__)


def find_component(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    """Return the submodule whose last path segment matches ``name``.

    Walks ``named_modules()`` and matches by suffix so we don't care whether
    the model is the older top-level layout (``model.language_model``) or
    the newer nested layout (``model.model.language_model``). Prefers the
    shortest matching path to avoid grabbing a deeply nested namesake.
    """
    candidates: list[tuple[str, torch.nn.Module]] = []
    for qualified_name, module in model.named_modules():
        if not qualified_name:
            continue
        tail = qualified_name.split(".")[-1]
        if tail == name:
            candidates.append((qualified_name, module))

    if not candidates:
        return None
    # Prefer the highest-level match (shortest path)
    candidates.sort(key=lambda kv: kv[0].count("."))
    qualified_name, module = candidates[0]
    logger.debug("  Resolved '%s' -> '%s'", name, qualified_name)
    return module


def component_drift(
    base_model: torch.nn.Module,
    tuned_model: torch.nn.Module,
    component_names: list[str],
) -> dict:
    """Compute per-parameter and per-component drift between two models."""
    report: dict = {"components": {}, "parameters": {}}

    for cname in component_names:
        base_comp = find_component(base_model, cname)
        tuned_comp = find_component(tuned_model, cname)
        if base_comp is None or tuned_comp is None:
            logger.warning(
                "Component '%s' missing on one of the models (base=%s, tuned=%s); skipping",
                cname,
                base_comp is not None,
                tuned_comp is not None,
            )
            continue

        base_params = dict(base_comp.named_parameters())
        tuned_params = dict(tuned_comp.named_parameters())

        comp_init_sq = 0.0
        comp_delta_sq = 0.0
        comp_n_elements = 0
        n_updated = 0

        for pname, base_p in base_params.items():
            if pname not in tuned_params:
                logger.warning("Param '%s.%s' missing in tuned model", cname, pname)
                continue
            tuned_p = tuned_params[pname]

            base = base_p.detach().float().cpu()
            tuned = tuned_p.detach().float().cpu()
            if base.shape != tuned.shape:
                logger.warning(
                    "Shape mismatch for %s.%s: %s vs %s",
                    cname,
                    pname,
                    tuple(base.shape),
                    tuple(tuned.shape),
                )
                continue

            delta = tuned - base
            init_norm = base.norm().item()
            delta_norm = delta.norm().item()
            rel_update = delta_norm / max(init_norm, 1e-12)

            report["parameters"][f"{cname}.{pname}"] = {
                "n_elements": int(base.numel()),
                "init_norm": init_norm,
                "final_norm": tuned.norm().item(),
                "delta_norm": delta_norm,
                "relative_update": rel_update,
                "max_abs_delta": delta.abs().max().item(),
            }

            comp_init_sq += init_norm**2
            comp_delta_sq += delta_norm**2
            comp_n_elements += int(base.numel())
            if delta_norm > 0:
                n_updated += 1

        comp_init_norm = comp_init_sq**0.5
        comp_delta_norm = comp_delta_sq**0.5
        report["components"][cname] = {
            "n_tensors": len(base_params),
            "n_tensors_updated": n_updated,
            "n_elements": comp_n_elements,
            "init_norm": comp_init_norm,
            "delta_norm": comp_delta_norm,
            "relative_update": comp_delta_norm / max(comp_init_norm, 1e-12),
        }

    return report


def load_model(path: str) -> torch.nn.Module:
    """Load a model from disk or HF hub, on CPU, in fp32."""
    logger.info("Loading %s ...", path)
    model = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    return model


def fmt_sci(x: float) -> str:
    return f"{x:.3e}"


def debug_structure(model: torch.nn.Module, depth: int = 2) -> None:
    """Print the top-level module tree to help diagnose attribute paths."""
    logger.info("Top-level structure of loaded model:")
    for qualified_name, _ in model.named_modules():
        if not qualified_name:
            continue
        if qualified_name.count(".") < depth:
            logger.info("  %s", qualified_name)


# --- Per-layer LM attention drift -----------------------------------------


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _count_lm_layers(model: torch.nn.Module) -> int:
    """Return the number of decoder layers in the model's language_model.

    Walks the FULL model's named_parameters and picks the max ``layers.<N>``
    index found on a ``self_attn.q_proj.weight`` tensor. Returns ``0`` if no
    such parameter is found. We walk the whole model rather than a nested
    ``language_model`` submodule because ``transformers``' LLaVA nesting has
    shifted across versions and the submodule's own ``.named_parameters()``
    output varies.
    """
    max_layer = -1
    for pname, _ in model.named_parameters():
        if "self_attn.q_proj.weight" not in pname:
            continue
        m = _LAYER_RE.search(pname)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    return max_layer + 1


def extract_attention_weights(
    model: torch.nn.Module,
    layers_by_proj: dict[str, list[int]],
) -> dict[tuple[int, str], torch.Tensor]:
    """Pull out attention-projection weights for the listed (layer, proj) pairs.

    Returns ``{(layer_index, proj_name): weight_tensor_cpu_fp32}``. Walks the
    FULL model's named_parameters and matches by suffix (same approach
    :func:`extract_attention_proj_drifts` uses on the per-parameter report),
    so it's robust to however ``transformers`` nests the language_model
    submodule in the current version.
    """
    wanted: set[tuple[int, str]] = {
        (layer_idx, proj)
        for proj, layer_list in layers_by_proj.items()
        for layer_idx in layer_list
    }
    if not wanted:
        return {}

    out: dict[tuple[int, str], torch.Tensor] = {}
    for pname, param in model.named_parameters():
        m = _LAYER_RE.search(pname)
        if not m:
            continue
        layer_idx = int(m.group(1))
        for proj in layers_by_proj:
            key = (layer_idx, proj)
            if key in wanted and f"self_attn.{proj}.weight" in pname:
                out[key] = param.detach().float().cpu().clone()
                break

    missing = wanted - set(out)
    if missing:
        logger.warning(
            "Could not extract %d/%d attention weight tensors: %s",
            len(missing), len(wanted), sorted(missing)[:5],
        )
    return out


def per_head_drift(
    base_weight: torch.Tensor,
    tuned_weight: torch.Tensor,
    num_heads: int,
) -> tuple[list[float], list[float]]:
    """Slice a Q/V weight matrix into per-head blocks and compute per-head drift.

    For LLaMA-style attention, the projection weight has shape
    ``[num_heads * head_dim, hidden_size]``. Rows are partitioned into
    ``num_heads`` contiguous blocks of size ``head_dim``, one per head.
    Each block maps the input hidden state to one head's output. Slicing
    by rows therefore gives us per-head contributions that are not mixed
    across heads.

    Returns ``(per_head_rel_updates, per_head_init_norms)``, each a list
    of length ``num_heads``.
    """
    total_rows, _hidden = base_weight.shape
    head_dim, rem = divmod(total_rows, num_heads)
    if rem != 0:
        raise ValueError(
            f"Weight shape {tuple(base_weight.shape)} not divisible "
            f"by num_heads={num_heads}"
        )

    delta = tuned_weight - base_weight
    rel_updates: list[float] = []
    init_norms: list[float] = []
    for h in range(num_heads):
        lo, hi = h * head_dim, (h + 1) * head_dim
        head_init = base_weight[lo:hi, :]
        head_delta = delta[lo:hi, :]
        init_norm = head_init.norm().item()
        delta_norm = head_delta.norm().item()
        rel_updates.append(delta_norm / max(init_norm, 1e-12))
        init_norms.append(init_norm)
    return rel_updates, init_norms


def extract_attention_proj_drifts(
    report: dict, proj_names: list[str]
) -> dict[str, dict[int, float]]:
    """Pull per-layer, per-projection drift out of a per-parameter report.

    Returns a nested mapping ``{proj_name: {layer_index: relative_update}}``.
    Matches parameters of the form ``*.layers.<N>.(self_attn|mlp).<proj>.weight``
    under the ``language_model`` component. Relies on the per-parameter
    ``relative_update`` already computed by :func:`component_drift`.

    Despite the function name, this works for both attention projections
    (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``) and MLP projections
    (``gate_proj``, ``up_proj``, ``down_proj``) — the MLP rows are useful
    as a control when testing whether adaptation concentrates in attention
    specifically or spreads across the LLM uniformly.
    """
    out: dict[str, dict[int, float]] = {p: {} for p in proj_names}
    for pname, stats in report["parameters"].items():
        if not pname.startswith("language_model."):
            continue
        m = _LAYER_RE.search(pname)
        if not m:
            continue
        layer_idx = int(m.group(1))
        for proj in proj_names:
            # Match both self_attn.<proj>.weight and mlp.<proj>.weight so
            # the per-layer table can include MLP controls alongside Q/V.
            if (
                f"self_attn.{proj}.weight" in pname
                or f"mlp.{proj}.weight" in pname
            ):
                out[proj][layer_idx] = stats["relative_update"]
                break
    return out


def gini(xs: list[float]) -> float:
    """Gini coefficient of a list of non-negative values. 0 = perfectly
    uniform, 1 = all mass in one element. Useful for quantifying how
    concentrated attention-layer drift is across layers."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    total = sum(xs)
    if total <= 0:
        return 0.0
    # Gini = (2 * sum(i * x_i) - (n+1) * sum(x)) / (n * sum(x))
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * weighted - (n + 1) * total) / (n * total)


def print_per_layer_analysis(
    reports: dict[str, dict],
    labels: list[str],
    proj_names: list[str],
    top_k: int,
) -> None:
    """Print per-layer LM attention-projection drift across runs.

    For each run:
      * Lists the top-K most-drifting layers for each projection.
      * Reports the Gini coefficient across layers (concentration).
      * Reports max/mean rel_update ratio.

    Also prints a cross-run comparison of the top-K layers in the
    baseline-of-interest (typically lm_only or lm_proj).
    """
    # Extract drift vectors once per run
    drift_by_run: dict[str, dict[str, dict[int, float]]] = {}
    for label in labels:
        drift_by_run[label] = extract_attention_proj_drifts(
            reports[label], proj_names
        )

    # ---- Summary table: concentration per (run, projection) ----
    print("\n=== Per-layer LM attention-projection drift: concentration ===")
    print(
        "Gini coefficient across layers (0 = uniform, 1 = concentrated in one layer).\n"
        "A high Gini is evidence of 'localization-head'-style sparse adaptation."
    )
    header = (
        f"{'run':<55s} {'projection':>11s} "
        f"{'n_layers':>9s} {'gini':>7s} "
        f"{'mean_rel':>11s} {'max_rel':>11s} "
        f"{'top_layer':>10s}"
    )
    print(header)
    print("-" * len(header))
    for label in labels:
        for proj in proj_names:
            vals = drift_by_run[label][proj]
            if not vals:
                continue
            values = list(vals.values())
            max_layer, max_val = max(vals.items(), key=lambda kv: kv[1])
            print(
                f"{label:<55s} {proj:>11s} "
                f"{len(values):>9d} {gini(values):>7.3f} "
                f"{fmt_sci(sum(values) / len(values)):>11s} "
                f"{fmt_sci(max_val):>11s} "
                f"{max_layer:>10d}"
            )

    # ---- Top-K drifting layers per (run, projection) ----
    print(f"\n=== Top-{top_k} most-drifting LM layers per run ===")
    print(
        "Relative update per layer, sorted descending. If drift concentrates\n"
        "in a small number of layers across runs, it's evidence of mechanistic\n"
        "'localization-head' adaptation (Kang et al. 2025a)."
    )
    for label in labels:
        for proj in proj_names:
            vals = drift_by_run[label][proj]
            if not vals:
                continue
            print(f"\n{label}  |  {proj}")
            ranked = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            for rank, (layer_idx, rel_update) in enumerate(ranked, start=1):
                bar_units = int(60 * rel_update / max(ranked[0][1], 1e-30))
                bar = "#" * bar_units
                print(
                    f"  #{rank:<2d} layer {layer_idx:>3d}  "
                    f"rel_update = {fmt_sci(rel_update)}  {bar}"
                )

    # ---- Cross-run overlap: do the same layers adapt in lm_only and lm_proj? ----
    if len(labels) >= 2:
        print(
            f"\n=== Cross-run top-{top_k} layer overlap ===\n"
            "If lm_only and lm_proj adapt the SAME layers, that's evidence\n"
            "the LM's solution is stable regardless of whether the projector\n"
            "is also trainable — supporting the 'non-compositional solutions' story."
        )
        for proj in proj_names:
            top_layers_per_run: dict[str, set[int]] = {}
            for label in labels:
                vals = drift_by_run[label][proj]
                if not vals:
                    continue
                top_layers_per_run[label] = {
                    layer_idx
                    for layer_idx, _ in sorted(
                        vals.items(), key=lambda kv: kv[1], reverse=True
                    )[:top_k]
                }
            if len(top_layers_per_run) < 2:
                continue

            print(f"\n{proj} — top-{top_k} layers per run:")
            for label, layers in top_layers_per_run.items():
                sorted_layers = sorted(layers)
                print(f"  {label:<55s} {sorted_layers}")

            # Pairwise Jaccard
            run_items = list(top_layers_per_run.items())
            print(f"\n{proj} — pairwise top-{top_k} Jaccard overlap:")
            for i, (la, sa) in enumerate(run_items):
                for lb, sb in run_items[i + 1 :]:
                    if sa or sb:
                        jaccard = len(sa & sb) / len(sa | sb)
                    else:
                        jaccard = 0.0
                    shared = sorted(sa & sb)
                    print(
                        f"  {la:<40s} vs {lb:<40s} "
                        f"J={jaccard:.2f}  shared={shared}"
                    )


def print_per_head_analysis(
    per_head_data: dict[str, dict[tuple[int, str], list[float]]],
    labels: list[str],
    proj_names: list[str],
    head_top_k: int,
    num_heads: int,
) -> None:
    """Print per-head drift analysis for the pre-selected (run, layer, proj) cells.

    ``per_head_data`` is ``{label: {(layer_idx, proj): [rel_update_per_head]}}``.

    For each (run, layer, proj) cell we report Gini across heads and the
    top-K heads. Then we compute cross-run Jaccard on top-K head sets: if
    lm_only and lm_proj pick the same heads within the same layer, that's
    evidence of genuine localization-head convergence rather than
    coincidental layer-level alignment.
    """
    if not per_head_data:
        print("\n(No per-head data collected — skipping per-head analysis.)")
        return

    # Collect the (layer, proj) cells actually present
    all_cells = sorted({cell for d in per_head_data.values() for cell in d})

    print("\n=== Per-HEAD LM attention drift: concentration ===")
    print(
        f"For each analysed (layer, projection) cell, slice the weight matrix\n"
        f"into {num_heads} per-head blocks and measure drift within each block.\n"
        f"High Gini => drift concentrates in a small number of heads — the\n"
        f"'localization-head' signature of Kang et al. 2025a."
    )
    header = (
        f"{'run':<40s} {'layer':>5s} {'proj':>7s} "
        f"{'gini':>7s} {'mean_rel':>11s} {'max_rel':>11s} "
        f"{'top_head':>8s} {'n_heads_above_mean':>18s}"
    )
    print(header)
    print("-" * len(header))
    for label in labels:
        cells = per_head_data.get(label, {})
        for layer_idx, proj in all_cells:
            vals = cells.get((layer_idx, proj))
            if not vals:
                continue
            g = gini(vals)
            mean_rel = sum(vals) / len(vals)
            max_rel = max(vals)
            top_head = vals.index(max_rel)
            n_above = sum(1 for v in vals if v > mean_rel)
            print(
                f"{label:<40s} {layer_idx:>5d} {proj:>7s} "
                f"{g:>7.3f} {fmt_sci(mean_rel):>11s} {fmt_sci(max_rel):>11s} "
                f"{top_head:>8d} {n_above:>18d}"
            )

    # Per-cell top-K heads
    print(f"\n=== Top-{head_top_k} heads per (run, layer, projection) ===")
    for layer_idx, proj in all_cells:
        print(f"\n-- layer {layer_idx}, {proj} --")
        for label in labels:
            vals = per_head_data.get(label, {}).get((layer_idx, proj))
            if not vals:
                continue
            ranked = sorted(enumerate(vals), key=lambda kv: kv[1], reverse=True)[
                :head_top_k
            ]
            peak = ranked[0][1] if ranked else 1e-30
            print(f"  {label}:")
            for rank, (head_idx, rel) in enumerate(ranked, start=1):
                bar = "#" * int(40 * rel / max(peak, 1e-30))
                print(
                    f"    #{rank:<2d} head {head_idx:>3d}  "
                    f"rel_update = {fmt_sci(rel)}  {bar}"
                )

    # Cross-run head overlap within each (layer, proj) cell
    if len(labels) >= 2:
        print(
            f"\n=== Cross-run per-HEAD top-{head_top_k} overlap ===\n"
            f"Within each (layer, projection) cell, are the TOP heads the same\n"
            f"across runs? High Jaccard => supervised saliency consistently\n"
            f"activates the same localization heads regardless of which\n"
            f"components are trainable."
        )
        for layer_idx, proj in all_cells:
            top_by_run: dict[str, set[int]] = {}
            for label in labels:
                vals = per_head_data.get(label, {}).get((layer_idx, proj))
                if not vals:
                    continue
                top_by_run[label] = {
                    head_idx
                    for head_idx, _ in sorted(
                        enumerate(vals), key=lambda kv: kv[1], reverse=True
                    )[:head_top_k]
                }
            if len(top_by_run) < 2:
                continue
            print(f"\nlayer {layer_idx}, {proj} — top-{head_top_k} heads:")
            for label, heads in top_by_run.items():
                print(f"  {label:<40s} {sorted(heads)}")
            print(f"layer {layer_idx}, {proj} — pairwise Jaccard:")
            items = list(top_by_run.items())
            for i, (la, sa) in enumerate(items):
                for lb, sb in items[i + 1 :]:
                    jaccard = len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0
                    shared = sorted(sa & sb)
                    print(
                        f"  {la:<40s} vs {lb:<40s} "
                        f"J={jaccard:.2f}  shared={shared}"
                    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="Paths to fine-tuned model directories. Directory name is "
        "used as the run label.",
    )
    ap.add_argument(
        "--base",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HF hub id or local path of the base model to diff against.",
    )
    ap.add_argument(
        "--components",
        nargs="+",
        default=["multi_modal_projector", "language_model", "vision_tower"],
        help="Submodule names to compare (matched by last path segment).",
    )
    ap.add_argument(
        "--save-reports",
        action="store_true",
        help="Write drift_report.json into each checkpoint dir.",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Run label to use as denominator for cross-run ratios. "
        "Defaults to the first checkpoint passed.",
    )
    ap.add_argument(
        "--debug-structure",
        action="store_true",
        help="Print the top-level module tree of the base model and exit.",
    )
    ap.add_argument(
        "--per-layer",
        action="store_true",
        help="Also analyse per-layer LM attention-projection drift "
        "(q_proj, k_proj, v_proj, o_proj). Tests whether saliency "
        "supervision concentrates in a small number of 'localization heads'.",
    )
    ap.add_argument(
        "--per-layer-projs",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Attention projection names to analyse per layer. "
        "Default: q_proj v_proj (the routing-relevant ones for Kang et al.).",
    )
    ap.add_argument(
        "--per-layer-top-k",
        type=int,
        default=10,
        help="Print the top-K most-drifting layers in the per-layer table.",
    )
    ap.add_argument(
        "--per-head",
        action="store_true",
        help="Also slice the top-drifting layers into per-head blocks and "
        "measure per-head drift. Requires --per-layer (auto-enabled). "
        "Tests whether drift concentrates in a small set of 'localization heads'.",
    )
    ap.add_argument(
        "--per-head-layers-k",
        type=int,
        default=5,
        help="How many top-drifting layers (per projection) to analyse per-head. "
        "Layers are auto-selected from the first run with non-zero drift.",
    )
    ap.add_argument(
        "--per-head-top-k",
        type=int,
        default=8,
        help="Top-K heads to list per (run, layer, projection) cell.",
    )
    ap.add_argument(
        "--num-heads",
        type=int,
        default=32,
        help="Number of attention heads in the LM (32 for LLaMA/Vicuna-7B, "
        "which is LLaVA-1.5-7B's backbone).",
    )
    args = ap.parse_args()

    base_model = load_model(args.base)
    if args.debug_structure:
        debug_structure(base_model)
        return 0

    # If per-head analysis is requested, ensure per-layer is also enabled
    # (we derive head-analysis layers from per-layer top-K).
    if args.per_head and not args.per_layer:
        args.per_layer = True
        logger.info("--per-head implies --per-layer; enabling it.")

    # If we'll need per-head analysis, snapshot ALL attention weights for the
    # requested projections across ALL layers from the BASE model up front,
    # so we can slice per-head deltas without reloading anything.
    base_attn_weights: dict[tuple[int, str], torch.Tensor] = {}
    if args.per_head:
        # Only slice canonical attention projections per-head. MLP projections
        # (gate_proj/up_proj/down_proj) pass through --per-layer-projs but
        # don't have a meaningful per-head structure, so we exclude them here.
        attn_projs_for_head = [
            p for p in args.per_layer_projs
            if p in {"q_proj", "k_proj", "v_proj", "o_proj"}
        ]
        if not attn_projs_for_head:
            logger.warning(
                "--per-head requested but no attention projection is in "
                "--per-layer-projs; skipping per-head analysis."
            )
            args.per_head = False
        else:
            n_layers = _count_lm_layers(base_model)
            if n_layers == 0:
                logger.warning(
                    "Could not determine number of LM layers; skipping per-head."
                )
                args.per_head = False
            else:
                all_layers_by_proj = {
                    proj: list(range(n_layers)) for proj in attn_projs_for_head
                }
                base_attn_weights = extract_attention_weights(
                    base_model, all_layers_by_proj
                )
                logger.info(
                    "Snapshotted %d base attention weight tensors for per-head analysis "
                    "(projections: %s, %d layers)",
                    len(base_attn_weights),
                    attn_projs_for_head,
                    n_layers,
                )

    labels = [p.name for p in args.checkpoints]
    reports: dict[str, dict] = {}
    # {label: {(layer_idx, proj): [rel_update_per_head]}}
    per_head_data: dict[str, dict[tuple[int, str], list[float]]] = {}

    for path, label in zip(args.checkpoints, labels, strict=True):
        tuned_model = load_model(str(path))
        report = component_drift(base_model, tuned_model, args.components)
        reports[label] = report

        if args.save_reports:
            out = path / "drift_report.json"
            with out.open("w") as f:
                json.dump(report, f, indent=2)
            logger.info("Wrote %s", out)

        if args.per_head and base_attn_weights:
            layers_by_proj = {
                proj: sorted({layer for (layer, p) in base_attn_weights if p == proj})
                for proj in args.per_layer_projs
            }
            tuned_attn_weights = extract_attention_weights(tuned_model, layers_by_proj)
            cells: dict[tuple[int, str], list[float]] = {}
            for key, base_w in base_attn_weights.items():
                tuned_w = tuned_attn_weights.get(key)
                if tuned_w is None:
                    continue
                try:
                    rel, _ = per_head_drift(base_w, tuned_w, args.num_heads)
                    cells[key] = rel
                except ValueError as e:
                    logger.warning("Skipping head-slice for %s: %s", key, e)
            per_head_data[label] = cells
            del tuned_attn_weights

        del tuned_model
        gc.collect()

    # ---- Per-component absolute table ----
    for cname in args.components:
        print(f"\n=== {cname} ===")
        header = (
            f"{'run':<55s} "
            f"{'n_upd':>8s} "
            f"{'init_norm':>12s} "
            f"{'delta_norm':>12s} "
            f"{'rel_update':>12s}"
        )
        print(header)
        print("-" * len(header))
        for label in labels:
            stats = reports[label]["components"].get(cname)
            if stats is None:
                print(f"{label:<55s}  (component not tracked)")
                continue
            n_upd = f"{stats['n_tensors_updated']}/{stats['n_tensors']}"
            print(
                f"{label:<55s} "
                f"{n_upd:>8s} "
                f"{fmt_sci(stats['init_norm']):>12s} "
                f"{fmt_sci(stats['delta_norm']):>12s} "
                f"{fmt_sci(stats['relative_update']):>12s}"
            )

    # ---- Cross-run ratios ----
    if len(labels) > 1:
        baseline_label = args.baseline or labels[0]
        if baseline_label not in reports:
            print(f"\nError: --baseline '{baseline_label}' not among runs: {labels}")
            return 1

        print(f"\n=== Ratios relative to baseline run '{baseline_label}' ===")
        print(
            "A ratio << 1 means the component moved much LESS in that run than\n"
            "in the baseline — evidence the gradient signal was absorbed elsewhere."
        )

        for cname in args.components:
            base_stats = reports[baseline_label]["components"].get(cname)
            if base_stats is None:
                continue
            print(f"\n{cname}:")
            print(
                f"  baseline ({baseline_label}):  "
                f"delta_norm={fmt_sci(base_stats['delta_norm'])}  "
                f"rel_update={fmt_sci(base_stats['relative_update'])}"
            )
            for label in labels:
                if label == baseline_label:
                    continue
                stats = reports[label]["components"].get(cname)
                if stats is None:
                    continue
                delta_ratio = stats["delta_norm"] / max(base_stats["delta_norm"], 1e-30)
                rel_ratio = stats["relative_update"] / max(
                    base_stats["relative_update"], 1e-30
                )
                print(
                    f"  {label:<50s} "
                    f"delta_ratio={delta_ratio:7.3f}  rel_ratio={rel_ratio:7.3f}"
                )

    # ---- Within-run cross-component ratios (useful for single-checkpoint runs) ----
    if len(args.components) > 1:
        print("\n=== Within-run component ratios (projector vs language_model) ===")
        print(
            "If projector moved much less than LM in the same run, the LM\n"
            "is absorbing the gradient signal."
        )
        for label in labels:
            comps = reports[label]["components"]
            if "multi_modal_projector" not in comps or "language_model" not in comps:
                continue
            proj_rel = comps["multi_modal_projector"]["relative_update"]
            lm_rel = comps["language_model"]["relative_update"]
            if lm_rel > 1e-30:
                ratio = proj_rel / lm_rel
                print(
                    f"  {label:<50s} "
                    f"proj_rel/lm_rel = {ratio:.4f} "
                    f"(proj={fmt_sci(proj_rel)}, lm={fmt_sci(lm_rel)})"
                )

    # ---- Per-layer LM attention analysis ----
    if args.per_layer:
        print_per_layer_analysis(
            reports, labels, args.per_layer_projs, args.per_layer_top_k
        )

    # ---- Per-head LM attention analysis (top-drifting layers only) ----
    if args.per_head and per_head_data:
        # Only consider attention projections for per-head analysis. MLP
        # projections appear in args.per_layer_projs for per-layer / Gini
        # analysis but don't have a meaningful per-head structure.
        attn_projs_for_head = [
            p for p in args.per_layer_projs
            if p in {"q_proj", "k_proj", "v_proj", "o_proj"}
        ]

        # Pick the layers to analyse per-head: top-K from the first run with
        # non-zero drift for each projection. Using a single selection across
        # runs makes the cross-run comparison apples-to-apples.
        target_layers_by_proj: dict[str, list[int]] = {}
        for proj in attn_projs_for_head:
            for label in labels:
                drifts = extract_attention_proj_drifts(reports[label], [proj])[proj]
                # Skip runs with all-zero drift (component frozen)
                if not drifts or max(drifts.values()) == 0:
                    continue
                top = sorted(drifts.items(), key=lambda kv: kv[1], reverse=True)[
                    : args.per_head_layers_k
                ]
                target_layers_by_proj[proj] = sorted(layer for layer, _ in top)
                logger.info(
                    "per-head: selected layers for %s from run '%s': %s",
                    proj, label, target_layers_by_proj[proj],
                )
                break
            else:
                logger.warning(
                    "per-head: no run has non-zero %s drift; skipping this proj",
                    proj,
                )

        # Filter per_head_data down to the target layers
        filtered: dict[str, dict[tuple[int, str], list[float]]] = {}
        for label, cells in per_head_data.items():
            filtered[label] = {
                (layer, proj): vals
                for (layer, proj), vals in cells.items()
                if layer in target_layers_by_proj.get(proj, [])
            }

        print_per_head_analysis(
            filtered, labels, attn_projs_for_head, args.per_head_top_k, args.num_heads
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())