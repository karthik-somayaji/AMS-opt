#!/usr/bin/env python3
"""Plot training/validation losses from TensorBoard event files.

Usage:
  python3 scripts/plot_tb_losses.py --logdir logs_full_training --out umap_results_full/loss_curves.png

Notes:
- Uses TensorBoard's event processing utilities.
- Plots both raw epoch losses and EMA-smoothed losses if available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def _load_scalars(logdir: Path, tags: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(
        str(logdir),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.TENSORS: 0,
        },
    )
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    out: Dict[str, List[Tuple[int, float]]] = {}

    for tag in tags:
        if tag not in available:
            continue
        out[tag] = [(e.step, float(e.value)) for e in ea.Scalars(tag)]

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, required=True, help="TensorBoard logdir (contains events.out.tfevents.*)")
    ap.add_argument("--out", type=Path, required=True, help="Output image path (.png or .pdf)")
    ap.add_argument("--title", type=str, default="Training curves", help="Plot title")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    tags = [
        "loss/train_epoch",
        "loss/train_nt_xent_epoch",
        "loss/train_consistency_epoch",
        "loss_ema/train_epoch",
        "loss_ema/train_nt_xent_epoch",
        "loss_ema/train_consistency_epoch",
    ]

    scalars = _load_scalars(args.logdir, tags)
    if not scalars:
        raise SystemExit(f"No scalar tags found in {args.logdir}. Run training first.")

    def plot_tag(ax, tag: str, label: str, style: str = "-"):
        if tag not in scalars:
            return
        steps, vals = zip(*scalars[tag])
        ax.plot(steps, vals, style, linewidth=2, label=label)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    plot_tag(ax, "loss/train_epoch", "train total", "-")
    plot_tag(ax, "loss/train_nt_xent_epoch", "train nt_xent", "--")
    plot_tag(ax, "loss/train_consistency_epoch", "train consistency", ":")

    # If EMA tags exist, overlay them as thicker/darker versions
    # plot_tag(ax, "loss_ema/train_epoch", "EMA train total", "-")
    # plot_tag(ax, "loss_ema/train_nt_xent_epoch", "EMA train nt_xent", "--")
    # plot_tag(ax, "loss_ema/train_consistency_epoch", "EMA train consistency", ":")

    ax.set_title(args.title)
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
