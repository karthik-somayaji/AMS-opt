#!/usr/bin/env python3
"""Plot the *terminal-reported* (log file) epoch losses.

This parses the training log produced by `gnn_training.scripts.train` (files like
`logs_full_training/train_YYYYMMDD_HHMMSS.log`) and extracts exactly the same
numbers you see in the terminal, e.g.:

  Epoch 2/20 - Loss: 0.4014 (NT-Xent: 0.3768, Consistency: 0.6226)

It then plots ONE point per epoch for:
- total train loss
- NT-Xent loss
- consistency loss

Usage:
  python3 scripts/plot_train_log_losses.py \
    --log logs_full_training/train_20260114_090741.log \
    --out umap_results_full/train_log_losses.png

Or automatically pick latest log:
  python3 scripts/plot_train_log_losses.py \
    --logdir logs_full_training \
    --out umap_results_full/train_log_losses.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+-\s+Loss:\s+(?P<loss>[0-9]*\.?[0-9]+)\s+\(NT-Xent:\s+(?P<nt>[0-9]*\.?[0-9]+),\s+Consistency:\s+(?P<cons>[0-9]*\.?[0-9]+)\)"
)
ALT_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+-\s+Train Loss:\s+(?P<loss>[0-9]*\.?[0-9]+)"
)


@dataclass(frozen=True)
class EpochPoint:
    epoch: int
    loss: float
    nt_xent: Optional[float]
    consistency: Optional[float]


def _latest_train_log(logdir: Path) -> Path:
    candidates = sorted(logdir.glob("train_*.log"))
    if not candidates:
        raise FileNotFoundError(f"No train_*.log files found in {logdir}")
    return candidates[-1]


def _parse_log(path: Path) -> List[EpochPoint]:
    points: List[EpochPoint] = []
    seen_epochs = set()

    for line in path.read_text(errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if m:
            epoch = int(m.group("epoch"))
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)
            points.append(
                EpochPoint(
                    epoch=epoch,
                    loss=float(m.group("loss")),
                    nt_xent=float(m.group("nt")),
                    consistency=float(m.group("cons")),
                )
            )
            continue

        m2 = ALT_RE.search(line)
        if m2:
            epoch = int(m2.group("epoch"))
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)
            points.append(
                EpochPoint(
                    epoch=epoch,
                    loss=float(m2.group("loss")),
                    nt_xent=None,
                    consistency=None,
                )
            )

    points.sort(key=lambda p: p.epoch)
    if not points:
        raise ValueError(
            f"No epoch loss lines found in {path}. Expected lines like 'Epoch 2/20 - Loss: ... (NT-Xent: ..., Consistency: ...)'"
        )
    return points


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=None, help="Path to a specific train_*.log")
    ap.add_argument("--logdir", type=Path, default=None, help="Directory containing train_*.log (auto-picks latest)")
    ap.add_argument("--out", type=Path, required=True, help="Output image path (.png or .pdf)")
    ap.add_argument("--title", type=str, default="Train losses (from log)")
    args = ap.parse_args()

    if args.log is None:
        if args.logdir is None:
            raise SystemExit("Provide either --log or --logdir")
        log_path = _latest_train_log(args.logdir)
    else:
        log_path = args.log

    points = _parse_log(log_path)

    import matplotlib.pyplot as plt

    epochs = [p.epoch for p in points]
    losses = [p.loss for p in points]
    nts = [p.nt_xent for p in points]
    cons = [p.consistency for p in points]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, losses, "-", linewidth=2.5, label="train total")

    if all(v is not None for v in nts):
        ax.plot(epochs, [float(v) for v in nts], "--", linewidth=2.0, label="nt_xent")

    if all(v is not None for v in cons):
        ax.plot(epochs, [float(v) for v in cons], ":", linewidth=2.0, label="consistency")

    ax.set_title(f"{args.title}\n{log_path.name}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=3, fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=220)
    print(f"Parsed {log_path}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
