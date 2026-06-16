#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


INPUT_CSV = Path("umap_results_full/sg_skg_analysis/sg_skg_distances.csv")
OUTPUT_PNG = Path("results/paper_plots/sg_skg_cosine_distance_histogram.png")
OUTPUT_JSON = Path("results/analysis/sg_skg_cosine_distance_summary.json")
FIG_WIDTH = 3.45
FIG_HEIGHT = 2.25


def _style_matplotlib() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.12,
        },
    )


def load_rows(csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "source": row["source"],
                    "family": row["family"],
                    "circuit_id": row["circuit_id"],
                    "split": row["split"],
                    "cosine_dist": float(row["cosine_dist"]),
                }
            )
    return rows


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main() -> None:
    rows = load_rows(INPUT_CSV)
    if not rows:
        raise ValueError(f"No rows found in {INPUT_CSV}")

    netlist_rows = [row for row in rows if row["source"] == "netlists"]
    llmbo_rows = [row for row in rows if row["source"] == "llmbo"]

    all_values = np.array([row["cosine_dist"] for row in rows], dtype=float)
    netlist_values = np.array([row["cosine_dist"] for row in netlist_rows], dtype=float)
    llmbo_values = np.array([row["cosine_dist"] for row in llmbo_rows], dtype=float)

    summary = {
        "all": summarize(all_values),
        "netlists": summarize(netlist_values),
        "llmbo": summarize(llmbo_values),
        "llmbo_by_anchor": {
            str(row["circuit_id"]): float(row["cosine_dist"]) for row in llmbo_rows
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w") as handle:
        json.dump(summary, handle, indent=2)

    _style_matplotlib()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
    upper_bound = max(0.16, float(np.percentile(netlist_values, 97.5)) * 1.05)
    bins = np.linspace(0.0, upper_bound, 20)

    hist_color = sns.color_palette("crest", 6)[3]
    sns.histplot(
        netlist_values,
        bins=bins,
        stat="count",
        color=hist_color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.95,
        ax=ax,
    )
    sns.kdeplot(
        netlist_values,
        color=sns.color_palette("crest", 6)[5],
        linewidth=1.1,
        cut=0,
        clip=(0.0, upper_bound),
        ax=ax,
    )

    mean_value = summary["netlists"]["mean"]
    median_value = summary["netlists"]["median"]
    ax.axvline(mean_value, color="#b22222", linestyle="--", linewidth=1.0)
    ax.axvline(median_value, color="#222222", linestyle=":", linewidth=1.0)

    if llmbo_rows:
        ymax = ax.get_ylim()[1]
        marker_y = ymax * 0.08
        for row in llmbo_rows:
            value = float(row["cosine_dist"])
            label = str(row["circuit_id"])
            ax.scatter(
                [value],
                [marker_y],
                marker="D",
                s=18,
                color="#d62728",
                edgecolors="white",
                linewidths=0.4,
                zorder=5,
            )
            ax.annotate(
                label,
                (value, marker_y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#7f0000",
            )

    ax.set_xlim(0.0, upper_bound)
    ax.set_xlabel("Cosine distance (SKG vs SG)")
    ax.set_ylabel("Count")
    ax.set_title("SG-SKG embedding shift", pad=2)
    ax.text(
        0.98,
        0.95,
        f"mean={mean_value:.3f}\nmedian={median_value:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#d9d9d9", "alpha": 0.92},
    )
    sns.despine(ax=ax, offset=2)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG)
    fig.savefig(OUTPUT_PNG.with_suffix(".pdf"))
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved plot: {OUTPUT_PNG}")
    print(f"Saved summary: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()