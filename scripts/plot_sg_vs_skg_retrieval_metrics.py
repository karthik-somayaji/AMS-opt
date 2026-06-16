from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CIRCUITS = ["Amp", "FC", "Comp", "LDO"]

# Extracted from the user-provided SG retrieval table.
SG_BOTTOMK_FOM = np.array([0.860, 0.800, 0.848, 0.852])
SG_TOPK_PASSK = np.array([15, 12, 5, 8])

# Extracted from the user-provided SKG retrieval table.
SKG_BOTTOMK_FOM = np.array([0.862, 0.793, 0.750, 0.861])
SKG_TOPK_PASSK = np.array([2, 6, 9, 4])

BAR_COLORS = sns.color_palette("crest", 6)
SG_COLOR = BAR_COLORS[1]
SKG_COLOR = BAR_COLORS[4]


def _style_matplotlib() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
        },
    )


def _annotate_bars(ax, bars, fmt: str, x_text_offset: float = 0.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(x_text_offset, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_grouped_bars(
    *,
    labels,
    left_values,
    right_values,
    left_name,
    right_name,
    ylabel,
    out_path,
    value_fmt,
    ylim=None,
) -> None:
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    left_bars = ax.bar(
        x - width / 2,
        left_values,
        width,
        label=left_name,
        color=SG_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )
    right_bars = ax.bar(
        x + width / 2,
        right_values,
        width,
        label=right_name,
        color=SKG_COLOR,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=6)

    if ylim is not None:
        ax.set_ylim(*ylim)

    _annotate_bars(ax, left_bars, value_fmt, x_text_offset=-10)
    _annotate_bars(ax, right_bars, value_fmt, x_text_offset=10)

    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_combined_figure(out_path) -> None:
    x = np.arange(len(CIRCUITS))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), constrained_layout=True)

    panels = [
        (
            axes[0],
            SG_BOTTOMK_FOM,
            SKG_BOTTOMK_FOM,
            "FOM",
            "Bottom-k Retrieval",
            "{:.3f}",
            (0, 1.05),
        ),
        (
            axes[1],
            SG_TOPK_PASSK,
            SKG_TOPK_PASSK,
            "Pass@k",
            "Top-k Retrieval",
            "{:.0f}",
            (0, max(np.max(SG_TOPK_PASSK), np.max(SKG_TOPK_PASSK)) + 3),
        ),
    ]

    for ax, left_values, right_values, ylabel, panel_title, value_fmt, ylim in panels:
        left_bars = ax.bar(
            x - width / 2,
            left_values,
            width,
            label="SG retrieval",
            color=SG_COLOR,
            edgecolor="black",
            linewidth=0.6,
        )
        right_bars = ax.bar(
            x + width / 2,
            right_values,
            width,
            label="SKG retrieval",
            color=SKG_COLOR,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(CIRCUITS)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, pad=6)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", pad=4)
        ax.tick_params(axis="y", pad=4)
        # if ylabel == "FOM":
        #     _annotate_bars(ax, left_bars, value_fmt, x_text_offset=-14)
        #     _annotate_bars(ax, right_bars, value_fmt, x_text_offset=14)
        # else:
        #     _annotate_bars(ax, left_bars, value_fmt, x_text_offset=-8)
        #     _annotate_bars(ax, right_bars, value_fmt, x_text_offset=8)

    axes[1].legend(
        [axes[1].containers[0][0], axes[1].containers[1][0]],
        ["SG retrieval", "SKG retrieval"],
        loc="upper right",
        frameon=False,
        fontsize=9,
        borderaxespad=0.2,
    )

    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    _style_matplotlib()

    out_dir = Path("results/paper_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_grouped_bars(
        labels=CIRCUITS,
        left_values=SG_BOTTOMK_FOM,
        right_values=SKG_BOTTOMK_FOM,
        left_name="SG retrieval",
        right_name="SKG retrieval",
        ylabel="FOM",
        out_path=out_dir / "bottomk_fom_sg_vs_skg.png",
        value_fmt="{:.3f}",
        ylim=(0, 1.05),
    )

    _plot_grouped_bars(
        labels=CIRCUITS,
        left_values=SG_TOPK_PASSK,
        right_values=SKG_TOPK_PASSK,
        left_name="SG retrieval",
        right_name="SKG retrieval",
        ylabel="Pass@k",
        out_path=out_dir / "topk_passk_sg_vs_skg.png",
        value_fmt="{:.0f}",
        ylim=(0, max(np.max(SG_TOPK_PASSK), np.max(SKG_TOPK_PASSK)) + 3),
    )

    _plot_combined_figure(out_dir / "sg_vs_skg_retrieval_combined.png")

    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()