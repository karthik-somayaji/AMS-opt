# # # from __future__ import annotations

# # # import json
# # # from pathlib import Path

# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import seaborn as sns

# # # try:
# # #     import umap
# # # except ImportError as exc:  # pragma: no cover
# # #     raise SystemExit("umap-learn is required to generate the SG family UMAP plot.") from exc


# # # INPUT_JSON = Path("umap_results_full/gnn_embeddings_sg.json")
# # # OUTPUT_STEM = Path("results/paper_plots/sg_family_umap_paper")
# # # DISPLAY_NAMES = {
# # #     "LDO": "LDO",
# # #     "comparators": "Comparators",
# # #     "diff_amps": "Diff. amps",
# # #     "op-amp": "Op-amp",
# # # }


# # # def _load_embeddings(path: Path) -> tuple[np.ndarray, list[dict]]:
# # #     with path.open() as handle:
# # #         payload = json.load(handle)
# # #     embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
# # #     metadata = list(payload["metadata"])
# # #     return embeddings, metadata


# # # def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
# # #     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
# # #     norms = np.maximum(norms, 1e-12)
# # #     return embeddings / norms


# # # def _compute_umap(embeddings: np.ndarray) -> np.ndarray:
# # #     reducer = umap.UMAP(
# # #         n_neighbors=10,
# # #         min_dist=0.55,
# # #         spread=2.2,
# # #         repulsion_strength=1.35,
# # #         negative_sample_rate=12,
# # #         metric="cosine",
# # #         random_state=42,
# # #     )
# # #     return reducer.fit_transform(embeddings)


# # # def _family_palette(families: list[str]) -> dict[str, tuple[float, float, float]]:
# # #     ordered = sorted(set(families))
# # #     palette = sns.color_palette("colorblind", len(ordered))
# # #     return {family: palette[idx] for idx, family in enumerate(ordered)}


# # # def _style_plot() -> None:
# # #     sns.set_theme(
# # #         context="paper",
# # #         style="white",
# # #         font="DejaVu Sans",
# # #         rc={
# # #             "figure.dpi": 180,
# # #             "savefig.dpi": 300,
# # #             "axes.spines.top": False,
# # #             "axes.spines.right": False,
# # #             "axes.grid": False,
# # #             "axes.labelsize": 9,
# # #             "axes.labelweight": "semibold",
# # #             "axes.linewidth": 0.8,
# # #             "xtick.labelsize": 8,
# # #             "ytick.labelsize": 8,
# # #             "legend.fontsize": 8,
# # #         },
# # #     )


# # # def _plot(umap_xy: np.ndarray, metadata: list[dict], out_stem: Path) -> None:
# # #     families = [str(item["family"]) for item in metadata]
# # #     palette = _family_palette(families)

# # #     fig, ax = plt.subplots(figsize=(3.45, 2.7), constrained_layout=True)

# # #     for family in sorted(set(families)):
# # #         mask = np.array([value == family for value in families])
# # #         points = umap_xy[mask]
# # #         ax.scatter(
# # #             points[:, 0],
# # #             points[:, 1],
# # #             s=22,
# # #             alpha=0.88,
# # #             c=[palette[family]],
# # #             edgecolors="white",
# # #             linewidths=0.45,
# # #             label=DISPLAY_NAMES.get(family, family),
# # #         )

# # #         centroid = np.median(points, axis=0)
# # #         ax.text(
# # #             centroid[0],
# # #             centroid[1],
# # #             DISPLAY_NAMES.get(family, family),
# # #             fontsize=6.4,
# # #             weight="semibold",
# # #             ha="center",
# # #             va="center",
# # #             bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.72},
# # #         )

# # #     ax.set_xlabel("UMAP-1")
# # #     ax.set_ylabel("UMAP-2")
# # #     ax.legend(
# # #         loc="upper center",
# # #         bbox_to_anchor=(0.5, 1.01),
# # #         ncol=2,
# # #         frameon=False,
# # #         title=None,
# # #         handletextpad=0.35,
# # #         columnspacing=0.9,
# # #         borderaxespad=0.15,
# # #     )
# # #     ax.tick_params(length=2.5, width=0.7, pad=1.5)
# # #     ax.margins(0.05)
# # #     sns.despine(ax=ax)

# # #     fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
# # #     fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
# # #     plt.close(fig)


# # # def main() -> None:
# # #     _style_plot()
# # #     embeddings, metadata = _load_embeddings(INPUT_JSON)
# # #     normalized_embeddings = _l2_normalize(embeddings)
# # #     umap_xy = _compute_umap(normalized_embeddings)
# # #     OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
# # #     _plot(umap_xy, metadata, OUTPUT_STEM)
# # #     print(f"Saved SG family UMAP plot to {OUTPUT_STEM.parent.resolve()}")


# # # if __name__ == "__main__":
# # #     main()


# # from __future__ import annotations

# # import json
# # from pathlib import Path

# # import matplotlib.pyplot as plt
# # from matplotlib.lines import Line2D
# # import numpy as np
# # import seaborn as sns

# # try:
# #     import umap
# # except ImportError as exc:  # pragma: no cover
# #     raise SystemExit("umap-learn is required to generate the SG family UMAP plot.") from exc


# # INPUT_JSON = Path("umap_results_full/gnn_embeddings_sg.json")
# # OUTPUT_STEM = Path("results/paper_plots/sg_family_umap_paper")

# # DISPLAY_NAMES = {
# #     "LDO": "LDO",
# #     "comparators": "Comparators",
# #     "diff_amps": "Diff. amps",
# #     "op-amp": "Op-amp",
# # }

# # # Explicit plotting / legend order.
# # FAMILY_ORDER = ["diff_amps", "comparators", "op-amp", "LDO"]


# # def _load_embeddings(path: Path) -> tuple[np.ndarray, list[dict]]:
# #     with path.open() as handle:
# #         payload = json.load(handle)
# #     embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
# #     metadata = list(payload["metadata"])
# #     return embeddings, metadata


# # def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
# #     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
# #     norms = np.maximum(norms, 1e-12)
# #     return embeddings / norms


# # def _compute_umap(embeddings: np.ndarray) -> np.ndarray:
# #     reducer = umap.UMAP(
# #         n_neighbors=10,
# #         min_dist=0.55,
# #         spread=2.2,
# #         repulsion_strength=1.35,
# #         negative_sample_rate=12,
# #         metric="cosine",
# #         random_state=42,
# #     )
# #     return reducer.fit_transform(embeddings)


# # def _family_palette(families: list[str]) -> dict[str, tuple[float, float, float]]:
# #     present_families = set(families)
# #     ordered = [fam for fam in FAMILY_ORDER if fam in present_families]

# #     # Add any unexpected families at the end to stay robust.
# #     extras = sorted(present_families - set(ordered))
# #     ordered.extend(extras)

# #     palette = sns.color_palette("deep", n_colors=len(ordered))
# #     return {family: palette[idx] for idx, family in enumerate(ordered)}


# # def _style_plot() -> None:
# #     sns.set_theme(
# #         context="paper",
# #         style="ticks",
# #         palette="deep",
# #         font="DejaVu Sans",
# #         rc={
# #             "figure.dpi": 180,
# #             "savefig.dpi": 300,
# #             "axes.spines.top": False,
# #             "axes.spines.right": False,
# #             "axes.grid": False,
# #             #"axes.facecolor": "#FCFCFC",
# #             #"figure.facecolor": #"white",
# #             "axes.facecolor": "#F2F3F5",   # light gray background
# #             "axes.labelsize": 9,
# #             "axes.labelweight": "semibold",
# #             "axes.linewidth": 0.8,
# #             "xtick.labelsize": 8,
# #             "ytick.labelsize": 8,
# #             "xtick.major.width": 0.7,
# #             "ytick.major.width": 0.7,
# #             "xtick.major.size": 2.5,
# #             "ytick.major.size": 2.5,
# #             "legend.fontsize": 7.8,
# #         },
# #     )


# # def _make_legend_handles(
# #     palette: dict[str, tuple[float, float, float]],
# #     families: list[str],
# # ) -> list[Line2D]:
# #     present_families = set(families)
# #     ordered = [fam for fam in FAMILY_ORDER if fam in present_families]
# #     ordered.extend(sorted(present_families - set(ordered)))

# #     handles = []
# #     for family in ordered:
# #         handles.append(
# #             Line2D(
# #                 [0],
# #                 [0],
# #                 marker="o",
# #                 linestyle="",
# #                 markersize=5.5,
# #                 markerfacecolor=palette[family],
# #                 markeredgecolor="white",
# #                 markeredgewidth=0.6,
# #                 label=DISPLAY_NAMES.get(family, family),
# #             )
# #         )
# #     return handles


# # def _plot(umap_xy: np.ndarray, metadata: list[dict], out_stem: Path) -> None:
# #     families = [str(item["family"]) for item in metadata]
# #     palette = _family_palette(families)

# #     present_families = set(families)
# #     ordered = [fam for fam in FAMILY_ORDER if fam in present_families]
# #     ordered.extend(sorted(present_families - set(ordered)))

# #     fig, ax = plt.subplots(figsize=(3.8, 2.7), constrained_layout=True)

# #     for family in ordered:
# #         mask = np.array([value == family for value in families])
# #         points = umap_xy[mask]
# #         if len(points) == 0:
# #             continue

# #         ax.scatter(
# #             points[:, 0],
# #             points[:, 1],
# #             s=24,
# #             alpha=0.9,
# #             color=palette[family],
# #             edgecolors="white",
# #             linewidths=0.5,
# #             zorder=2,
# #         )

# #         centroid = np.median(points, axis=0)
# #         ax.text(
# #             centroid[0],
# #             centroid[1],
# #             DISPLAY_NAMES.get(family, family),
# #             fontsize=6.5,
# #             fontweight="semibold",
# #             ha="center",
# #             va="center",
# #             zorder=3,
# #             bbox={
# #                 "boxstyle": "round,pad=0.2",
# #                 "fc": "white",
# #                 "ec": "none",
# #                 "alpha": 0.78,
# #             },
# #         )

# #     ax.set_xlabel("UMAP-1")
# #     ax.set_ylabel("UMAP-2")

# #     legend_handles = _make_legend_handles(palette, families)
# #     # ax.legend(
# #     #     handles=legend_handles,
# #     #     loc="upper center",
# #     #     bbox_to_anchor=(0.5, 1.02),
# #     #     ncol=1, #2,
# #     #     frameon=False,
# #     #     handletextpad=0.4,
# #     #     columnspacing=1.0,
# #     #     borderaxespad=0.2,
# #     # )
# #     ax.legend(
# #         handles=legend_handles,
# #         loc="upper left",
# #         bbox_to_anchor=(1.01, 1.0),
# #         ncol=1,                 # vertical legend
# #         frameon=True,
# #         fancybox=True,
# #         framealpha=0.95,
# #         edgecolor="#D0D4D8",
# #         facecolor="white",
# #         handletextpad=0.45,
# #         columnspacing=0.8,
# #         borderpad=0.35,
# #         labelspacing=0.45,
# #     )

# #     ax.tick_params(pad=1.5)
# #     ax.margins(0.06)
# #     sns.despine(ax=ax, offset=2, trim=False)

# #     fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
# #     fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
# #     plt.close(fig)


# # def main() -> None:
# #     _style_plot()
# #     embeddings, metadata = _load_embeddings(INPUT_JSON)
# #     normalized_embeddings = _l2_normalize(embeddings)
# #     umap_xy = _compute_umap(normalized_embeddings)
# #     OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
# #     _plot(umap_xy, metadata, OUTPUT_STEM)
# #     print(f"Saved SG family UMAP plot to {OUTPUT_STEM.parent.resolve()}")


# # if __name__ == "__main__":
# #     main()

# from __future__ import annotations

# import json
# from pathlib import Path

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import numpy as np
# import seaborn as sns

# try:
#     import umap
# except ImportError as exc:  # pragma: no cover
#     raise SystemExit("umap-learn is required to generate the SG family UMAP plot.") from exc


# INPUT_JSON = Path("umap_results_full/gnn_embeddings_sg.json")
# OUTPUT_STEM = Path("results/paper_plots/sg_family_umap_paper")

# DISPLAY_NAMES = {
#     "LDO": "LDO",
#     "comparators": "Comparators",
#     "diff_amps": "Diff. amps",
#     "op-amp": "Op-amp",
# }

# # Explicit plotting / legend order.
# FAMILY_ORDER = ["diff_amps", "comparators", "op-amp", "LDO"]


# def _load_embeddings(path: Path) -> tuple[np.ndarray, list[dict]]:
#     with path.open() as handle:
#         payload = json.load(handle)
#     embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
#     metadata = list(payload["metadata"])
#     return embeddings, metadata


# def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     norms = np.maximum(norms, 1e-12)
#     return embeddings / norms


# def _compute_umap(embeddings: np.ndarray) -> np.ndarray:
#     reducer = umap.UMAP(
#         n_neighbors=10,
#         min_dist=0.55,
#         spread=2.2,
#         repulsion_strength=1.35,
#         negative_sample_rate=12,
#         metric="cosine",
#         random_state=42,
#     )
#     return reducer.fit_transform(embeddings)


# def _family_palette(families: list[str]) -> dict[str, tuple[float, float, float]]:
#     present_families = set(families)
#     ordered = [fam for fam in FAMILY_ORDER if fam in present_families]

#     # Add any unexpected families at the end to stay robust.
#     extras = sorted(present_families - set(ordered))
#     ordered.extend(extras)

#     palette = sns.color_palette("deep", n_colors=len(ordered))
#     return {family: palette[idx] for idx, family in enumerate(ordered)}


# def _style_plot() -> None:
#     sns.set_theme(
#         context="paper",
#         style="ticks",
#         palette="deep",
#         font="DejaVu Sans",
#         rc={
#             "figure.dpi": 180,
#             "savefig.dpi": 300,
#             "axes.spines.top": False,
#             "axes.spines.right": False,
#             "axes.grid": False,
#             "axes.facecolor": "#F2F3F5",  # light gray background
#             "axes.labelsize": 9,
#             "axes.labelweight": "semibold",
#             "axes.linewidth": 0.8,
#             "xtick.labelsize": 8,
#             "ytick.labelsize": 8,
#             "xtick.major.width": 0.7,
#             "ytick.major.width": 0.7,
#             "xtick.major.size": 2.5,
#             "ytick.major.size": 2.5,
#             "legend.fontsize": 6.4,  # smaller legend text
#         },
#     )


# def _make_legend_handles(
#     palette: dict[str, tuple[float, float, float]],
#     families: list[str],
# ) -> list[Line2D]:
#     present_families = set(families)
#     ordered = [fam for fam in FAMILY_ORDER if fam in present_families]
#     ordered.extend(sorted(present_families - set(ordered)))

#     handles = []
#     for family in ordered:
#         handles.append(
#             Line2D(
#                 [0],
#                 [0],
#                 marker="o",
#                 linestyle="",
#                 markersize=4.0,  # smaller marker in legend
#                 markerfacecolor=palette[family],
#                 markeredgecolor="white",
#                 markeredgewidth=0.5,
#                 label=DISPLAY_NAMES.get(family, family),
#             )
#         )
#     return handles


# def _plot(umap_xy: np.ndarray, metadata: list[dict], out_stem: Path) -> None:
#     families = [str(item["family"]) for item in metadata]
#     palette = _family_palette(families)

#     present_families = set(families)
#     ordered = [fam for fam in FAMILY_ORDER if fam in present_families]
#     ordered.extend(sorted(present_families - set(ordered)))

#     fig, ax = plt.subplots(figsize=(3.8, 2.7), constrained_layout=True)

#     for family in ordered:
#         mask = np.array([value == family for value in families])
#         points = umap_xy[mask]
#         if len(points) == 0:
#             continue

#         ax.scatter(
#             points[:, 0],
#             points[:, 1],
#             s=24,
#             alpha=0.9,
#             color=palette[family],
#             edgecolors="white",
#             linewidths=0.5,
#             zorder=2,
#         )

#         centroid = np.median(points, axis=0)
#         ax.text(
#             centroid[0],
#             centroid[1],
#             DISPLAY_NAMES.get(family, family),
#             fontsize=6.5,
#             fontweight="semibold",
#             ha="center",
#             va="center",
#             zorder=3,
#             bbox={
#                 "boxstyle": "round,pad=0.2",
#                 "fc": "white",
#                 "ec": "none",
#                 "alpha": 0.78,
#             },
#         )

#     ax.set_xlabel("UMAP-1")
#     ax.set_ylabel("UMAP-2")

#     legend_handles = _make_legend_handles(palette, families)
#     ax.legend(
#         handles=legend_handles,
#         loc="upper left",
#         bbox_to_anchor=(1.01, 1.0),
#         ncol=1,
#         frameon=True,
#         fancybox=True,
#         framealpha=0.95,
#         edgecolor="#D0D4D8",
#         facecolor="white",
#         fontsize=6.2,
#         handlelength=0.8,
#         handletextpad=0.3,
#         columnspacing=0.6,
#         borderpad=0.22,
#         labelspacing=0.25,
#         borderaxespad=0.15,
#     )

#     ax.tick_params(pad=1.5)
#     ax.margins(0.06)
#     sns.despine(ax=ax, offset=2, trim=False)

#     fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
#     fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
#     plt.close(fig)


# def main() -> None:
#     _style_plot()
#     embeddings, metadata = _load_embeddings(INPUT_JSON)
#     normalized_embeddings = _l2_normalize(embeddings)
#     umap_xy = _compute_umap(normalized_embeddings)
#     OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
#     _plot(umap_xy, metadata, OUTPUT_STEM)
#     print(f"Saved SG family UMAP plot to {OUTPUT_STEM.parent.resolve()}")


# if __name__ == "__main__":
#     main()

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("umap-learn is required to generate the SG family UMAP plot.") from exc


INPUT_JSON = Path("umap_results_full/gnn_embeddings_sg.json")
OUTPUT_STEM = Path("results/paper_plots/sg_family_umap_paper")

DISPLAY_NAMES = {
    "LDO": "LDO",
    "comparators": "Comparators",
    "diff_amps": "Diff. amps",
    "op-amp": "Op-amp",
}

FAMILY_ORDER = ["diff_amps", "comparators", "op-amp", "LDO"]


def _load_embeddings(path: Path) -> tuple[np.ndarray, list[dict]]:
    with path.open() as handle:
        payload = json.load(handle)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    metadata = list(payload["metadata"])
    return embeddings, metadata


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def _compute_umap(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.55,
        spread=2.2,
        repulsion_strength=1.35,
        negative_sample_rate=12,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def _family_palette(families: list[str]) -> dict[str, tuple[float, float, float]]:
    present_families = set(families)
    ordered = [fam for fam in FAMILY_ORDER if fam in present_families]

    extras = sorted(present_families - set(ordered))
    ordered.extend(extras)

    palette = sns.color_palette("deep", n_colors=len(ordered))
    return {family: palette[idx] for idx, family in enumerate(ordered)}


def _style_plot() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="deep",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.facecolor": "#F2F3F5",
            "figure.facecolor": "white",
            "axes.labelsize": 9,
            "axes.labelweight": "semibold",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6.2,
        },
    )


def _make_legend_handles(
    palette: dict[str, tuple[float, float, float]],
    families: list[str],
) -> list[Line2D]:
    present_families = set(families)
    ordered = [fam for fam in FAMILY_ORDER if fam in present_families]
    ordered.extend(sorted(present_families - set(ordered)))

    handles = []
    for family in ordered:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=4.0,
                markerfacecolor=palette[family],
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=DISPLAY_NAMES.get(family, family),
            )
        )
    return handles


def _plot(umap_xy: np.ndarray, metadata: list[dict], out_stem: Path) -> None:
    families = [str(item["family"]) for item in metadata]
    palette = _family_palette(families)

    present_families = set(families)
    ordered = [fam for fam in FAMILY_ORDER if fam in present_families]
    ordered.extend(sorted(present_families - set(ordered)))

    fig, ax = plt.subplots(figsize=(3.8, 2.7), constrained_layout=True)

    for family in ordered:
        mask = np.array([value == family for value in families])
        points = umap_xy[mask]
        if len(points) == 0:
            continue

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=24,
            alpha=0.9,
            color=palette[family],
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

        centroid = np.median(points, axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            DISPLAY_NAMES.get(family, family),
            fontsize=6.5,
            fontweight="semibold",
            ha="center",
            va="center",
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.78,
            },
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    legend_handles = _make_legend_handles(palette, families)
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        frameon=False,      # no white legend box
        handlelength=0.8,
        handletextpad=0.3,
        borderpad=0.2,
        labelspacing=0.25,
        borderaxespad=0.2,
    )

    ax.tick_params(pad=1.5)
    ax.margins(0.06)
    sns.despine(ax=ax, offset=2, trim=False)

    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style_plot()
    embeddings, metadata = _load_embeddings(INPUT_JSON)
    normalized_embeddings = _l2_normalize(embeddings)
    umap_xy = _compute_umap(normalized_embeddings)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    _plot(umap_xy, metadata, OUTPUT_STEM)
    print(f"Saved SG family UMAP plot to {OUTPUT_STEM.parent.resolve()}")


if __name__ == "__main__":
    main()