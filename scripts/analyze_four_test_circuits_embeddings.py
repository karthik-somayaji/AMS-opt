#!/usr/bin/env python3
"""scripts/analyze_four_test_circuits_embeddings.py

Analyze four external (LLMBO) test circuits against the latest trained GNN.

What this script produces
- UMAP overlay PNG: train/test circuits (colored by family) + 4 test circuits highlighted.
- CSV of normalized distances from each test circuit to each family centroid.
- CSV/JSON of top-k nearest neighbors for each test circuit + family-mismatch counts.

Important implementation choices
- Embeddings are extracted from the trained GNN encoder via `model.encode(features, adjacency)`.
- For UMAP, we fit a reducer on the normalized reference embeddings (train+test) and then
  transform the 4 test circuits into the same 2D space. This makes the overlay comparable.
- Distances/nearest-neighbors are computed in normalized embedding space (cosine distance
  equivalents on the unit sphere, but implemented as Euclidean on normalized vectors).

Usage
  python3 scripts/analyze_four_test_circuits_embeddings.py \
    --checkpoint checkpoints_full_training/best_model.pt \
    --train-config checkpoints_full_training/config.yaml \
    --data-dir netlists \
    --llmbo-circuits \
      amp2:LLMBO/amp2_ati_new \
      FC:LLMBO/FC_ati_new \
      comp:LLMBO/comp_ati_new \
      ldo:LLMBO/ldo_ati_new \
    --out-dir umap_results_four

Notes
- This expects each LLMBO circuit dir to contain `comb_graph_gnn.npz` and
  `comb_graph_gnn_meta.json` (like your prompt stated).
"""

from __future__ import annotations

import argparse
import sys
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'numpy'. This script expects the `analog-rep` Python environment. "
        "Try running with `/home/karthik/miniconda3/envs/analog-rep/bin/python` or `conda activate analog-rep`. "
        f"Current interpreter: {sys.executable}"
    ) from e
try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'torch'. This script expects the `analog-rep` Python environment. "
        "Try running with `/home/karthik/miniconda3/envs/analog-rep/bin/python` or `conda activate analog-rep`. "
        f"Current interpreter: {sys.executable}"
    ) from e

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_training.data import CircuitDataLoader
from gnn_training.models import ContrastiveGINModel


def _lazy_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except ImportError:
        print("Installing pyyaml...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
        import yaml  # type: ignore

        return yaml


def _lazy_import_umap():
    try:
        import umap  # type: ignore

        return umap
    except ImportError:
        print("Installing umap-learn...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "-q"])
        import umap  # type: ignore

        return umap


def _lazy_import_plotting():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        return plt, mpatches
    except ImportError:
        print("Installing matplotlib...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        return plt, mpatches


def load_yaml(path: str) -> Dict:
    yaml = _lazy_import_yaml()
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_cfg: Optional[Dict] = None,
) -> Tuple[ContrastiveGINModel, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = model_cfg or {}

    input_dim = int(model_cfg.get("input_dim", 79))
    hidden_dims = list(model_cfg.get("hidden_dims", [64, 64, 32]))
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    projection_dim = int(model_cfg.get("projection_dim", 128))
    dropout = float(model_cfg.get("dropout", 0.1))
    use_batch_norm = bool(model_cfg.get("use_batch_norm", True))

    model = ContrastiveGINModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )

    # Expected key in this repo's checkpoints
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model, checkpoint


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


def euclidean_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise distances between rows of a and rows of b."""
    # (a-b)^2 = a^2 + b^2 - 2ab
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    d2 = np.maximum(a2 + b2 - 2.0 * ab, 0.0)
    return np.sqrt(d2)


@dataclass(frozen=True)
class LLMBOCircuitSpec:
    name: str
    path: str
    expected_family: str


def parse_llmbo_specs(specs: List[str]) -> List[LLMBOCircuitSpec]:
    """Parses `name:/path` items into specs with expected family mapping."""
    expected_family_by_name = {
        "amp2": "diff_amps",
        "FC": "diff_amps",
        "comp": "comparators",
        "ldo": "LDO",
    }

    parsed: List[LLMBOCircuitSpec] = []
    for item in specs:
        if ":" not in item:
            raise ValueError(f"Invalid --llmbo-circuits item: {item}. Expected name:path")
        name, path = item.split(":", 1)
        name = name.strip()
        path = path.strip()
        if name not in expected_family_by_name:
            raise ValueError(
                f"Unknown LLMBO circuit name '{name}'. "
                f"Expected one of {sorted(expected_family_by_name.keys())}"
            )
        parsed.append(
            LLMBOCircuitSpec(
                name=name,
                path=path,
                expected_family=expected_family_by_name[name],
            )
        )
    return parsed


def extract_embedding_for_npz_dir(
    model: ContrastiveGINModel, npz_dir: str, device: torch.device
) -> np.ndarray:
    """Loads comb_graph_gnn.npz from an arbitrary directory and encodes."""
    npz_path = os.path.join(npz_dir, "comb_graph_gnn.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    adjacency = data["adjacency"].astype(np.float32)

    # Validate feature dimensionality matches the trained model.
    expected_dim = int(getattr(model, "input_dim", features.shape[1]))
    if features.ndim != 2:
        raise ValueError(f"Expected features to be 2D (N,D); got shape {features.shape}")
    if int(features.shape[1]) != expected_dim:
        raise ValueError(
            "Feature dimension mismatch between this circuit and the trained model. "
            f"Circuit features have D={features.shape[1]}, but the model expects input_dim={expected_dim}. "
            "This usually means the LLMBO circuit was converted to GNN features with a different "
            "feature schema than the training netlists. Regenerate `comb_graph_gnn.npz` for the LLMBO "
            "circuits using the same pipeline/config that produced `netlists/*/*/comb_graph_gnn.npz`."
        )

    with torch.no_grad():
        feat_t = torch.tensor(features, dtype=torch.float32, device=device)
        adj_t = torch.tensor(adjacency, dtype=torch.float32, device=device)
        emb = model.encode(feat_t, adj_t).detach().cpu().numpy()

    if emb.ndim != 1:
        emb = emb.reshape(-1)
    return emb


def flatten_circuits(circuits_by_family: Dict) -> Tuple[List[str], List[str]]:
    circuit_ids: List[str] = []
    families: List[str] = []
    for family, ids in circuits_by_family.items():
        for cid in ids:
            circuit_ids.append(str(cid))
            families.append(str(family))
    return circuit_ids, families


def extract_reference_embeddings(
    model: ContrastiveGINModel,
    data_dir: str,
    circuits_by_family: Dict,
    device: torch.device,
) -> Tuple[np.ndarray, List[Dict]]:
    circuit_ids, families = flatten_circuits(circuits_by_family)

    embeddings: List[np.ndarray] = []
    metadata: List[Dict] = []
    with torch.no_grad():
        for cid, fam in zip(circuit_ids, families):
            loader = CircuitDataLoader(cid, f"{data_dir}/{fam}")
            graph = loader.get_graph()
            feat_t = torch.tensor(graph["features"], dtype=torch.float32, device=device)
            adj_t = torch.tensor(graph["adjacency"], dtype=torch.float32, device=device)
            emb = model.encode(feat_t, adj_t).detach().cpu().numpy().reshape(-1)
            embeddings.append(emb)
            metadata.append({"circuit_id": cid, "family": fam})

    return np.vstack(embeddings), metadata


def compute_family_centroids(embeddings: np.ndarray, metadata: List[Dict]) -> Dict[str, np.ndarray]:
    fam_to_rows: Dict[str, List[int]] = {}
    for i, m in enumerate(metadata):
        fam_to_rows.setdefault(m["family"], []).append(i)

    centroids: Dict[str, np.ndarray] = {}
    for fam, idxs in fam_to_rows.items():
        centroids[fam] = np.mean(embeddings[idxs, :], axis=0)
    return centroids


def normalize_distances_to_centroids(
    test_emb: np.ndarray,
    centroids: Dict[str, np.ndarray],
    normalize: str = "minmax",
) -> Dict[str, float]:
    fams = sorted(centroids.keys())
    centroid_mat = np.vstack([centroids[f] for f in fams])

    dists = euclidean_distances(test_emb[None, :], centroid_mat).reshape(-1)

    if normalize == "minmax":
        dmin = float(np.min(dists))
        dmax = float(np.max(dists))
        denom = (dmax - dmin) if (dmax - dmin) > 1e-12 else 1.0
        normed = (dists - dmin) / denom
    elif normalize == "zscore":
        mu = float(np.mean(dists))
        std = float(np.std(dists))
        denom = std if std > 1e-12 else 1.0
        normed = (dists - mu) / denom
    else:
        raise ValueError("normalize must be one of: minmax, zscore")

    return {fam: float(val) for fam, val in zip(fams, normed)}


def plot_umap_overlay(
    umap_ref: np.ndarray,
    ref_meta: List[Dict],
    umap_test: np.ndarray,
    test_specs: List[LLMBOCircuitSpec],
    output_path: str,
):
    plt, mpatches = _lazy_import_plotting()

    families = [m["family"] for m in ref_meta]
    unique_families = sorted(set(families))

    base_colors = {
        "diff_amps": "#1f77b4",
        "comparators": "#ff7f0e",
        "LDO": "#2ca02c",
        "ldo": "#2ca02c",
        "op-amp": "#d62728",
        "opamp": "#d62728",
    }

    palette = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    family_to_color: Dict[str, str] = {}
    palette_idx = 0
    for fam in unique_families:
        if fam in base_colors:
            family_to_color[fam] = base_colors[fam]
        else:
            family_to_color[fam] = palette[palette_idx % len(palette)]
            palette_idx += 1

    fig, ax = plt.subplots(figsize=(12, 8))

    # Reference points
    for fam in unique_families:
        mask = np.array([f == fam for f in families])
        ax.scatter(
            umap_ref[mask, 0],
            umap_ref[mask, 1],
            label=fam,
            s=60,
            alpha=0.55,
            color=family_to_color[fam],
            linewidth=0,
        )

    # Test points (highlight)
    marker_by_name = {"amp2": "*", "FC": "*", "comp": "X", "ldo": "D"}
    for i, spec in enumerate(test_specs):
        ax.scatter(
            [umap_test[i, 0]],
            [umap_test[i, 1]],
            s=260,
            color="black",
            marker=marker_by_name.get(spec.name, "*"),
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
        )
        ax.annotate(
            spec.name,
            (umap_test[i, 0], umap_test[i, 1]),
            fontsize=11,
            fontweight="bold",
            color="black",
            xytext=(6, 6),
            textcoords="offset points",
        )

    # Legend augment with test markers
    test_patches = [
        mpatches.Patch(color="black", label="LLMBO test circuits (highlighted)")
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + test_patches, title="Circuit Family", fontsize=10)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("GNN Embeddings UMAP: Train/Test families + 4 LLMBO circuits")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Analyze 4 external circuits against trained GNN embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--checkpoint",
        default="checkpoints_full_training/best_model.pt",
        help="Trained checkpoint (best_model.pt recommended).",
    )
    ap.add_argument(
        "--train-config",
        default="checkpoints_full_training/config.yaml",
        help="Training config that contains the circuits list (for reference embedding extraction).",
    )
    ap.add_argument(
        "--data-dir",
        default="netlists",
        help="Netlists root used for train/test circuits used during training.",
    )
    ap.add_argument(
        "--llmbo-circuits",
        nargs="+",
        required=True,
        help="List of name:path items. Expected names: amp2, FC, comp, ldo.",
    )
    ap.add_argument("--out-dir", default="umap_results_four", help="Output directory")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--umap-neighbors", type=int, default=10)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--distance-normalization",
        choices=["minmax", "zscore"],
        default="minmax",
        help="How to normalize per-test-circuit distances to family centroids.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_specs = parse_llmbo_specs(args.llmbo_circuits)

    device = torch.device(args.device)

    cfg = load_yaml(args.train_config)
    model_cfg = cfg.get("model", {})
    circuits_by_family = cfg.get("data", {}).get("circuits", {})

    print(f"Loading model: {args.checkpoint}")
    model, ckpt_info = load_checkpoint(args.checkpoint, device=device, model_cfg=model_cfg)
    if isinstance(ckpt_info, dict) and "epoch" in ckpt_info:
        print(f"✓ Loaded checkpoint epoch={ckpt_info['epoch']}")

    print("Extracting reference embeddings (train+test pool from config)...")
    ref_emb, ref_meta = extract_reference_embeddings(
        model=model, data_dir=args.data_dir, circuits_by_family=circuits_by_family, device=device
    )
    ref_emb_n = normalize_rows(ref_emb)
    print(f"✓ Reference embeddings: {ref_emb_n.shape}")

    print("Extracting 4 LLMBO embeddings...")
    test_emb_list: List[np.ndarray] = []
    for spec in test_specs:
        emb = extract_embedding_for_npz_dir(model=model, npz_dir=spec.path, device=device)
        test_emb_list.append(emb)
        print(f"✓ {spec.name}: embedding_dim={emb.shape[0]} (expected family={spec.expected_family})")

    test_emb = np.vstack(test_emb_list)
    test_emb_n = normalize_rows(test_emb)

    # Fit UMAP on reference embeddings, transform test circuits
    umap = _lazy_import_umap()

    print("Fitting UMAP on reference set, then transforming test circuits...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=int(args.umap_neighbors),
        min_dist=float(args.umap_min_dist),
        metric="euclidean",
    )
    umap_ref = reducer.fit_transform(ref_emb_n)
    umap_test = reducer.transform(test_emb_n)

    plot_path = str(out_dir / "umap_overlay_train_test_plus_4.png")
    plot_umap_overlay(
        umap_ref=umap_ref,
        ref_meta=ref_meta,
        umap_test=umap_test,
        test_specs=test_specs,
        output_path=plot_path,
    )
    print(f"✓ Saved UMAP overlay: {plot_path}")

    # Family centroids in embedding space (reference set)
    centroids = compute_family_centroids(ref_emb_n, ref_meta)

    # Distances to family centroids (normalized per test circuit)
    dist_rows: List[Dict] = []
    for i, spec in enumerate(test_specs):
        normed = normalize_distances_to_centroids(
            test_emb_n[i, :], centroids=centroids, normalize=args.distance_normalization
        )
        row = {"test_circuit": spec.name, "expected_family": spec.expected_family}
        row.update({f"dist_to_{fam}": normed[fam] for fam in sorted(normed.keys())})
        dist_rows.append(row)

    dist_csv = str(out_dir / "test_circuits_distance_to_family_centroids.csv")
    fieldnames = ["test_circuit", "expected_family"] + [
        f"dist_to_{fam}" for fam in sorted(centroids.keys())
    ]
    write_csv(dist_csv, dist_rows, fieldnames=fieldnames)
    print(f"✓ Saved centroid distances: {dist_csv}")

    # Nearest neighbors in embedding space (reference pool)
    dists = euclidean_distances(test_emb_n, ref_emb_n)  # [4, N]

    nn_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for i, spec in enumerate(test_specs):
        order = np.argsort(dists[i, :])
        topk = int(args.topk)
        nn_idx = order[:topk]

        nn_fams = [ref_meta[j]["family"] for j in nn_idx]
        diff_family_count = int(sum(f != spec.expected_family for f in nn_fams))

        summary_rows.append(
            {
                "test_circuit": spec.name,
                "expected_family": spec.expected_family,
                "topk": topk,
                "num_neighbors_not_in_expected_family": diff_family_count,
                "num_neighbors_in_expected_family": topk - diff_family_count,
            }
        )

        for rank, j in enumerate(nn_idx, start=1):
            nn_rows.append(
                {
                    "test_circuit": spec.name,
                    "expected_family": spec.expected_family,
                    "rank": rank,
                    "neighbor_family": ref_meta[j]["family"],
                    "neighbor_circuit_id": ref_meta[j]["circuit_id"],
                    "distance": float(dists[i, j]),
                }
            )

    nn_csv = str(out_dir / "test_circuits_topk_neighbors.csv")
    write_csv(
        nn_csv,
        nn_rows,
        fieldnames=[
            "test_circuit",
            "expected_family",
            "rank",
            "neighbor_family",
            "neighbor_circuit_id",
            "distance",
        ],
    )

    summary_csv = str(out_dir / "test_circuits_topk_family_mismatch_summary.csv")
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "test_circuit",
            "expected_family",
            "topk",
            "num_neighbors_in_expected_family",
            "num_neighbors_not_in_expected_family",
        ],
    )

    with open(out_dir / "test_circuits_topk_neighbors.json", "w") as f:
        json.dump(
            {
                "topk": int(args.topk),
                "neighbors": nn_rows,
                "summary": summary_rows,
            },
            f,
            indent=2,
        )

    print(f"✓ Saved top-k neighbors: {nn_csv}")
    print(f"✓ Saved family mismatch summary: {summary_csv}")

    # Also store the 2D coordinates for convenience
    with open(out_dir / "umap_projection_with_test.json", "w") as f:
        json.dump(
            {
                "reference": {
                    "umap": umap_ref.tolist(),
                    "metadata": ref_meta,
                },
                "test": {
                    "umap": umap_test.tolist(),
                    "metadata": [
                        {
                            "name": s.name,
                            "path": s.path,
                            "expected_family": s.expected_family,
                        }
                        for s in test_specs
                    ],
                },
            },
            f,
            indent=2,
        )

    print(f"\nAll outputs are in: {out_dir}")


if __name__ == "__main__":
    main()
