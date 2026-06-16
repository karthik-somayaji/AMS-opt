"""Compute and visualize SG vs SKG embeddings.

Outputs:
- CSV of per-circuit distances between SG and SKG embeddings (cosine + L2 on L2-normalized vectors)
- PCA plot showing paired SG/SKG points per circuit
- Distance plots (netlists circuits + separate plot for LLMBO test circuits)
- Optional export of netlists-only embeddings JSONs for LLMBO retrieval

Run:
  python gnn_training/scripts/analyze_sg_skg_embeddings.py \
    --checkpoint checkpoints_sg_vs_skg/best_model.pt \
    --netlists_root netlists \
    --out_dir umap_results_full/sg_skg_analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, Set

import numpy as np
import torch
import yaml
import yaml


# Allow running directly by adding repo root (parent of `gnn_training/`) to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class CircuitRef(NamedTuple):
    source: str  # "netlists" | "llmbo"
    family: str
    circuit_id: str
    npz_path: str
    label: str


def _load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    from gnn_training.models import ContrastiveGINModel

    ckpt_abs = checkpoint_path if os.path.isabs(checkpoint_path) else str(_REPO_ROOT / checkpoint_path)
    ckpt = torch.load(ckpt_abs, map_location=torch.device(device))
    state_dict = ckpt.get("model_state", ckpt)

    def _shape(key: str) -> Tuple[int, ...]:
        if key not in state_dict:
            raise KeyError(f"Missing key '{key}' in checkpoint state_dict")
        return tuple(state_dict[key].shape)

    input_dim = _shape("encoder.gin_layers.0.mlp.0.weight")[1]
    hidden_dims: List[int] = []
    i = 0
    while f"encoder.gin_layers.{i}.mlp.0.weight" in state_dict:
        hidden_dims.append(_shape(f"encoder.gin_layers.{i}.mlp.0.weight")[0])
        i += 1
    embedding_dim = _shape("encoder.readout_mlp.0.weight")[0]
    projection_dim = _shape("projection_head.mlp.0.weight")[0]
    use_batch_norm = any(k.startswith("encoder.batch_norms.") for k in state_dict.keys())

    model = ContrastiveGINModel(
        input_dim=int(input_dim),
        hidden_dims=[int(x) for x in hidden_dims],
        embedding_dim=int(embedding_dim),
        projection_dim=int(projection_dim),
        dropout=0.0,
        use_batch_norm=bool(use_batch_norm),
    )
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()
    return model


def _load_npz_graph(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    nodes = data["nodes"]
    features = data["features"].astype(np.float32)
    adjacency = data["adjacency"].astype(np.float32)
    return {"nodes": nodes, "features": features, "adjacency": adjacency}


def _to_sg_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    from gnn_training.perturbations import RemoveKnowledgeNodes

    return RemoveKnowledgeNodes(graph).apply()


def _encode(model: torch.nn.Module, graph: Dict[str, Any], device: str) -> np.ndarray:
    features = graph["features"]
    adjacency = graph["adjacency"]
    with torch.no_grad():
        feat_t = torch.tensor(features, dtype=torch.float32, device=torch.device(device))
        adj_t = torch.tensor(adjacency, dtype=torch.float32, device=torch.device(device))
        emb = model.encode(feat_t, adj_t).detach().cpu().numpy().reshape(-1)
    return emb


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _cosine_distance(a_hat: np.ndarray, b_hat: np.ndarray) -> float:
    """Cosine distance scaled to [0, 1].

    Assumes inputs are already L2-normalized.

    - cosine similarity is in [-1, 1]
    - (1 - cos) is in [0, 2]
    - (1 - cos) / 2 is in [0, 1]
    """
    cos_sim = float(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0))
    return float((1.0 - cos_sim) / 2.0)


def _l2_distance(a_hat: np.ndarray, b_hat: np.ndarray) -> float:
    return float(np.linalg.norm(a_hat - b_hat))


def _walk_netlists(netlists_root: str) -> List[CircuitRef]:
    root = Path(netlists_root)
    refs: List[CircuitRef] = []
    if not root.exists():
        raise FileNotFoundError(f"netlists_root not found: {root}")

    for family_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        family = family_dir.name
        for cid_dir in sorted([p for p in family_dir.iterdir() if p.is_dir()]):
            npz_path = cid_dir / "comb_graph_gnn.npz"
            if not npz_path.exists():
                continue
            cid = cid_dir.name
            refs.append(
                CircuitRef(
                    source="netlists",
                    family=family,
                    circuit_id=str(cid),
                    npz_path=str(npz_path),
                    label=f"{family}/{cid}",
                )
            )
    return refs


def _llmbo_test_refs(repo_root: Path) -> List[CircuitRef]:
    # These are single-circuit directories, not family/id trees.
    mapping = {
        "amp2": repo_root / "LLMBO" / "amp2_ati_new" / "comb_graph_gnn.npz",
        "FC": repo_root / "LLMBO" / "FC_ati_new" / "comb_graph_gnn.npz",
        "comp": repo_root / "LLMBO" / "comp_ati_new" / "comb_graph_gnn.npz",
        "ldo": repo_root / "LLMBO" / "ldo_ati_new" / "comb_graph_gnn.npz",
    }
    refs: List[CircuitRef] = []
    for key, npz in mapping.items():
        if not npz.exists():
            raise FileNotFoundError(f"Missing LLMBO test circuit npz: {npz}")
        refs.append(
            CircuitRef(
                source="llmbo",
                family=key,
                circuit_id=key,
                npz_path=str(npz),
                label=f"LLMBO/{key}",
            )
        )
    return refs


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _export_embeddings_json(path: Path, embeddings: np.ndarray, meta: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "embeddings": embeddings.astype(float).tolist(),
        "metadata": meta,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _make_plots(
    out_dir: Path,
    refs: List[CircuitRef],
    emb_skg: np.ndarray,
    emb_sg: np.ndarray,
    distances: List[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_indices = [i for i, r in enumerate(refs) if r.source == "llmbo"]
    netlist_family_colors = {
        "diff_amps": "#6a4c93",
        "comparators": "#1982c4",
        "LDO": "#8ac926",
        "op-amp": "#ffca3a",
        "masala_chai": "#ff595e",
    }

    def _plot_umap_with_anchors(
        points: np.ndarray,
        title: str,
        filename: str,
        anchor_suffix: str,
    ) -> None:
        plt.figure(figsize=(10, 8))

        base_indices = [i for i in range(points.shape[0]) if i not in anchor_indices]
        if base_indices:
            family_to_indices: Dict[str, List[int]] = {}
            for idx in base_indices:
                family_to_indices.setdefault(refs[idx].family, []).append(idx)

            for family, indices in sorted(family_to_indices.items()):
                color = netlist_family_colors.get(family, "#9aa0a6")
                plt.scatter(
                    points[indices, 0],
                    points[indices, 1],
                    s=10,
                    alpha=0.35,
                    c=color,
                    label=f"Netlists: {family}",
                )

        anchor_colors = {
            "amp2": "#d62728",
            "FC": "#1f77b4",
            "comp": "#2ca02c",
            "ldo": "#ff7f0e",
        }
        for idx in anchor_indices:
            ref = refs[idx]
            x, y = points[idx]
            color = anchor_colors.get(ref.family, "#111111")
            plt.scatter(
                [x],
                [y],
                s=140,
                c=color,
                edgecolors="black",
                linewidths=0.8,
                marker="*",
                zorder=5,
                label=f"Anchor: {ref.family}",
            )
            plt.annotate(
                f"{ref.family} ({anchor_suffix})",
                xy=(x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color=color,
                weight="bold",
            )

        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    # 1) PCA paired plot (normalized embeddings)
    X = np.concatenate([_l2_normalize(emb_skg), _l2_normalize(emb_sg)], axis=0)
    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(X)
    n = emb_skg.shape[0]
    skg2 = X2[:n]
    sg2 = X2[n:]

    plt.figure(figsize=(10, 8))
    plt.scatter(skg2[:, 0], skg2[:, 1], s=10, alpha=0.5, label="SKG (SG+KG)")
    plt.scatter(sg2[:, 0], sg2[:, 1], s=10, alpha=0.5, label="SG (KG removed)")
    for i in range(n):
        plt.plot([skg2[i, 0], sg2[i, 0]], [skg2[i, 1], sg2[i, 1]], linewidth=0.5, alpha=0.15)

    plt.title("Paired SKG vs SG embeddings (PCA, L2-normalized)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_pairs.png", dpi=200)
    plt.close()

    # 1b) UMAP paired plot (normalized embeddings), if available.
    # Note: UMAP is intended for visualization; do not interpret 2D distances as a metric.
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=0,
        )
        U2 = reducer.fit_transform(X)
        skg_u = U2[:n]
        sg_u = U2[n:]

        plt.figure(figsize=(10, 8))
        plt.scatter(skg_u[:, 0], skg_u[:, 1], s=10, alpha=0.5, label="SKG (SG+KG)")
        plt.scatter(sg_u[:, 0], sg_u[:, 1], s=10, alpha=0.5, label="SG (KG removed)")
        for i in range(n):
            plt.plot([skg_u[i, 0], sg_u[i, 0]], [skg_u[i, 1], sg_u[i, 1]], linewidth=0.5, alpha=0.15)
        plt.title("Paired SKG vs SG embeddings (UMAP, L2-normalized)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "umap_pairs.png", dpi=200)
        plt.close()

        _plot_umap_with_anchors(
            skg_u,
            "SKG embeddings (UMAP, anchors highlighted)",
            "umap_skg_with_anchors.png",
            "SKG",
        )
        _plot_umap_with_anchors(
            sg_u,
            "SG embeddings (UMAP, anchors highlighted)",
            "umap_sg_with_anchors.png",
            "SG",
        )
    except Exception:
        pass

    # 2) Distance plots (sorted)
    # split by source
    net_idx = [i for i, r in enumerate(refs) if r.source == "netlists"]
    llmbo_idx = [i for i, r in enumerate(refs) if r.source == "llmbo"]

    # Further split netlists into train vs test when available.
    net_train_idx = [i for i in net_idx if distances[i].get("split") == "train"]
    net_test_idx = [i for i in net_idx if distances[i].get("split") == "test"]

    def _plot_sorted(idxs: List[int], title: str, filename: str) -> None:
        if not idxs:
            return
        cos = np.array([distances[i]["cosine_dist"] for i in idxs], dtype=float)
        order = np.argsort(cos)
        cos_s = cos[order]

        plt.figure(figsize=(12, 4))
        plt.plot(cos_s, label="cosine_dist ((1 - cos) / 2)")
        plt.title(title)
        plt.xlabel("circuits (sorted by cosine_dist)")
        plt.ylabel("distance")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    _plot_sorted(net_idx, "SG vs SKG distance per netlists circuit (normalized)", "distances_netlists_sorted.png")

    # Optional train/test-only views (requires split labels).
    if net_train_idx and net_test_idx:
        _plot_sorted(
            net_train_idx,
            "SG vs SKG distance per TRAIN netlists circuit (normalized)",
            "distances_netlists_sorted_train.png",
        )
        _plot_sorted(
            net_test_idx,
            "SG vs SKG distance per TEST netlists circuit (normalized)",
            "distances_netlists_sorted_test.png",
        )

    # 3) LLMBO 4 test circuits bar plot
    if llmbo_idx:
        labels = [refs[i].family for i in llmbo_idx]
        cos = [distances[i]["cosine_dist"] for i in llmbo_idx]
        x = np.arange(len(labels))
        plt.figure(figsize=(8, 4))
        plt.bar(x, cos, width=0.6, label="cosine_dist")
        plt.xticks(x, labels)
        plt.title("LLMBO test circuits: SG vs SKG distance")
        plt.ylabel("distance")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "distances_llmbo_tests.png", dpi=200)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="checkpoints_sg_vs_skg/best_model.pt")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--netlists_root", type=str, default=str(_REPO_ROOT / "netlists"))
    ap.add_argument("--out_dir", type=str, default=str(_REPO_ROOT / "umap_results_full" / "sg_skg_analysis"))
    ap.add_argument("--max_netlists", type=int, default=0, help="If >0, cap number of netlists circuits (debug)")
    ap.add_argument(
        "--split_config",
        type=str,
        default="",
        help=(
            "Optional training YAML config to split netlists into train vs test. "
            "If not provided, will auto-load `config.yaml` next to the checkpoint (if present)."
        ),
    )
    ap.add_argument(
        "--export_llmbo_json",
        action="store_true",
        help="Also export netlists-only SG/SKG embeddings JSONs for LLMBO similarity search.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.checkpoint, device=args.device)

    # Determine train/test split for netlists circuits.
    # Prefer explicit --split_config; else try config.yaml next to checkpoint.
    all_cfg_circuits = set()  # type: Set[Tuple[str, str]]
    test_circuits = set()  # type: Set[Tuple[str, str]]
    train_circuits = set()  # type: Set[Tuple[str, str]]
    split_cfg_path: Optional[Path] = None
    if args.split_config:
        split_cfg_path = Path(args.split_config)
    else:
        ckpt_path = Path(args.checkpoint)
        ckpt_dir = ckpt_path.parent if ckpt_path.suffix else ckpt_path
        candidate = ckpt_dir / "config.yaml"
        if candidate.exists():
            split_cfg_path = candidate

    if split_cfg_path is not None and split_cfg_path.exists():
        with open(split_cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        data_cfg = cfg.get("data") or {}
        circuits_cfg = data_cfg.get("circuits") or {}
        for family, cids in circuits_cfg.items():
            if cids is None:
                continue
            for cid in cids:
                all_cfg_circuits.add((str(family), str(cid)))

        test_cfg = data_cfg.get("test_circuits") or {}
        for family, cids in test_cfg.items():
            if cids is None:
                continue
            for cid in cids:
                test_circuits.add((str(family), str(cid)))

        train_circuits = all_cfg_circuits - test_circuits

    # Build circuit list
    net_refs = _walk_netlists(args.netlists_root)
    if args.max_netlists and args.max_netlists > 0:
        net_refs = net_refs[: int(args.max_netlists)]
    llmbo_refs = _llmbo_test_refs(_REPO_ROOT)
    refs = net_refs + llmbo_refs

    emb_skg_list: List[np.ndarray] = []
    emb_sg_list: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    for r in refs:
        g = _load_npz_graph(r.npz_path)
        g_sg = _to_sg_graph(g)

        z_skg = _encode(model, g, device=args.device)
        z_sg = _encode(model, g_sg, device=args.device)

        z_skg_hat = _l2_normalize(z_skg)
        z_sg_hat = _l2_normalize(z_sg)

        cos_d = _cosine_distance(z_skg_hat, z_sg_hat)
        l2_d = _l2_distance(z_skg_hat, z_sg_hat)

        emb_skg_list.append(z_skg)
        emb_sg_list.append(z_sg)
        rows.append(
            {
                "source": r.source,
                "family": r.family,
                "circuit_id": r.circuit_id,
                "label": r.label,
                "split": (
                    "llmbo"
                    if r.source == "llmbo"
                    else (
                        "test"
                        if (r.family, r.circuit_id) in test_circuits
                        else ("train" if (r.family, r.circuit_id) in train_circuits else "other")
                    )
                ),
                "cosine_dist": cos_d,
                "l2_dist_norm": l2_d,
                "skg_norm": float(np.linalg.norm(z_skg)),
                "sg_norm": float(np.linalg.norm(z_sg)),
                "npz_path": r.npz_path,
            }
        )

    emb_skg = np.stack(emb_skg_list, axis=0)
    emb_sg = np.stack(emb_sg_list, axis=0)

    # Save combined NPZ (includes LLMBO test circuits)
    meta = [
        {
            "source": r.source,
            "family": r.family,
            "circuit_id": r.circuit_id,
            "label": r.label,
            "npz_path": r.npz_path,
        }
        for r in refs
    ]
    np.savez_compressed(
        out_dir / "sg_skg_embeddings.npz",
        emb_skg=emb_skg.astype(np.float32),
        emb_sg=emb_sg.astype(np.float32),
        meta_json=np.array(json.dumps(meta), dtype=object),
    )

    # Save distances
    _write_csv(out_dir / "sg_skg_distances.csv", rows)

    # Make plots
    _make_plots(out_dir, refs, emb_skg, emb_sg, rows)

    # Optionally export netlists-only embeddings JSONs for LLMBO retrieval
    if args.export_llmbo_json:
        net_meta = [{"circuit_id": r.circuit_id, "family": r.family} for r in net_refs]
        net_skg = emb_skg[: len(net_refs)]
        net_sg = emb_sg[: len(net_refs)]
        _export_embeddings_json(out_dir / "gnn_embeddings_skg_all_netlists.json", net_skg, net_meta)
        _export_embeddings_json(out_dir / "gnn_embeddings_sg_all_netlists.json", net_sg, net_meta)


if __name__ == "__main__":
    main()
