#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def euclidean_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    ab = a @ b.T
    d2 = np.maximum(a2 + b2 - 2.0 * ab, 0.0)
    return np.sqrt(d2)


def _ordered_multiset_overlap(left: List[str], right: List[str]) -> List[str]:
    remaining = Counter(right)
    overlap: List[str] = []
    for value in left:
        if remaining[value] > 0:
            overlap.append(value)
            remaining[value] -= 1
    return overlap


def compute_overlap(npz_path: Path, topk: int, compare_by: str) -> Dict[str, Dict[str, object]]:
    data = np.load(npz_path, allow_pickle=True)
    emb_sg = data["emb_sg"].astype(np.float64)
    emb_skg = data["emb_skg"].astype(np.float64)
    meta = json.loads(str(data["meta_json"].item()))

    anchors = ["amp2", "FC", "comp", "ldo"]
    net_indices = [idx for idx, row in enumerate(meta) if row["source"] == "netlists"]
    anchor_indices = {
        row["circuit_id"]: idx for idx, row in enumerate(meta) if row["source"] == "llmbo"
    }

    emb_sg_n = normalize_rows(emb_sg)
    emb_skg_n = normalize_rows(emb_skg)

    results: Dict[str, Dict[str, object]] = {}
    for anchor in anchors:
        anchor_idx = anchor_indices[anchor]
        sg_dist = euclidean_distances(emb_sg_n[[anchor_idx]], emb_sg_n[net_indices]).reshape(-1)
        skg_dist = euclidean_distances(emb_skg_n[[anchor_idx]], emb_skg_n[net_indices]).reshape(-1)

        sg_order = np.argsort(sg_dist)[:topk]
        skg_order = np.argsort(skg_dist)[:topk]

        sg_topk = [str(meta[net_indices[idx]]["circuit_id"]) for idx in sg_order]
        skg_topk = [str(meta[net_indices[idx]]["circuit_id"]) for idx in skg_order]

        if compare_by == "family":
            sg_items = [str(meta[net_indices[idx]]["family"]) for idx in sg_order]
            skg_items = [str(meta[net_indices[idx]]["family"]) for idx in skg_order]
        else:
            sg_items = sg_topk
            skg_items = skg_topk

        overlap_items = _ordered_multiset_overlap(sg_items, skg_items)
        overlap_count = len(overlap_items)
        union_count = len(sg_items) + len(skg_items) - overlap_count

        results[anchor] = {
            "sg_topk": sg_topk,
            "skg_topk": skg_topk,
            "sg_topk_families": [str(meta[net_indices[idx]]["family"]) for idx in sg_order],
            "skg_topk_families": [str(meta[net_indices[idx]]["family"]) for idx in skg_order],
            "compare_by": compare_by,
            "overlap_items": overlap_items,
            "overlap_count": overlap_count,
            "overlap_at_k": overlap_count / float(topk),
            "jaccard": overlap_count / float(union_count) if union_count else 0.0,
        }

    return results


def write_csv(csv_path: Path, results: Dict[str, Dict[str, object]], topk: int) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "anchor",
                "topk",
                "compare_by",
                "overlap_count",
                "overlap_at_k",
                "jaccard",
                "overlap_items",
                "sg_topk_families",
                "skg_topk_families",
                "sg_topk",
                "skg_topk",
            ],
        )
        writer.writeheader()
        for anchor, payload in results.items():
            writer.writerow(
                {
                    "anchor": anchor,
                    "topk": topk,
                    "compare_by": payload["compare_by"],
                    "overlap_count": payload["overlap_count"],
                    "overlap_at_k": payload["overlap_at_k"],
                    "jaccard": payload["jaccard"],
                    "overlap_items": json.dumps(payload["overlap_items"]),
                    "sg_topk_families": json.dumps(payload["sg_topk_families"]),
                    "skg_topk_families": json.dumps(payload["skg_topk_families"]),
                    "sg_topk": json.dumps(payload["sg_topk"]),
                    "skg_topk": json.dumps(payload["skg_topk"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SG-vs-SKG top-k retrieval overlap for the four LLMBO anchor circuits."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("umap_results_full/sg_skg_analysis/sg_skg_embeddings.npz"),
        help="Combined SG/SKG embeddings artifact produced by analyze_sg_skg_embeddings.py",
    )
    parser.add_argument("--topk", type=int, default=10, help="Number of retrieved circuits to compare")
    parser.add_argument(
        "--compare-by",
        choices=["circuit_id", "family"],
        default="circuit_id",
        help="Compare retrieved items by exact circuit ID or by family label",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/analysis/sg_skg_topk_overlap.json"),
        help="Output JSON summary path",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/analysis/sg_skg_topk_overlap.csv"),
        help="Output CSV summary path",
    )
    args = parser.parse_args()

    results = compute_overlap(args.npz, args.topk, args.compare_by)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as handle:
        json.dump({"topk": args.topk, "compare_by": args.compare_by, "results": results}, handle, indent=2)

    write_csv(args.out_csv, results, args.topk)

    print(json.dumps({"topk": args.topk, "compare_by": args.compare_by, "results": results}, indent=2))
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()