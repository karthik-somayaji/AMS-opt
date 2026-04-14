#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def infer_gnn_model_cfg_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    def shape(key: str) -> Tuple[int, ...]:
        if key not in state_dict:
            raise KeyError(f"Missing key '{key}' in checkpoint state_dict")
        return tuple(state_dict[key].shape)

    input_dim = shape("encoder.gin_layers.0.mlp.0.weight")[1]
    hidden_dims: List[int] = []
    index = 0
    while f"encoder.gin_layers.{index}.mlp.0.weight" in state_dict:
        hidden_dims.append(int(shape(f"encoder.gin_layers.{index}.mlp.0.weight")[0]))
        index += 1

    embedding_dim = shape("encoder.readout_mlp.0.weight")[0]
    projection_dim = shape("projection_head.mlp.0.weight")[0]
    use_batch_norm = any(key.startswith("encoder.batch_norms.") for key in state_dict.keys())
    return {
        "input_dim": int(input_dim),
        "hidden_dims": hidden_dims,
        "embedding_dim": int(embedding_dim),
        "projection_dim": int(projection_dim),
        "use_batch_norm": bool(use_batch_norm),
    }


def numeric_sort_key(value: str) -> Tuple[int, Any]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def scan_circuit_bank(netlists_root: Path) -> List[Dict[str, str]]:
    metadata: List[Dict[str, str]] = []
    for family_dir in sorted(path for path in netlists_root.iterdir() if path.is_dir()):
        family = family_dir.name
        circuit_dirs = sorted(
            [path for path in family_dir.iterdir() if path.is_dir() and (path / "comb_graph_gnn.npz").is_file()],
            key=lambda path: numeric_sort_key(path.name),
        )
        for circuit_dir in circuit_dirs:
            metadata.append({"family": family, "circuit_id": circuit_dir.name})
    if not metadata:
        raise ValueError(f"No circuit directories with comb_graph_gnn.npz found under {netlists_root}")
    return metadata


def load_model(checkpoint_path: Path, device: str):
    import torch

    repo_root = checkpoint_path.resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from gnn_training.models import ContrastiveGINModel

    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device(device))
    state_dict = checkpoint.get("model_state", checkpoint)
    cfg = infer_gnn_model_cfg_from_state_dict(state_dict)

    model = ContrastiveGINModel(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        embedding_dim=cfg["embedding_dim"],
        projection_dim=cfg["projection_dim"],
        dropout=0.0,
        use_batch_norm=cfg["use_batch_norm"],
    )
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()
    return model, cfg


def recompute_embeddings(
    *,
    model: Any,
    metadata: List[Dict[str, str]],
    netlists_root: Path,
    device: str,
    embedding_mode: str,
) -> np.ndarray:
    import torch

    repo_root = netlists_root.resolve().parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from gnn_training.data import CircuitDataLoader
    from gnn_training.perturbations import RemoveKnowledgeNodes

    embeddings: List[np.ndarray] = []
    for index, item in enumerate(metadata, start=1):
        family = str(item["family"])
        circuit_id = str(item["circuit_id"])
        loader = CircuitDataLoader(circuit_id, str(netlists_root / family), cache=False)
        graph = loader.get_graph()
        if embedding_mode == "sg":
            graph = RemoveKnowledgeNodes(graph).apply()

        with torch.no_grad():
            features = torch.tensor(graph["features"], dtype=torch.float32, device=torch.device(device))
            adjacency = torch.tensor(graph["adjacency"], dtype=torch.float32, device=torch.device(device))
            embedding = model.encode(features, adjacency).detach().cpu().numpy().reshape(-1)
        embeddings.append(embedding)

        if index % 50 == 0 or index == len(metadata):
            print(f"[recompute_reference_embeddings] Encoded {index}/{len(metadata)} circuits", flush=True)

    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute raw GNN reference embeddings from netlists/*/*")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_full_training/best_model.pt")
    parser.add_argument("--netlists_root", type=str, default="netlists")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--embedding_mode", type=str, default="sg", choices=["sg", "skg"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()
    netlists_root = Path(args.netlists_root)
    if not netlists_root.is_absolute():
        netlists_root = (repo_root / netlists_root).resolve()
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = (repo_root / out_json).resolve()

    if out_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_json}. Use --overwrite to replace it.")

    metadata = scan_circuit_bank(netlists_root)
    print(
        f"[recompute_reference_embeddings] Found {len(metadata)} circuits across "
        f"{len(sorted(set(item['family'] for item in metadata)))} families under {netlists_root}",
        flush=True,
    )

    model, cfg = load_model(checkpoint_path=checkpoint_path, device=str(args.device))
    embeddings = recompute_embeddings(
        model=model,
        metadata=metadata,
        netlists_root=netlists_root,
        device=str(args.device),
        embedding_mode=str(args.embedding_mode).lower(),
    )

    payload = {
        "embeddings": embeddings.tolist(),
        "metadata": metadata,
        "embedding_dim": int(embeddings.shape[1]),
        "num_circuits": int(embeddings.shape[0]),
        "embedding_mode": str(args.embedding_mode).lower(),
        "checkpoint": str(checkpoint_path),
        "netlists_root": str(netlists_root),
        "model_cfg": cfg,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as handle:
        json.dump(payload, handle)

    print(
        f"[recompute_reference_embeddings] Wrote {embeddings.shape[0]} embeddings "
        f"(D={embeddings.shape[1]}) to {out_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()