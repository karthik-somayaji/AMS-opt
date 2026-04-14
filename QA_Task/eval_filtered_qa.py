#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from eval_mcq import (
    _maybe_apply_chat_template,
    _optional_import,
    _repo_root,
    configure_vllm_worker_multiproc_method,
    extract_text_from_chat_completion,
    load_fun_graph,
    load_reference_embeddings,
    parse_choice_letter,
    vllm_chat_completions,
)


SUPPORTED_CIRCUITS = ["amp2", "FC", "comp", "ldo"]

TARGET_ALIAS_TO_REF_FAMILY = {
    "amp2": "diff_amps",
    "FC": "diff_amps",
    "comp": "comparators",
    "ldo": "LDO",
}

TARGET_ALIAS_TO_ATI_DIR = {
    "amp2": "amp2_ati_new",
    "FC": "FC_ati_new",
    "comp": "comp_ati_new",
    "ldo": "ldo_ati_new",
}

TARGET_ALIAS_TO_CURRENT_NETLIST_CANDIDATES = {
    "amp2": ["amp2.cir", "amp2_0.cir", "amp2_0.sp", "amp2.sp"],
    "FC": ["fc_0.cir", "fc_0.sp", "fc_og.sp"],
    "comp": ["comp_0.cir", "comp_0.sp", "comp.sp"],
    "ldo": ["ldo_0.cir", "ldo_0.sp", "ldo.sp"],
}

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Dict[str, Any]]] = {}
_ANCHOR_CACHE: Dict[Tuple[str, str, str, str], np.ndarray] = {}
_HF_CACHE_ENV_VARS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.7
_DEFAULT_OUT_JSONL = "filtered_qa_eval.jsonl"
_DEFAULT_SUMMARY_JSON = "filtered_qa_summary.json"
_DEFAULT_NETLIST_MAX_CHARS = 12000

_MODEL_PRESETS: List[Dict[str, Any]] = [
    {
        "name": "llama3-8b-instruct",
        "filename_slug": "llama3_8b",
        "aliases": [
            "llama3_8b",
            "llama3-8b",
            "llama3-8b-instruct",
            "meta-llama/meta-llama-3-8b-instruct",
            "models--meta-llama--meta-llama-3-8b-instruct",
        ],
        "cache_dir_name": "models--meta-llama--Meta-Llama-3-8B-Instruct",
        "hf_repo_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "is_reasoning_model": False,
        "recommended_gpu_memory_utilization": 0.7,
    },
    {
        "name": "llama3-70b-instruct",
        "filename_slug": "llama3_70b",
        "aliases": [
            "llama3_70b",
            "llama3-70b",
            "llama3-70b-instruct",
            "meta-llama/meta-llama-3-70b-instruct",
            "models--meta-llama--meta-llama-3-70b-instruct",
        ],
        "cache_dir_name": "models--meta-llama--Meta-Llama-3-70B-Instruct",
        "hf_repo_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "is_reasoning_model": False,
        "recommended_gpu_memory_utilization": 0.95,
    },
    {
        "name": "qwen2-7b-instruct",
        "filename_slug": "qwen2_7b",
        "aliases": [
            "qwen2_7b",
            "qwen2-7b",
            "qwen2-7b-instruct",
            "qwen/qwen2-7b-instruct",
            "models--qwen--qwen2-7b-instruct",
        ],
        "cache_dir_name": "models--Qwen--Qwen2-7B-Instruct",
        "hf_repo_id": "Qwen/Qwen2-7B-Instruct",
        "is_reasoning_model": False,
        "recommended_gpu_memory_utilization": 0.75,
    },
    {
        "name": "qwen2.5-32b-instruct",
        "filename_slug": "qwen2.5_32b",
        "aliases": [
            "qwen2.5_32b",
            "qwen2_5_32b",
            "qwen2.5-32b",
            "qwen2.5-32b-instruct",
            "qwen/qwen2.5-32b-instruct",
            "models--qwen--qwen2.5-32b-instruct",
        ],
        "cache_dir_name": "models--Qwen--Qwen2.5-32B-Instruct",
        "hf_repo_id": "Qwen/Qwen2.5-32B-Instruct",
        "is_reasoning_model": False,
        "recommended_gpu_memory_utilization": 0.95,
    },
    {
        "name": "phi-4-reasoning",
        "filename_slug": "phi4_reasoning",
        "aliases": [
            "phi4",
            "phi4_reasoning",
            "phi-4-reasoning",
            "phi-4-reasoning",
            "microsoft/phi-4-reasoning",
            "models--microsoft--phi-4-reasoning",
        ],
        "cache_dir_name": "models--microsoft--Phi-4-reasoning",
        "hf_repo_id": "microsoft/Phi-4-reasoning",
        "is_reasoning_model": True,
        "recommended_gpu_memory_utilization": 0.82,
    },
    {
        "name": "deepseek-r1-distill-qwen-14b",
        "filename_slug": "deepseek_qwen14b",
        "aliases": [
            "deepseek_qwen14b",
            "deepseek-qwen14b",
            "deepseek-r1-distill-qwen-14b",
            "deepseek-ai/deepseek-r1-distill-qwen-14b",
            "models--deepseek-ai--deepseek-r1-distill-qwen-14b",
        ],
        "cache_dir_name": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "is_reasoning_model": True,
        "recommended_gpu_memory_utilization": 0.86,
    },
    {
        "name": "deepseek-r1-distill-qwen-32b",
        "filename_slug": "deepseek_qwen32b",
        "aliases": [
            "deepseek_qwen32b",
            "deepseek-qwen32b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-ai/deepseek-r1-distill-qwen-32b",
            "models--deepseek-ai--deepseek-r1-distill-qwen-32b",
        ],
        "cache_dir_name": "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "is_reasoning_model": True,
        "recommended_gpu_memory_utilization": 0.95,
    },
    {
        "name": "deepseek-r1-distill-llama-70b",
        "filename_slug": "deepseek_llama70b",
        "aliases": [
            "deepseek_llama70b",
            "deepseek-llama70b",
            "deepseek--llama-70b",
            "deepseek-r1-distill-llama-70b",
            "deepseek-ai/deepseek-r1-distill-llama-70b",
            "models--deepseek-ai--deepseek-r1-distill-llama-70b",
        ],
        "cache_dir_name": "models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B",
        "hf_repo_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "is_reasoning_model": True,
        "recommended_gpu_memory_utilization": 0.95,
    },
]
_MODEL_PRESETS_BY_ALIAS: Dict[str, Dict[str, Any]] = {
    alias: preset for preset in _MODEL_PRESETS for alias in preset["aliases"]
}
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=\-]?\s*([ABCD])\b", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _emit_timing(args: argparse.Namespace, label: str, duration_s: float, **extra: Any) -> None:
    if not getattr(args, "log_timings", False):
        return
    extra_parts = [f"{key}={value}" for key, value in extra.items() if value is not None]
    suffix = f" ({', '.join(extra_parts)})" if extra_parts else ""
    print(f"[QA_Task][timing] {label}: {duration_s:.3f}s{suffix}", flush=True)


def _iter_hf_cache_roots() -> Iterable[Path]:
    seen: set[str] = set()
    for env_name in _HF_CACHE_ENV_VARS:
        value = os.environ.get(env_name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            yield candidate

    for candidate in [Path("/data/karthik/huggingface/hub"), Path.home() / ".cache" / "huggingface" / "hub"]:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            yield candidate


def _resolve_cached_model_dir(cache_dir_name: str) -> Optional[Path]:
    for root in _iter_hf_cache_roots():
        candidate = root / cache_dir_name
        if candidate.exists():
            return candidate.resolve()
    return None


def _infer_reasoning_model(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return any(token in normalized for token in ["reason", "r1", "deepseek-r1", "phi-4-reasoning"])


def resolve_model_profile(model_id_or_path: str) -> Dict[str, Any]:
    requested = str(model_id_or_path).strip()
    normalized = requested.lower()
    preset = _MODEL_PRESETS_BY_ALIAS.get(normalized)

    cache_dir_name = None
    hf_repo_id = requested
    model_source = requested
    if preset is not None:
        cache_dir_name = str(preset["cache_dir_name"])
        hf_repo_id = str(preset["hf_repo_id"])
        cached_dir = _resolve_cached_model_dir(cache_dir_name)
        model_source = str(cached_dir) if cached_dir is not None else hf_repo_id
    else:
        requested_path = Path(requested).expanduser()
        if requested_path.exists():
            model_source = str(requested_path)
        elif requested.startswith("models--"):
            cached_dir = _resolve_cached_model_dir(requested)
            if cached_dir is not None:
                cache_dir_name = requested
                model_source = str(cached_dir)

    resolved_model = resolve_vllm_model_path(model_source)
    return {
        "requested": requested,
        "profile_name": str(preset["name"]) if preset is not None else None,
        "filename_slug": str(preset["filename_slug"]) if preset is not None else None,
        "cache_dir_name": cache_dir_name,
        "hf_repo_id": hf_repo_id,
        "model_source": model_source,
        "resolved_model": resolved_model,
        "is_reasoning_model": bool(preset["is_reasoning_model"]) if preset is not None else _infer_reasoning_model(requested),
        "recommended_gpu_memory_utilization": (
            float(preset["recommended_gpu_memory_utilization"]) if preset is not None else None
        ),
    }


def apply_model_profile_defaults(args: argparse.Namespace) -> Dict[str, Any]:
    profile = resolve_model_profile(args.vllm_model)
    args.vllm_model_source = profile["model_source"]
    args.resolved_vllm_model = profile["resolved_model"]
    args.model_profile_name = profile["profile_name"]
    args.model_is_reasoning = bool(profile["is_reasoning_model"])
    args.model_filename_slug = profile.get("filename_slug")
    if (
        profile.get("recommended_gpu_memory_utilization") is not None
        and float(args.gpu_memory_utilization) == float(_DEFAULT_GPU_MEMORY_UTILIZATION)
    ):
        args.gpu_memory_utilization = float(profile["recommended_gpu_memory_utilization"])
    filename_slug = str(profile.get("filename_slug") or "").strip()
    if filename_slug and str(args.out_jsonl) == _DEFAULT_OUT_JSONL:
        args.out_jsonl = f"filtered_qa_eval_{filename_slug}.jsonl"
    if filename_slug and str(args.summary_json) == _DEFAULT_SUMMARY_JSON:
        args.summary_json = f"filtered_qa_eval_{filename_slug}_summary.json"
    return profile


def _parse_choice_letter_robust(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = _THINK_BLOCK_RE.sub(" ", text).strip()
    direct = _FINAL_ANSWER_RE.search(cleaned)
    if direct:
        return str(direct.group(1)).upper()
    parsed = parse_choice_letter(cleaned)
    if parsed is not None:
        return parsed
    matches = _LETTER_RE.findall(cleaned.upper())
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[-1]
    return None


def _has_vllm_model_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    weight_patterns = ["*.safetensors", "*.bin", "*.pt", "*.pth"]
    for pattern in weight_patterns:
        if any(path.glob(pattern)):
            return True
    return any(
        (path / filename).is_file()
        for filename in [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]
    )


def resolve_vllm_model_path(model_id_or_path: str) -> str:
    candidate = Path(str(model_id_or_path)).expanduser()
    if not candidate.exists():
        return str(model_id_or_path)

    if candidate.is_dir():
        snapshots_dir = candidate / "snapshots"
        refs_main = candidate / "refs" / "main"
        if snapshots_dir.is_dir():
            if refs_main.is_file():
                revision = refs_main.read_text().strip()
                preferred = snapshots_dir / revision
                if _has_vllm_model_weights(preferred):
                    return str(preferred.resolve())

            snapshot_dirs = sorted(
                [path for path in snapshots_dir.iterdir() if path.is_dir()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for snapshot_dir in snapshot_dirs:
                if _has_vllm_model_weights(snapshot_dir):
                    return str(snapshot_dir.resolve())

    return str(candidate.resolve())


def _normalize_text(text: Any) -> str:
    return str(text).strip().casefold()


def _truncate_text_block(text: Optional[str], *, max_chars: int, suffix: str) -> Optional[str]:
    if text is None:
        return None
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + suffix
    return text


def _resolve_current_circuit_netlist_path(repo_root: Path, circuit: str) -> Path:
    if circuit not in TARGET_ALIAS_TO_ATI_DIR:
        raise ValueError(f"Unsupported circuit alias: {circuit}")

    ati_dir = repo_root / "LLMBO" / TARGET_ALIAS_TO_ATI_DIR[circuit]
    for candidate_name in TARGET_ALIAS_TO_CURRENT_NETLIST_CANDIDATES.get(circuit, []):
        candidate_path = ati_dir / candidate_name
        if candidate_path.is_file():
            return candidate_path

    for pattern in ["*.cir", "*.sp"]:
        matches = sorted(path for path in ati_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Missing current circuit netlist for '{circuit}' in {ati_dir}")


def load_current_circuit_netlist(repo_root: Path, circuit: str, *, netlist_max_chars: int) -> Dict[str, Any]:
    path = _resolve_current_circuit_netlist_path(repo_root, circuit)
    netlist_text = path.read_text().strip()
    return {
        "netlist_loaded": True,
        "netlist_text": _truncate_text_block(netlist_text, max_chars=netlist_max_chars, suffix="\n\n(TRUNCATED NETLIST)\n"),
        "netlist_path": str(path),
    }


def load_related_netlists(
    repo_root: Path,
    related: List[Dict[str, Any]],
    *,
    netlist_max_chars: int,
) -> Tuple[List[Dict[str, Any]], int]:
    related_netlists: List[Dict[str, Any]] = []
    num_missing = 0
    for rel in related:
        circuit_dir = repo_root / "netlists" / rel["family"] / str(rel["circuit_id"])
        candidates = [
            circuit_dir / f"{rel['circuit_id']}.cir",
            circuit_dir / f"{rel['circuit_id']}.sp",
        ]
        resolved_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if resolved_path is None:
            for pattern in ["*.cir", "*.sp"]:
                matches = sorted(path for path in circuit_dir.glob(pattern) if path.is_file())
                if matches:
                    resolved_path = matches[0]
                    break

        if resolved_path is None:
            num_missing += 1
            related_netlists.append(
                {
                    "netlist_loaded": False,
                    "netlist_text": None,
                    "netlist_path": str(circuit_dir),
                }
            )
            continue

        netlist_text = resolved_path.read_text().strip()
        related_netlists.append(
            {
                "netlist_loaded": True,
                "netlist_text": _truncate_text_block(
                    netlist_text,
                    max_chars=netlist_max_chars,
                    suffix="\n\n(TRUNCATED NETLIST)\n",
                ),
                "netlist_path": str(resolved_path),
            }
        )
    return related_netlists, num_missing


def iter_filtered_qa_files(filtered_root: Path, circuits: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    for circuit in circuits:
        path = filtered_root / f"{circuit}_filtered_QA.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing filtered QA file for circuit '{circuit}': {path}")
        yield circuit, path


def _answer_letter_from_options(options: Dict[str, Any], golden_answer: str, *, path: Path, index: int) -> str:
    normalized_golden = _normalize_text(golden_answer)
    for letter in ["A", "B", "C", "D"]:
        if _normalize_text(options.get(letter)) == normalized_golden:
            return letter
    raise ValueError(
        f"Could not map golden_answer to A/B/C/D option at {path} item #{index}: golden_answer={golden_answer!r}"
    )


def load_filtered_qa_items(path: Path, circuit: str) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise ValueError(f"Filtered QA file must be a JSON array: {path}")

    items: List[Dict[str, Any]] = []
    for idx, entry in enumerate(obj):
        if not isinstance(entry, dict):
            raise ValueError(f"Filtered QA item must be an object at {path} item #{idx}")

        question = entry.get("question")
        options = entry.get("options")
        golden_answer = entry.get("golden_answer")
        item_id = entry.get("id")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing/invalid 'question' at {path} item #{idx}")
        if not isinstance(options, dict) or not all(letter in options for letter in ["A", "B", "C", "D"]):
            raise ValueError(f"Missing/invalid 'options' at {path} item #{idx}")
        if not isinstance(golden_answer, str) or not golden_answer.strip():
            raise ValueError(f"Missing/invalid 'golden_answer' at {path} item #{idx}")

        answer_letter = _answer_letter_from_options(options, golden_answer, path=path, index=idx)
        items.append(
            {
                "circuit": circuit,
                "reference_family": TARGET_ALIAS_TO_REF_FAMILY[circuit],
                "question_id": str(item_id) if item_id is not None else f"{circuit}_{idx:05d}",
                "question": question.strip(),
                "options": {letter: str(options[letter]) for letter in ["A", "B", "C", "D"]},
                "answer": answer_letter,
                "golden_answer": golden_answer.strip(),
                "sentence": str(entry.get("sentence", "")).strip(),
                "verification": entry.get("verification"),
                "source_path": str(path),
            }
        )
    return items


def build_question_only_prompt(
    item: Dict[str, Any],
    *,
    current_netlist: Dict[str, Any],
    reasoning_model: bool = False,
) -> str:
    options = item["options"]
    reasoning_suffix = (
        "Do not output your reasoning, chain-of-thought, or any <think> block. "
        "If you reason internally, keep it hidden. Return exactly one uppercase letter.\n\n"
        if reasoning_model
        else ""
    )
    if not current_netlist.get("netlist_loaded") or not current_netlist.get("netlist_text"):
        raise ValueError(f"Missing current circuit netlist for circuit={item['circuit']}")
    return (
        "You are an analog circuit expert. Answer the multiple-choice question using the current circuit netlist.\n"
        "Return ONLY the option letter (A, B, C, or D). No explanation.\n\n"
        f"{reasoning_suffix}"
        "CURRENT CIRCUIT NETLIST:\n"
        "```spice\n"
        f"{current_netlist['netlist_text']}\n"
        "```\n\n"
        f"QUESTION: {item['question']}\n\n"
        "OPTIONS:\n"
        f"A) {options['A']}\n"
        f"B) {options['B']}\n"
        f"C) {options['C']}\n"
        f"D) {options['D']}\n"
    )


def build_retrieval_augmented_prompt(
    item: Dict[str, Any],
    current_netlist: Dict[str, Any],
    related: List[Dict[str, Any]],
    related_netlists: List[Dict[str, Any]],
    related_kgs: List[Dict[str, Any]],
    *,
    reasoning_model: bool = False,
) -> str:
    baseline = build_question_only_prompt(item, current_netlist=current_netlist, reasoning_model=reasoning_model)
    blocks: List[str] = [
        "\nADDITIONAL HELPFUL CONTEXT FROM RELATED CIRCUITS:\n"
        "Use the related circuit netlists and their associated knowledge graphs as supporting evidence. "
        "Compare them with the current circuit netlist to infer structure-function relationships that help answer the current question. "
        "Treat the related circuits as analogous references, not as the target circuit itself.\n"
    ]
    for rel, netlist, kg in zip(related, related_netlists, related_kgs):
        header = f"[RELATED_CIRCUIT family={rel['family']} id={rel['circuit_id']} similarity={rel['similarity']:.4f}]"
        parts = [header]
        if netlist.get("netlist_text"):
            parts.append("RELATED CIRCUIT NETLIST:\n```spice\n" + str(netlist["netlist_text"]) + "\n```")
        else:
            parts.append("RELATED CIRCUIT NETLIST:\n(NETLIST missing)")
        if kg.get("kg_text"):
            parts.append("RELATED CIRCUIT KG:\n```json\n" + kg["kg_text"] + "\n```")
        else:
            parts.append("RELATED CIRCUIT KG:\n(KG missing)")
        blocks.append("\n".join(parts) + "\n")
    return baseline + "\n".join(blocks)


def _infer_gnn_model_cfg_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    def _shape(key: str) -> Tuple[int, ...]:
        if key not in state_dict:
            raise KeyError(f"Missing key '{key}' in checkpoint state_dict")
        return tuple(state_dict[key].shape)

    input_dim = _shape("encoder.gin_layers.0.mlp.0.weight")[1]
    hidden_dims: List[int] = []
    index = 0
    while f"encoder.gin_layers.{index}.mlp.0.weight" in state_dict:
        hidden_dims.append(int(_shape(f"encoder.gin_layers.{index}.mlp.0.weight")[0]))
        index += 1

    embedding_dim = _shape("encoder.readout_mlp.0.weight")[0]
    projection_dim = _shape("projection_head.mlp.0.weight")[0]
    use_batch_norm = any(key.startswith("encoder.batch_norms.") for key in state_dict.keys())
    return {
        "input_dim": int(input_dim),
        "hidden_dims": hidden_dims,
        "embedding_dim": int(embedding_dim),
        "projection_dim": int(projection_dim),
        "use_batch_norm": bool(use_batch_norm),
    }


def _load_gnn_model(checkpoint_path: Path, device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    cache_key = (str(checkpoint_path), str(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    torch = _optional_import("torch")
    if torch is None:
        raise RuntimeError("Missing dependency 'torch'. Install it in the active environment.")

    repo_root = _repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from gnn_training.models import ContrastiveGINModel

    ckpt = torch.load(str(checkpoint_path), map_location=torch.device(device))
    state_dict = ckpt.get("model_state", ckpt)
    cfg = _infer_gnn_model_cfg_from_state_dict(state_dict)

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

    _MODEL_CACHE[cache_key] = (model, cfg)
    return model, cfg


def _encode_comb_graph_npz_mode(
    model: Any,
    *,
    npz_path: Path,
    device: str = "cpu",
    embedding_mode: str = "sg",
) -> np.ndarray:
    torch = _optional_import("torch")
    if torch is None:
        raise RuntimeError("Missing dependency 'torch'. Install it in the active environment.")

    mode = str(embedding_mode or "sg").lower()
    if mode not in {"sg", "skg"}:
        raise ValueError(f"Invalid embedding_mode: {mode}. Expected 'sg' or 'skg'.")

    data = np.load(str(npz_path), allow_pickle=True)
    nodes = data.get("nodes", None)
    features = data["features"].astype(np.float32)
    adjacency = data["adjacency"].astype(np.float32)

    if mode == "sg":
        repo_root = _repo_root()
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from gnn_training.perturbations import RemoveKnowledgeNodes

        if nodes is None:
            raise ValueError(f"SG mode requires 'nodes' in npz: {npz_path}")
        sg_graph = RemoveKnowledgeNodes({"nodes": nodes, "features": features, "adjacency": adjacency}).apply()
        features = sg_graph["features"]
        adjacency = sg_graph["adjacency"]

    with torch.no_grad():
        feat_t = torch.tensor(features, dtype=torch.float32, device=torch.device(device))
        adj_t = torch.tensor(adjacency, dtype=torch.float32, device=torch.device(device))
        emb = model.encode(feat_t, adj_t).detach().cpu().numpy().reshape(-1)
    return emb


def resolve_embeddings_path(args: argparse.Namespace, repo_root: Path) -> Path:
    embedding_mode = str(getattr(args, "gnn_embedding_mode", "sg") or "sg").lower()
    default_rel = "umap_results_full/gnn_embeddings_sg.json" if embedding_mode == "sg" else "umap_results_full/gnn_embeddings.json"
    requested = args.embeddings_json or default_rel
    if requested == "umap_results_full/gnn_embeddings.json" and embedding_mode == "sg":
        requested = default_rel
    return Path(requested) if os.path.isabs(requested) else (repo_root / requested).resolve()


def get_anchor_embedding(args: argparse.Namespace, circuit: str) -> np.ndarray:
    repo_root = _repo_root()
    checkpoint = args.gnn_checkpoint or str((repo_root / "checkpoints_full_training" / "best_model.pt").resolve())
    checkpoint_path = Path(checkpoint) if os.path.isabs(checkpoint) else (repo_root / checkpoint).resolve()
    device = str(getattr(args, "gnn_device", "cpu") or "cpu")
    embedding_mode = str(getattr(args, "gnn_embedding_mode", "sg") or "sg").lower()
    cache_key = (circuit, str(checkpoint_path), device, embedding_mode)
    cached = _ANCHOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if circuit not in TARGET_ALIAS_TO_ATI_DIR:
        raise ValueError(f"Unsupported circuit alias: {circuit}")

    npz_path = repo_root / "LLMBO" / TARGET_ALIAS_TO_ATI_DIR[circuit] / "comb_graph_gnn.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing anchor graph features for circuit '{circuit}': {npz_path}")

    model, _ = _load_gnn_model(checkpoint_path=checkpoint_path, device=device)
    anchor = _encode_comb_graph_npz_mode(model, npz_path=npz_path, device=device, embedding_mode=embedding_mode)
    _ANCHOR_CACHE[cache_key] = anchor
    return anchor


def cosine_topk_from_anchor(
    embeddings: np.ndarray,
    metadata: List[Dict[str, str]],
    *,
    query_family: str,
    query_vector: np.ndarray,
    k: int,
    descending: bool = True,
) -> List[Dict[str, Any]]:
    family_indices = [index for index, meta in enumerate(metadata) if str(meta.get("family")) == str(query_family)]
    if not family_indices:
        raise KeyError(f"No reference circuits found for family={query_family!r}")

    family_embeddings = embeddings[family_indices]
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if family_embeddings.shape[1] != query.shape[0]:
        raise ValueError(
            "Anchor embedding dimension does not match reference embeddings. "
            f"Got anchor_dim={query.shape[0]} but reference_dim={family_embeddings.shape[1]}."
        )

    denom = np.linalg.norm(family_embeddings, axis=1) * (np.linalg.norm(query) + 1e-8) + 1e-8
    sims = (family_embeddings @ query) / denom
    order = np.argsort(-sims if descending else sims)
    top_count = min(max(int(k), 1), len(family_indices))

    out: List[Dict[str, Any]] = []
    for local_idx in order[:top_count].tolist():
        global_idx = family_indices[local_idx]
        meta = metadata[global_idx]
        out.append(
            {
                "family": str(meta["family"]),
                "circuit_id": str(meta["circuit_id"]),
                "similarity": float(sims[local_idx]),
            }
        )
    return out


def load_related_kgs(
    repo_root: Path,
    related: List[Dict[str, Any]],
    *,
    kg_max_chars: int,
) -> Tuple[List[Dict[str, Any]], int]:
    related_kgs: List[Dict[str, Any]] = []
    num_missing = 0
    for rel in related:
        kg_text = load_fun_graph(repo_root, rel["family"], rel["circuit_id"])
        if kg_text is None:
            num_missing += 1
        if kg_text is not None and kg_max_chars > 0 and len(kg_text) > kg_max_chars:
            kg_text = kg_text[:kg_max_chars] + "\n\n(TRUNCATED)\n"
        related_kgs.append(
            {
                "kg_loaded": kg_text is not None,
                "kg_text": kg_text,
                "kg_path": str(repo_root / "netlists" / rel["family"] / str(rel["circuit_id"]) / "fun_graph.json"),
            }
        )
    return related_kgs, num_missing


def _new_stats() -> Dict[str, Any]:
    return {
        "num_eval": 0,
        "baseline_correct": 0,
        "retrieval_correct": 0,
        "baseline_parse_fail": 0,
        "retrieval_parse_fail": 0,
    }


def _finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    num_eval = int(stats["num_eval"])
    result = dict(stats)
    if num_eval > 0:
        result.update(
            {
                "baseline_accuracy": float(stats["baseline_correct"] / num_eval),
                "retrieval_accuracy": float(stats["retrieval_correct"] / num_eval),
                "accuracy_gain": float((stats["retrieval_correct"] - stats["baseline_correct"]) / num_eval),
            }
        )
    return result


def prepare_eval_rows(args: argparse.Namespace) -> Dict[str, Any]:
    prep_t0 = time.perf_counter()
    tqdm_mod = _optional_import("tqdm")
    tqdm = getattr(tqdm_mod, "tqdm", None) if tqdm_mod else None

    repo_root = _repo_root()
    resolved_model = str(getattr(args, "resolved_vllm_model", "") or resolve_vllm_model_path(args.vllm_model))
    filtered_root = Path(args.filtered_qa_root)
    if not filtered_root.is_absolute():
        filtered_root = (repo_root / filtered_root).resolve()

    circuits = list(args.circuits or SUPPORTED_CIRCUITS)
    embeddings_path = resolve_embeddings_path(args, repo_root)
    embeddings, metadata = load_reference_embeddings(embeddings_path)

    file_items = list(iter_filtered_qa_files(filtered_root, circuits))
    iterator = tqdm(file_items, desc="filtered QA prep", total=len(file_items)) if tqdm is not None else file_items

    rows: List[Dict[str, Any]] = []
    prompts: List[str] = []
    by_circuit_related: Dict[str, List[Dict[str, Any]]] = {}
    by_circuit_related_netlists: Dict[str, List[Dict[str, Any]]] = {}
    by_circuit_related_kgs: Dict[str, List[Dict[str, Any]]] = {}
    by_circuit_current_netlist: Dict[str, Dict[str, Any]] = {}
    kg_missing = 0
    netlist_missing = 0
    num_seen = 0
    num_files = 0
    chat_template_calls = 0
    chat_template_total_s = 0.0
    first_chat_template_s: Optional[float] = None
    per_circuit_timings: Dict[str, Dict[str, float]] = {}

    for circuit, path in iterator:
        circuit_t0 = time.perf_counter()
        num_files += 1
        items = load_filtered_qa_items(path, circuit=circuit)
        load_items_s = time.perf_counter() - circuit_t0
        if args.limit_per_circuit and int(args.limit_per_circuit) > 0:
            items = items[: int(args.limit_per_circuit)]

        descending = str(args.related_mode).lower() != "bottomk"
        anchor_t0 = time.perf_counter()
        related = cosine_topk_from_anchor(
            embeddings,
            metadata,
            query_family=TARGET_ALIAS_TO_REF_FAMILY[circuit],
            query_vector=get_anchor_embedding(args, circuit),
            k=args.k,
            descending=descending,
        )
        retrieval_s = time.perf_counter() - anchor_t0
        current_netlist = load_current_circuit_netlist(
            repo_root,
            circuit,
            netlist_max_chars=int(args.netlist_max_chars),
        )
        netlist_t0 = time.perf_counter()
        related_netlists, missing_related_netlists = load_related_netlists(
            repo_root,
            related,
            netlist_max_chars=int(args.netlist_max_chars),
        )
        netlist_load_s = time.perf_counter() - netlist_t0
        kg_t0 = time.perf_counter()
        related_kgs, missing_for_circuit = load_related_kgs(
            repo_root,
            related,
            kg_max_chars=int(args.kg_max_chars),
        )
        kg_load_s = time.perf_counter() - kg_t0
        kg_missing += int(missing_for_circuit)
        netlist_missing += int(missing_related_netlists)
        by_circuit_related[circuit] = related
        by_circuit_related_netlists[circuit] = related_netlists
        by_circuit_related_kgs[circuit] = related_kgs
        by_circuit_current_netlist[circuit] = current_netlist

        prompt_build_s = 0.0
        for item in items:
            num_seen += 1
            prompt_t0 = time.perf_counter()
            baseline_prompt = build_question_only_prompt(
                item,
                current_netlist=current_netlist,
                reasoning_model=bool(getattr(args, "model_is_reasoning", False)),
            )
            retrieval_prompt = build_retrieval_augmented_prompt(
                item,
                current_netlist=current_netlist,
                related=related,
                related_netlists=related_netlists,
                related_kgs=related_kgs,
                reasoning_model=bool(getattr(args, "model_is_reasoning", False)),
            )

            if args.apply_chat_template:
                chat_t0 = time.perf_counter()
                baseline_prompt = _maybe_apply_chat_template(
                    prompt=baseline_prompt,
                    model_id_or_path=resolved_model,
                    trust_remote_code=bool(args.trust_remote_code),
                    download_dir=args.download_dir,
                    dry_run=bool(args.dry_run),
                )
                retrieval_prompt = _maybe_apply_chat_template(
                    prompt=retrieval_prompt,
                    model_id_or_path=resolved_model,
                    trust_remote_code=bool(args.trust_remote_code),
                    download_dir=args.download_dir,
                    dry_run=bool(args.dry_run),
                )
                chat_elapsed = time.perf_counter() - chat_t0
                chat_template_total_s += chat_elapsed
                chat_template_calls += 2
                if first_chat_template_s is None:
                    first_chat_template_s = chat_elapsed

            prompt_build_s += time.perf_counter() - prompt_t0

            row = {
                "item": item,
                "current_netlist": current_netlist,
                "related": related,
                "related_netlists": related_netlists,
                "related_kgs": related_kgs,
                "baseline_prompt": baseline_prompt,
                "retrieval_prompt": retrieval_prompt,
            }
            rows.append(row)
            prompts.append(baseline_prompt)
            prompts.append(retrieval_prompt)

        circuit_total_s = time.perf_counter() - circuit_t0
        per_circuit_timings[circuit] = {
            "load_items_s": float(load_items_s),
            "retrieval_s": float(retrieval_s),
            "netlist_load_s": float(netlist_load_s),
            "kg_load_s": float(kg_load_s),
            "prompt_build_s": float(prompt_build_s),
            "total_s": float(circuit_total_s),
        }
        _emit_timing(
            args,
            f"prep.{circuit}",
            circuit_total_s,
            items=len(items),
            retrieval_s=f"{retrieval_s:.3f}",
            netlist_load_s=f"{netlist_load_s:.3f}",
            kg_load_s=f"{kg_load_s:.3f}",
            prompt_build_s=f"{prompt_build_s:.3f}",
        )

    if args.dry_run and not args.limit_per_circuit and rows:
        rows = rows[:1]
        prompts = prompts[:2]

    prep_total_s = time.perf_counter() - prep_t0
    _emit_timing(
        args,
        "prepare_eval_rows",
        prep_total_s,
        files=num_files,
        rows=len(rows),
        prompts=len(prompts),
        chat_template_calls=chat_template_calls if chat_template_calls else None,
        chat_template_total_s=(f"{chat_template_total_s:.3f}" if chat_template_calls else None),
        first_chat_template_s=(f"{first_chat_template_s:.3f}" if first_chat_template_s is not None else None),
    )

    return {
        "repo_root": repo_root,
        "resolved_model": resolved_model,
        "filtered_root": filtered_root,
        "embeddings_path": embeddings_path,
        "embeddings": embeddings,
        "rows": rows,
        "prompts": prompts,
        "num_files": num_files,
        "num_seen": num_seen if not (args.dry_run and not args.limit_per_circuit and rows) else len(rows),
        "kg_missing": kg_missing,
        "netlist_missing": netlist_missing,
        "by_circuit_related": by_circuit_related,
        "by_circuit_related_netlists": by_circuit_related_netlists,
        "by_circuit_related_kgs": by_circuit_related_kgs,
        "by_circuit_current_netlist": by_circuit_current_netlist,
        "timings": {
            "prepare_eval_rows_s": float(prep_total_s),
            "chat_template_total_s": float(chat_template_total_s),
            "chat_template_calls": int(chat_template_calls),
            "first_chat_template_s": float(first_chat_template_s) if first_chat_template_s is not None else None,
            "per_circuit": per_circuit_timings,
        },
    }


def eval_with_vllm_local(args: argparse.Namespace) -> Dict[str, Any]:
    total_t0 = time.perf_counter()
    prepared = prepare_eval_rows(args)
    repo_root = prepared["repo_root"]
    rows = prepared["rows"]
    prompts = prepared["prompts"]
    embeddings = prepared["embeddings"]
    embeddings_path = prepared["embeddings_path"]
    resolved_model = prepared["resolved_model"]

    out_dir = repo_root / "QA_Task" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = (out_dir / out_jsonl).resolve()

    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = (out_dir / summary_path).resolve()

    outputs_text: List[str] = [""] * len(prompts)
    llm_init_s = 0.0
    generation_s = 0.0
    if not args.dry_run and prompts:
        llm_t0 = time.perf_counter()
        configure_vllm_worker_multiproc_method(getattr(args, "vllm_worker_multiproc_method", None))
        vllm = _optional_import("vllm")
        if vllm is None:
            raise RuntimeError("Missing dependency 'vllm'. Install with: pip install vllm")

        LLM = getattr(vllm, "LLM", None)
        SamplingParams = getattr(vllm, "SamplingParams", None)
        if LLM is None or SamplingParams is None:
            raise RuntimeError("vllm.LLM or vllm.SamplingParams not available")

        llm = LLM(
            model=resolved_model,
            tensor_parallel_size=int(args.tensor_parallel_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            dtype=(args.dtype or "auto"),
            trust_remote_code=bool(args.trust_remote_code),
            download_dir=args.download_dir,
            max_model_len=(int(args.max_model_len) if args.max_model_len else None),
            enforce_eager=bool(args.enforce_eager),
        )
        llm_init_s = time.perf_counter() - llm_t0
        _emit_timing(args, "vllm_load", llm_init_s, prompts=len(prompts), model=resolved_model)

        sampling = SamplingParams(
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            top_p=(float(args.top_p) if args.top_p is not None else 1.0),
        )

        batch_size = max(1, int(args.batch_size))
        gen_t0 = time.perf_counter()
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_outputs = llm.generate(batch_prompts, sampling)
            if len(batch_outputs) != len(batch_prompts):
                raise RuntimeError(
                    f"vLLM returned {len(batch_outputs)} outputs for a batch of {len(batch_prompts)} prompts"
                )
            for offset, req_out in enumerate(batch_outputs):
                text = ""
                try:
                    text = str(req_out.outputs[0].text)
                except Exception:
                    text = ""
                outputs_text[start + offset] = text
        generation_s = time.perf_counter() - gen_t0
        _emit_timing(args, "vllm_generate", generation_s, prompts=len(prompts), batch_size=batch_size)

    total_s = time.perf_counter() - total_t0
    _emit_timing(
        args,
        "eval_total_local",
        total_s,
        prep_s=f"{prepared['timings']['prepare_eval_rows_s']:.3f}",
        vllm_load_s=(f"{llm_init_s:.3f}" if llm_init_s else None),
        generation_s=(f"{generation_s:.3f}" if generation_s else None),
    )
    by_circuit: Dict[str, Dict[str, Any]] = {}
    overall = _new_stats()

    with out_jsonl.open("w") as f_out:
        for index, row in enumerate(rows):
            item = row["item"]
            base_text = outputs_text[2 * index] if (2 * index) < len(outputs_text) else ""
            retrieval_text = outputs_text[2 * index + 1] if (2 * index + 1) < len(outputs_text) else ""
            base_pred = _parse_choice_letter_robust(base_text)
            retrieval_pred = _parse_choice_letter_robust(retrieval_text)
            gold = item["answer"]

            base_ok = base_pred == gold
            retrieval_ok = retrieval_pred == gold

            circuit_stats = by_circuit.setdefault(circuit := item["circuit"], _new_stats())
            circuit_stats["num_eval"] += 1
            overall["num_eval"] += 1

            if not args.dry_run:
                if base_pred is None:
                    circuit_stats["baseline_parse_fail"] += 1
                    overall["baseline_parse_fail"] += 1
                if retrieval_pred is None:
                    circuit_stats["retrieval_parse_fail"] += 1
                    overall["retrieval_parse_fail"] += 1
                circuit_stats["baseline_correct"] += int(base_ok)
                circuit_stats["retrieval_correct"] += int(retrieval_ok)
                overall["baseline_correct"] += int(base_ok)
                overall["retrieval_correct"] += int(retrieval_ok)

            record = {
                "circuit": circuit,
                "reference_family": item["reference_family"],
                "question_id": item["question_id"],
                "question": item["question"],
                "golden_answer": item["golden_answer"],
                "answer": gold,
                "options": item["options"],
                "current_netlist": {
                    "netlist_loaded": bool(row["current_netlist"].get("netlist_loaded")),
                    "netlist_path": row["current_netlist"].get("netlist_path"),
                },
                "related": row["related"],
                "related_netlists": [
                    {
                        "netlist_loaded": bool(net.get("netlist_loaded")),
                        "netlist_path": net.get("netlist_path"),
                        "netlist_num_chars": len(net["netlist_text"]) if net.get("netlist_text") else 0,
                    }
                    for net in row["related_netlists"]
                ],
                "baseline": {
                    "pred": base_pred,
                    "raw": base_text,
                    "correct": bool(base_ok) if base_pred is not None else False,
                },
                "retrieval": {
                    "pred": retrieval_pred,
                    "raw": retrieval_text,
                    "correct": bool(retrieval_ok) if retrieval_pred is not None else False,
                },
                "baseline_prompt": row["baseline_prompt"] if args.save_prompts else None,
                "retrieval_prompt": row["retrieval_prompt"] if args.save_prompts else None,
                "source_path": item["source_path"],
            }
            f_out.write(json.dumps(record) + "\n")

    summary: Dict[str, Any] = {
        "mode": "eval_filtered_qa_vllm_local",
        "dry_run": bool(args.dry_run),
        "filtered_qa_root": str(prepared["filtered_root"]),
        "circuits": list(args.circuits or SUPPORTED_CIRCUITS),
        "embeddings_json": str(embeddings_path),
        "embedding_mode": str(args.gnn_embedding_mode),
        "num_reference": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "k": int(args.k),
        "related_mode": str(args.related_mode),
        "kg_max_chars": int(args.kg_max_chars),
        "netlist_max_chars": int(args.netlist_max_chars),
        "limit_per_circuit": int(args.limit_per_circuit) if args.limit_per_circuit else 0,
        "num_files": int(prepared["num_files"]),
        "num_seen": int(prepared["num_seen"]),
        "num_eval": int(overall["num_eval"]),
        "kg_missing": int(prepared["kg_missing"]),
        "netlist_missing": int(prepared["netlist_missing"]),
        "vllm_local": {
            "model": args.vllm_model,
            "model_source": str(getattr(args, "vllm_model_source", args.vllm_model)),
            "resolved_model": resolved_model,
            "profile": getattr(args, "model_profile_name", None),
            "is_reasoning_model": bool(getattr(args, "model_is_reasoning", False)),
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "dtype": args.dtype or "auto",
            "max_model_len": int(args.max_model_len) if args.max_model_len else None,
            "trust_remote_code": bool(args.trust_remote_code),
            "download_dir": args.download_dir,
            "apply_chat_template": bool(args.apply_chat_template),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p) if args.top_p is not None else None,
            "max_tokens": int(args.max_tokens),
            "batch_size": int(args.batch_size),
            "cuda_visible_devices": str(args.cuda_visible_devices or ""),
            "enforce_eager": bool(args.enforce_eager),
        },
        "timing_s": float(total_s),
        "stage_timings": {
            **prepared["timings"],
            "vllm_load_s": float(llm_init_s),
            "generation_s": float(generation_s),
            "eval_total_local_s": float(total_s),
        },
        "out_jsonl": str(out_jsonl),
        "summary_json": str(summary_path),
        "retrieved_circuits": {
            circuit: prepared["by_circuit_related"][circuit] for circuit in prepared["by_circuit_related"]
        },
    }

    if not args.dry_run:
        summary.update(_finalize_stats(overall))
        summary["by_circuit"] = {
            circuit: {
                **_finalize_stats(stats),
                "retrieved_circuits": prepared["by_circuit_related"].get(circuit, []),
            }
            for circuit, stats in sorted(by_circuit.items())
        }

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def eval_with_vllm_server(args: argparse.Namespace) -> Dict[str, Any]:
    total_t0 = time.perf_counter()
    prepared = prepare_eval_rows(args)
    repo_root = prepared["repo_root"]
    rows = prepared["rows"]
    embeddings = prepared["embeddings"]
    embeddings_path = prepared["embeddings_path"]
    resolved_model = prepared["resolved_model"]

    out_dir = repo_root / "QA_Task" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = (out_dir / out_jsonl).resolve()

    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = (out_dir / summary_path).resolve()

    by_circuit: Dict[str, Dict[str, Any]] = {}
    overall = _new_stats()

    request_total_s = 0.0
    first_request_started = False
    with out_jsonl.open("w") as f_out:
        for row in rows:
            item = row["item"]
            if not first_request_started:
                print(
                    "[QA_Task] Sending first API request "
                    f"to {args.vllm_base_url.rstrip('/')}/v1/chat/completions "
                    f"with model={resolved_model} timeout_s={float(args.timeout_s)}",
                    flush=True,
                )
                first_request_started = True
            request_t0 = time.perf_counter()
            base_resp = vllm_chat_completions(
                base_url=args.vllm_base_url,
                model=resolved_model,
                prompt=row["baseline_prompt"],
                api_key=args.vllm_api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_completion_tokens=args.max_completion_tokens,
                timeout_s=args.timeout_s,
                extra={"top_p": args.top_p} if args.top_p is not None else None,
                dry_run=bool(args.dry_run),
            )
            retrieval_resp = vllm_chat_completions(
                base_url=args.vllm_base_url,
                model=resolved_model,
                prompt=row["retrieval_prompt"],
                api_key=args.vllm_api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_completion_tokens=args.max_completion_tokens,
                timeout_s=args.timeout_s,
                extra={"top_p": args.top_p} if args.top_p is not None else None,
                dry_run=bool(args.dry_run),
            )
            request_total_s += time.perf_counter() - request_t0

            base_text = extract_text_from_chat_completion(base_resp) if not args.dry_run else ""
            retrieval_text = extract_text_from_chat_completion(retrieval_resp) if not args.dry_run else ""
            base_pred = _parse_choice_letter_robust(base_text)
            retrieval_pred = _parse_choice_letter_robust(retrieval_text)
            gold = item["answer"]

            base_ok = base_pred == gold
            retrieval_ok = retrieval_pred == gold

            circuit_stats = by_circuit.setdefault(circuit := item["circuit"], _new_stats())
            circuit_stats["num_eval"] += 1
            overall["num_eval"] += 1

            if not args.dry_run:
                if base_pred is None:
                    circuit_stats["baseline_parse_fail"] += 1
                    overall["baseline_parse_fail"] += 1
                if retrieval_pred is None:
                    circuit_stats["retrieval_parse_fail"] += 1
                    overall["retrieval_parse_fail"] += 1
                circuit_stats["baseline_correct"] += int(base_ok)
                circuit_stats["retrieval_correct"] += int(retrieval_ok)
                overall["baseline_correct"] += int(base_ok)
                overall["retrieval_correct"] += int(retrieval_ok)

            record = {
                "circuit": circuit,
                "reference_family": item["reference_family"],
                "question_id": item["question_id"],
                "question": item["question"],
                "golden_answer": item["golden_answer"],
                "answer": gold,
                "options": item["options"],
                "current_netlist": {
                    "netlist_loaded": bool(row["current_netlist"].get("netlist_loaded")),
                    "netlist_path": row["current_netlist"].get("netlist_path"),
                },
                "related": row["related"],
                "related_netlists": [
                    {
                        "netlist_loaded": bool(net.get("netlist_loaded")),
                        "netlist_path": net.get("netlist_path"),
                        "netlist_num_chars": len(net["netlist_text"]) if net.get("netlist_text") else 0,
                    }
                    for net in row["related_netlists"]
                ],
                "baseline": {
                    "pred": base_pred,
                    "raw": base_text,
                    "correct": bool(base_ok) if base_pred is not None else False,
                },
                "retrieval": {
                    "pred": retrieval_pred,
                    "raw": retrieval_text,
                    "correct": bool(retrieval_ok) if retrieval_pred is not None else False,
                },
                "baseline_prompt": row["baseline_prompt"] if args.save_prompts else None,
                "retrieval_prompt": row["retrieval_prompt"] if args.save_prompts else None,
                "source_path": item["source_path"],
            }
            f_out.write(json.dumps(record) + "\n")

            if args.dry_run and not args.limit_per_circuit:
                break

    total_s = time.perf_counter() - total_t0
    _emit_timing(
        args,
        "eval_total_server",
        total_s,
        prep_s=f"{prepared['timings']['prepare_eval_rows_s']:.3f}",
        request_total_s=f"{request_total_s:.3f}",
    )
    summary: Dict[str, Any] = {
        "mode": "eval_filtered_qa_vllm",
        "dry_run": bool(args.dry_run),
        "filtered_qa_root": str(prepared["filtered_root"]),
        "circuits": list(args.circuits or SUPPORTED_CIRCUITS),
        "embeddings_json": str(embeddings_path),
        "embedding_mode": str(args.gnn_embedding_mode),
        "num_reference": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "k": int(args.k),
        "related_mode": str(args.related_mode),
        "kg_max_chars": int(args.kg_max_chars),
        "netlist_max_chars": int(args.netlist_max_chars),
        "limit_per_circuit": int(args.limit_per_circuit) if args.limit_per_circuit else 0,
        "num_files": int(prepared["num_files"]),
        "num_seen": int(prepared["num_seen"]),
        "num_eval": int(overall["num_eval"]),
        "kg_missing": int(prepared["kg_missing"]),
        "netlist_missing": int(prepared["netlist_missing"]),
        "vllm": {
            "base_url": args.vllm_base_url,
            "model": args.vllm_model,
            "model_source": str(getattr(args, "vllm_model_source", args.vllm_model)),
            "resolved_model": resolved_model,
            "profile": getattr(args, "model_profile_name", None),
            "is_reasoning_model": bool(getattr(args, "model_is_reasoning", False)),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p) if args.top_p is not None else None,
            "max_tokens": int(args.max_tokens),
            "max_completion_tokens": int(args.max_completion_tokens) if args.max_completion_tokens is not None else None,
            "timeout_s": float(args.timeout_s),
        },
        "timing_s": float(total_s),
        "stage_timings": {
            **prepared["timings"],
            "request_total_s": float(request_total_s),
            "eval_total_server_s": float(total_s),
        },
        "out_jsonl": str(out_jsonl),
        "summary_json": str(summary_path),
        "retrieved_circuits": {
            circuit: prepared["by_circuit_related"][circuit] for circuit in prepared["by_circuit_related"]
        },
    }

    if not args.dry_run:
        summary.update(_finalize_stats(overall))
        summary["by_circuit"] = {
            circuit: {
                **_finalize_stats(stats),
                "retrieved_circuits": prepared["by_circuit_related"].get(circuit, []),
            }
            for circuit, stats in sorted(by_circuit.items())
        }

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_api_probe(args: argparse.Namespace) -> None:
    probe_prompt = (
        "Return exactly one uppercase letter A and nothing else."
    )
    resolved_model = str(getattr(args, "resolved_vllm_model", "") or resolve_vllm_model_path(args.vllm_model))
    print(
        "[QA_Task] API probe "
        f"target={args.vllm_base_url.rstrip('/')}/v1/chat/completions model={resolved_model} timeout_s={float(args.timeout_s)}",
        flush=True,
    )
    t0 = time.perf_counter()
    resp = vllm_chat_completions(
        base_url=args.vllm_base_url,
        model=resolved_model,
        prompt=probe_prompt,
        api_key=args.vllm_api_key,
        temperature=args.temperature,
        max_tokens=max(1, int(args.max_tokens)),
        max_completion_tokens=args.max_completion_tokens,
        timeout_s=args.timeout_s,
        extra={"top_p": args.top_p} if args.top_p is not None else None,
        dry_run=False,
    )
    elapsed = time.perf_counter() - t0
    text = extract_text_from_chat_completion(resp)
    print(f"[QA_Task] API probe completed in {elapsed:.3f}s", flush=True)
    print(f"[QA_Task] API probe response: {text!r}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate filtered QA pairs with and without SG-based retrieval.")
    ap.add_argument("--eval_vllm", action="store_true", help="Run evaluation through a running vLLM server.")
    ap.add_argument("--eval_vllm_local", action="store_true", help="Run evaluation using in-process vLLM.")

    ap.add_argument("--filtered_qa_root", type=str, default="QA_Task/filtered_QA")
    ap.add_argument("--circuits", nargs="+", default=SUPPORTED_CIRCUITS, choices=SUPPORTED_CIRCUITS)
    ap.add_argument("--limit_per_circuit", type=int, default=0, help="Limit QA items per circuit (0 = no limit).")
    ap.add_argument("--out_jsonl", type=str, default="filtered_qa_eval.jsonl")
    ap.add_argument("--summary_json", type=str, default="filtered_qa_summary.json")
    ap.add_argument("--save_prompts", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument("--k", type=int, default=2, help="How many SG-retrieved circuits to include.")
    ap.add_argument("--related_mode", type=str, default="topk", choices=["topk", "bottomk"])
    ap.add_argument("--kg_max_chars", type=int, default=12000)
    ap.add_argument("--netlist_max_chars", type=int, default=_DEFAULT_NETLIST_MAX_CHARS)
    ap.add_argument("--gnn_embedding_mode", type=str, default="sg", choices=["sg", "skg"])
    ap.add_argument("--embeddings_json", type=str, default="umap_results_full/gnn_embeddings.json")
    ap.add_argument("--gnn_checkpoint", type=str, default=None)
    ap.add_argument("--gnn_device", type=str, default="cpu")

    ap.add_argument("--vllm_base_url", type=str, default="http://localhost:8000")
    ap.add_argument("--vllm_model", type=str, default="")
    ap.add_argument(
        "--vllm_api_key",
        type=str,
        default=(os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"),
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help="Completion token budget for APIs/models that use max_completion_tokens, such as gpt-5.*",
    )
    ap.add_argument("--timeout_s", type=float, default=120.0)
    ap.add_argument(
        "--api_probe_only",
        action="store_true",
        help="Send one minimal OpenAI-compatible chat request and exit.",
    )

    ap.add_argument("--cuda_visible_devices", type=str, default="6")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    ap.add_argument("--dtype", type=str, default="")
    ap.add_argument("--max_model_len", type=int, default=0)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--download_dir", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--apply_chat_template", action="store_true")
    ap.add_argument(
        "--vllm_worker_multiproc_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="vLLM worker multiprocessing method for local mode. 'spawn' avoids fork-after-Torch deadlocks.",
    )
    ap.add_argument("--enforce_eager", action="store_true", help="Skip vLLM compile/cudagraph startup to reduce initialization latency.")
    ap.add_argument("--log_timings", action="store_true", help="Print stage-level timing diagnostics.")

    args = ap.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    if not args.eval_vllm and not args.eval_vllm_local:
        ap.error("Specify one of: --eval_vllm or --eval_vllm_local.")
    if not args.vllm_model:
        ap.error("Both eval modes require --vllm_model (HF id, local model path, or served model alias).")
    apply_model_profile_defaults(args)
    if args.max_model_len == 0:
        args.max_model_len = None

    if args.api_probe_only:
        run_api_probe(args)
        return

    if args.eval_vllm_local:
        summary = eval_with_vllm_local(args)
    else:
        summary = eval_with_vllm_server(args)

    if args.dry_run:
        print(f"[QA_Task] Dry-run wrote: {summary['out_jsonl']}")
        print(f"[QA_Task] Summary: {summary['summary_json']}")
        return

    per_circuit_lines: List[str] = []
    by_circuit = summary.get("by_circuit") or {}
    for circuit in list(args.circuits or SUPPORTED_CIRCUITS):
        circuit_summary = by_circuit.get(circuit)
        if not isinstance(circuit_summary, dict):
            continue
        per_circuit_lines.append(
            "[QA_Task] "
            f"{circuit}: baseline_acc={circuit_summary.get('baseline_accuracy')} "
            f"retrieval_acc={circuit_summary.get('retrieval_accuracy')} "
            f"(n={circuit_summary.get('num_eval')})"
        )

    per_circuit_block = ""
    if per_circuit_lines:
        per_circuit_block = "\n".join(per_circuit_lines) + "\n"

    print(
        "[QA_Task] "
        f"baseline_acc={summary.get('baseline_accuracy')} retrieval_acc={summary.get('retrieval_accuracy')} "
        f"gain={summary.get('accuracy_gain')} (n={summary.get('num_eval')})\n"
        f"{per_circuit_block}"
        f"[QA_Task] Details: {summary['out_jsonl']}\n"
        f"[QA_Task] Summary: {summary['summary_json']}"
    )


if __name__ == "__main__":
    main()