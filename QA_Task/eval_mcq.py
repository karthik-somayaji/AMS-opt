#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _optional_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def configure_vllm_worker_multiproc_method(requested_method: Optional[str] = None) -> str:
    method = str(requested_method or os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", "spawn")).strip().lower()
    if method:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = method
    return method


@dataclass(frozen=True)
class CircuitRef:
    circuit_id: str
    family: str


def _repo_root() -> Path:
    # `QA_Task/` lives at repo root.
    return Path(__file__).resolve().parents[1]


def load_reference_embeddings(path: Path) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """Load reference raw GNN embeddings JSON.

    Expected format:
      {"embeddings": [[...], ...], "metadata": [{"circuit_id": "..", "family": ".."}, ...]}
    """
    with path.open("r") as f:
        obj = json.load(f)

    embeddings = obj.get("embeddings")
    metadata = obj.get("metadata")
    if not isinstance(embeddings, list) or not isinstance(metadata, list) or len(embeddings) != len(metadata):
        raise ValueError(f"Invalid embeddings JSON structure: {path}")

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array; got shape {emb.shape} from {path}")

    meta: List[Dict[str, str]] = []
    for m in metadata:
        cid = str(m.get("circuit_id"))
        fam = str(m.get("family"))
        meta.append({"circuit_id": cid, "family": fam})

    return emb, meta


def build_index(metadata: List[Dict[str, str]]) -> Dict[Tuple[str, str], int]:
    idx: Dict[Tuple[str, str], int] = {}
    for i, m in enumerate(metadata):
        key = (m["family"], m["circuit_id"])
        if key not in idx:
            idx[key] = i
    return idx


def cosine_topk(
    embeddings: np.ndarray,
    metadata: List[Dict[str, str]],
    query_family: str,
    query_circuit_id: str,
    k: int,
) -> List[Dict[str, Any]]:
    index = build_index(metadata)
    key = (query_family, str(query_circuit_id))
    if key not in index:
        raise KeyError(
            f"Query circuit not found in reference metadata: family={query_family!r} circuit_id={query_circuit_id!r}. "
            "For now, run --sanity with a circuit_id present in umap_results_full/gnn_embeddings.json."
        )

    q_idx = index[key]
    q = embeddings[q_idx]

    # Cosine similarity against all
    denom = (np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q) + 1e-8) + 1e-8)
    sims = (embeddings @ q) / denom

    # Exclude itself
    sims[q_idx] = -np.inf

    k = int(k)
    k = max(1, k)
    top_idx = np.argpartition(-sims, kth=min(k, sims.shape[0] - 1))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    out: List[Dict[str, Any]] = []
    for i in top_idx.tolist():
        m = metadata[i]
        out.append(
            {
                "family": m["family"],
                "circuit_id": m["circuit_id"],
                "similarity": float(sims[i]),
            }
        )
    return out


def load_fun_graph(repo_root: Path, family: str, circuit_id: str) -> Optional[str]:
    p = repo_root / "netlists" / family / str(circuit_id) / "fun_graph.json"
    if not p.exists():
        return None
    return p.read_text()


def iter_mcq_files(mcq_root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (family, path) for each MCQ json under mcq_root/<family>/*.json."""
    if not mcq_root.exists():
        return
    for fam_dir in sorted([p for p in mcq_root.iterdir() if p.is_dir()]):
        family = fam_dir.name
        for p in sorted(fam_dir.glob("*.json")):
            yield family, p


def load_mcq(path: Path, family: str) -> Dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"MCQ file must be a JSON object: {path}")

    question = obj.get("question")
    options = obj.get("options")
    answer = obj.get("answer")
    circuit_id = obj.get("circuit_id")
    netlist = obj.get("netlist")

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing/invalid 'question' in {path}")
    if not isinstance(options, dict) or not all(k in options for k in ["A", "B", "C", "D"]):
        raise ValueError(f"Missing/invalid 'options' (must contain A/B/C/D) in {path}")
    if not isinstance(answer, str) or answer.strip() not in {"A", "B", "C", "D"}:
        raise ValueError(f"Missing/invalid 'answer' (A/B/C/D) in {path}")
    if circuit_id is None:
        raise ValueError(f"Missing 'circuit_id' in {path}")
    if not isinstance(netlist, str) or not netlist.strip():
        raise ValueError(f"Missing/invalid 'netlist' in {path}")

    return {
        "family": family,
        "circuit_id": str(circuit_id),
        "question": question.strip(),
        "options": {k: str(options[k]) for k in ["A", "B", "C", "D"]},
        "answer": answer.strip(),
        "netlist": netlist.strip(),
        "source_path": str(path),
    }


def build_baseline_prompt(mcq: Dict[str, Any]) -> str:
    opts = mcq["options"]
    return (
        "You are an analog circuit expert. Answer the multiple-choice question using the given netlist.\n"
        "Return ONLY the option letter (A, B, C, or D). No explanation.\n\n"
        "NETLIST:\n"
        "```\n"
        f"{mcq['netlist']}\n"
        "```\n\n"
        f"QUESTION: {mcq['question']}\n\n"
        "OPTIONS:\n"
        f"A) {opts['A']}\n"
        f"B) {opts['B']}\n"
        f"C) {opts['C']}\n"
        f"D) {opts['D']}\n"
    )


def build_augmented_prompt(
    mcq: Dict[str, Any],
    related: List[Dict[str, Any]],
    related_kgs: List[Dict[str, Any]],
) -> str:
    baseline = build_baseline_prompt(mcq)

    blocks: List[str] = []
    blocks.append(
        "\nRELATED CIRCUIT KNOWLEDGE GRAPHS (may be helpful prior knowledge):\n"
        "Use them only as supporting context.\n"
    )
    for r, kg in zip(related, related_kgs):
        header = f"[RELATED_CIRCUIT family={r['family']} id={r['circuit_id']} similarity={r['similarity']:.4f}]"
        if kg.get("kg_text"):
            blocks.append(header + "\n```json\n" + kg["kg_text"] + "\n```\n")
        else:
            blocks.append(header + "\n(KG missing)\n")

    return baseline + "\n".join(blocks)


_ANSWER_RE = re.compile(r"\b([ABCD])\b")


def parse_choice_letter(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = text.strip().upper()
    if t in {"A", "B", "C", "D"}:
        return t
    m = _ANSWER_RE.search(t)
    if m:
        return m.group(1)
    return None


def vllm_chat_completions(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    max_tokens: int = 8,
    timeout_s: float = 120.0,
    extra: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Call a running vLLM OpenAI-compatible server.

    vLLM typically exposes OpenAI-compatible endpoints such as:
      POST {base_url}/v1/chat/completions
    """

    requests = _optional_import("requests")
    if requests is None and not dry_run:
        raise RuntimeError("Missing dependency 'requests'. Install with: pip install requests")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
    }
    normalized_model = str(model or "").strip().lower()
    if normalized_model.startswith("gpt-5"):
        payload["max_completion_tokens"] = int(max_tokens)
    else:
        payload["max_tokens"] = int(max_tokens)
    if extra:
        payload.update(extra)

    if dry_run:
        return {
            "dry_run": True,
            "url": url,
            "payload": payload,
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_timeout = (min(10.0, max(1.0, float(timeout_s))), max(1.0, float(timeout_s)))
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
    except Exception as exc:
        raise RuntimeError(
            f"Chat completions request failed before receiving a response for {url}: {exc}"
        ) from exc
    if not resp.ok:
        response_text = ""
        try:
            response_text = resp.text.strip()
        except Exception:
            response_text = ""
        detail = f": {response_text}" if response_text else ""
        raise RuntimeError(f"Chat completions request failed with HTTP {resp.status_code} for {url}{detail}")
    data = resp.json()
    return data


def extract_text_from_chat_completion(resp: Dict[str, Any]) -> str:
    try:
        return str(resp["choices"][0]["message"]["content"])
    except Exception:
        return ""


_TOKENIZER_CACHE: Dict[Tuple[str, bool, Optional[str]], Any] = {}


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
                if preferred.is_dir():
                    return str(preferred.resolve())

            snapshot_dirs = sorted(
                [path for path in snapshots_dir.iterdir() if path.is_dir()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if snapshot_dirs:
                return str(snapshot_dirs[0].resolve())

    return str(candidate.resolve())


def _get_chat_tokenizer(
    *,
    model_id_or_path: str,
    trust_remote_code: bool,
    download_dir: Optional[str],
):
    key = (model_id_or_path, bool(trust_remote_code), download_dir)
    tok = _TOKENIZER_CACHE.get(key)
    if tok is not None:
        return tok

    transformers = _optional_import("transformers")
    if transformers is None:
        raise RuntimeError(
            "--apply_chat_template requires 'transformers'. Install with: pip install transformers"
        )

    AutoTokenizer = getattr(transformers, "AutoTokenizer", None)
    if AutoTokenizer is None:
        raise RuntimeError("transformers.AutoTokenizer not available; cannot apply chat template")

    tok = AutoTokenizer.from_pretrained(
        model_id_or_path,
        trust_remote_code=bool(trust_remote_code),
        cache_dir=download_dir,
    )
    _TOKENIZER_CACHE[key] = tok
    return tok


def _maybe_apply_chat_template(
    *,
    prompt: str,
    model_id_or_path: str,
    trust_remote_code: bool,
    download_dir: Optional[str],
    dry_run: bool,
) -> str:
    if not prompt:
        return prompt
    if dry_run:
        return prompt

    tokenizer = _get_chat_tokenizer(
        model_id_or_path=model_id_or_path,
        trust_remote_code=bool(trust_remote_code),
        download_dir=download_dir,
    )
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        raise RuntimeError(
            "Tokenizer does not support apply_chat_template; run without --apply_chat_template or use a chat-capable tokenizer."
        )

    messages = [{"role": "user", "content": prompt}]
    return str(apply(messages, tokenize=False, add_generation_prompt=True))


def build_prompts(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = _repo_root()
    embeddings_path = (repo_root / args.embeddings_json).resolve() if not os.path.isabs(args.embeddings_json) else Path(args.embeddings_json)
    embeddings, metadata = load_reference_embeddings(embeddings_path)

    mcq_root = Path(args.mcq_root)
    if not mcq_root.is_absolute():
        mcq_root = (repo_root / mcq_root).resolve()

    out_dir = repo_root / "QA_Task" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = (out_dir / out_jsonl).resolve()

    num_seen = 0
    num_written = 0
    num_skipped_missing_embedding = 0
    num_kg_missing = 0

    with out_jsonl.open("w") as f_out:
        for family, p in iter_mcq_files(mcq_root):
            num_seen += 1
            if args.limit and num_seen > int(args.limit):
                break

            mcq = load_mcq(p, family=family)

            if args.netlist_max_chars and int(args.netlist_max_chars) > 0:
                nmax = int(args.netlist_max_chars)
                if len(mcq["netlist"]) > nmax:
                    mcq = dict(mcq)
                    mcq["netlist"] = mcq["netlist"][:nmax] + "\n\n(TRUNCATED NETLIST)\n"

            try:
                related = cosine_topk(
                    embeddings=embeddings,
                    metadata=metadata,
                    query_family=mcq["family"],
                    query_circuit_id=mcq["circuit_id"],
                    k=args.k,
                )
            except KeyError:
                num_skipped_missing_embedding += 1
                continue

            related_kgs: List[Dict[str, Any]] = []
            for r in related:
                kg_txt = load_fun_graph(repo_root, r["family"], r["circuit_id"])
                if kg_txt is None:
                    num_kg_missing += 1
                if kg_txt is not None and args.kg_max_chars and len(kg_txt) > int(args.kg_max_chars):
                    kg_txt = kg_txt[: int(args.kg_max_chars)] + "\n\n(TRUNCATED)\n"
                related_kgs.append({
                    "kg_loaded": kg_txt is not None,
                    "kg_text": kg_txt,
                    "kg_path": str(repo_root / "netlists" / r["family"] / str(r["circuit_id"]) / "fun_graph.json"),
                })

            baseline_prompt = build_baseline_prompt(mcq)
            augmented_prompt = build_augmented_prompt(mcq, related=related, related_kgs=related_kgs)

            rec = {
                "family": mcq["family"],
                "circuit_id": mcq["circuit_id"],
                "answer": mcq["answer"],
                "question": mcq["question"],
                "options": mcq["options"],
                "source_path": mcq["source_path"],
                "related": related,
                "related_kgs": [{"kg_loaded": x["kg_loaded"], "kg_path": x["kg_path"], "kg_num_chars": len(x["kg_text"]) if x["kg_text"] else 0} for x in related_kgs],
                "baseline_prompt": baseline_prompt,
                "augmented_prompt": augmented_prompt,
            }
            f_out.write(json.dumps(rec) + "\n")
            num_written += 1

    return {
        "mode": "build_prompts",
        "mcq_root": str(mcq_root),
        "embeddings_json": str(embeddings_path),
        "num_reference": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "k": int(args.k),
        "kg_max_chars": int(args.kg_max_chars) if args.kg_max_chars else None,
        "limit": int(args.limit) if args.limit else None,
        "num_seen": int(num_seen),
        "num_written": int(num_written),
        "num_skipped_missing_embedding": int(num_skipped_missing_embedding),
        "num_kg_missing": int(num_kg_missing),
        "out_jsonl": str(out_jsonl),
    }


def eval_with_vllm(args: argparse.Namespace) -> Dict[str, Any]:
    """Run baseline vs augmented evaluation through a running vLLM server."""

    tqdm_mod = _optional_import("tqdm")
    tqdm = getattr(tqdm_mod, "tqdm", None) if tqdm_mod else None

    repo_root = _repo_root()
    embeddings_path = (repo_root / args.embeddings_json).resolve() if not os.path.isabs(args.embeddings_json) else Path(args.embeddings_json)
    embeddings, metadata = load_reference_embeddings(embeddings_path)

    mcq_root = Path(args.mcq_root)
    if not mcq_root.is_absolute():
        mcq_root = (repo_root / mcq_root).resolve()

    out_dir = repo_root / "QA_Task" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = (out_dir / out_jsonl).resolve()

    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = (out_dir / summary_path).resolve()

    resolved_model = resolve_vllm_model_path(args.vllm_model)

    items: List[Tuple[str, Path]] = list(iter_mcq_files(mcq_root))
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]

    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="QA_Task eval", total=len(items))

    num_seen = 0
    num_eval = 0
    skipped_missing_embedding = 0
    kg_missing = 0

    baseline_correct = 0
    augmented_correct = 0
    baseline_parse_fail = 0
    augmented_parse_fail = 0

    t0 = time.time()
    with out_jsonl.open("w") as f_out:
        for family, p in iterator:
            num_seen += 1
            mcq = load_mcq(p, family=family)

            if args.netlist_max_chars and int(args.netlist_max_chars) > 0:
                nmax = int(args.netlist_max_chars)
                if len(mcq["netlist"]) > nmax:
                    mcq = dict(mcq)
                    mcq["netlist"] = mcq["netlist"][:nmax] + "\n\n(TRUNCATED NETLIST)\n"

            try:
                related = cosine_topk(
                    embeddings=embeddings,
                    metadata=metadata,
                    query_family=mcq["family"],
                    query_circuit_id=mcq["circuit_id"],
                    k=args.k,
                )
            except KeyError:
                skipped_missing_embedding += 1
                continue

            related_kgs: List[Dict[str, Any]] = []
            for r in related:
                kg_txt = load_fun_graph(repo_root, r["family"], r["circuit_id"])
                if kg_txt is None:
                    kg_missing += 1
                if kg_txt is not None and args.kg_max_chars and int(args.kg_max_chars) > 0 and len(kg_txt) > int(args.kg_max_chars):
                    kg_txt = kg_txt[: int(args.kg_max_chars)] + "\n\n(TRUNCATED)\n"
                related_kgs.append({
                    "kg_loaded": kg_txt is not None,
                    "kg_text": kg_txt,
                    "kg_path": str(repo_root / "netlists" / r["family"] / str(r["circuit_id"]) / "fun_graph.json"),
                })

            baseline_prompt = build_baseline_prompt(mcq)
            augmented_prompt = build_augmented_prompt(mcq, related=related, related_kgs=related_kgs)

            # Call vLLM server
            resp_base = vllm_chat_completions(
                base_url=args.vllm_base_url,
                model=resolved_model,
                prompt=baseline_prompt,
                api_key=args.vllm_api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                extra={"top_p": args.top_p} if args.top_p is not None else None,
                dry_run=bool(args.dry_run),
            )

            resp_aug = vllm_chat_completions(
                base_url=args.vllm_base_url,
                model=resolved_model,
                prompt=augmented_prompt,
                api_key=args.vllm_api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout_s,
                extra={"top_p": args.top_p} if args.top_p is not None else None,
                dry_run=bool(args.dry_run),
            )

            base_text = extract_text_from_chat_completion(resp_base) if not args.dry_run else ""
            aug_text = extract_text_from_chat_completion(resp_aug) if not args.dry_run else ""

            base_pred = parse_choice_letter(base_text)
            aug_pred = parse_choice_letter(aug_text)

            gold = mcq["answer"]
            base_ok = (base_pred == gold)
            aug_ok = (aug_pred == gold)

            if not args.dry_run:
                if base_pred is None:
                    baseline_parse_fail += 1
                if aug_pred is None:
                    augmented_parse_fail += 1
                baseline_correct += int(base_ok)
                augmented_correct += int(aug_ok)

            num_eval += 1

            rec = {
                "family": mcq["family"],
                "circuit_id": mcq["circuit_id"],
                "answer": gold,
                "related": related,
                "baseline": {
                    "pred": base_pred,
                    "raw": base_text,
                    "correct": bool(base_ok) if base_pred is not None else False,
                },
                "augmented": {
                    "pred": aug_pred,
                    "raw": aug_text,
                    "correct": bool(aug_ok) if aug_pred is not None else False,
                },
                "baseline_prompt": baseline_prompt if args.save_prompts else None,
                "augmented_prompt": augmented_prompt if args.save_prompts else None,
                "source_path": mcq["source_path"],
            }
            f_out.write(json.dumps(rec) + "\n")

            if args.dry_run:
                # Only do one item in dry-run unless user explicitly sets limit.
                if args.limit == 0:
                    break

    dt = time.time() - t0
    summary: Dict[str, Any] = {
        "mode": "eval_vllm",
        "dry_run": bool(args.dry_run),
        "mcq_root": str(mcq_root),
        "embeddings_json": str(embeddings_path),
        "num_reference": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "k": int(args.k),
        "kg_max_chars": int(args.kg_max_chars) if args.kg_max_chars else None,
        "limit": int(args.limit) if args.limit else 0,
        "num_seen": int(num_seen),
        "num_eval": int(num_eval),
        "skipped_missing_embedding": int(skipped_missing_embedding),
        "kg_missing": int(kg_missing),
        "vllm": {
            "base_url": args.vllm_base_url,
            "model": args.vllm_model,
            "resolved_model": resolved_model,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p) if args.top_p is not None else None,
            "max_tokens": int(args.max_tokens),
            "timeout_s": float(args.timeout_s),
        },
        "timing_s": float(dt),
        "out_jsonl": str(out_jsonl),
        "summary_json": str(summary_path),
    }

    if not args.dry_run and num_eval > 0:
        summary.update(
            {
                "baseline_accuracy": float(baseline_correct / num_eval),
                "augmented_accuracy": float(augmented_correct / num_eval),
                "accuracy_gain": float((augmented_correct - baseline_correct) / num_eval),
                "baseline_correct": int(baseline_correct),
                "augmented_correct": int(augmented_correct),
                "baseline_parse_fail": int(baseline_parse_fail),
                "augmented_parse_fail": int(augmented_parse_fail),
            }
        )

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def eval_with_vllm_local(args: argparse.Namespace) -> Dict[str, Any]:
    """Run baseline vs augmented evaluation using in-process vLLM (no server)."""

    tqdm_mod = _optional_import("tqdm")
    tqdm = getattr(tqdm_mod, "tqdm", None) if tqdm_mod else None

    repo_root = _repo_root()
    embeddings_path = (repo_root / args.embeddings_json).resolve() if not os.path.isabs(args.embeddings_json) else Path(args.embeddings_json)
    embeddings, metadata = load_reference_embeddings(embeddings_path)

    mcq_root = Path(args.mcq_root)
    if not mcq_root.is_absolute():
        mcq_root = (repo_root / mcq_root).resolve()

    out_dir = repo_root / "QA_Task" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = (out_dir / out_jsonl).resolve()

    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = (out_dir / summary_path).resolve()

    resolved_model = resolve_vllm_model_path(args.vllm_model)

    items: List[Tuple[str, Path]] = list(iter_mcq_files(mcq_root))
    if args.limit and int(args.limit) > 0:
        items = items[: int(args.limit)]

    iterator = items
    if tqdm is not None:
        iterator = tqdm(items, desc="QA_Task eval (local vLLM)", total=len(items))

    num_seen = 0
    num_eval = 0
    skipped_missing_embedding = 0
    kg_missing = 0

    baseline_correct = 0
    augmented_correct = 0
    baseline_parse_fail = 0
    augmented_parse_fail = 0

    # Prepare all prompts first so we can run vLLM in batches.
    eval_rows: List[Dict[str, Any]] = []
    prompts: List[str] = []

    t0 = time.time()
    with out_jsonl.open("w") as f_out:
        for family, p in iterator:
            num_seen += 1
            mcq = load_mcq(p, family=family)

            if args.netlist_max_chars and int(args.netlist_max_chars) > 0:
                nmax = int(args.netlist_max_chars)
                if len(mcq["netlist"]) > nmax:
                    mcq = dict(mcq)
                    mcq["netlist"] = mcq["netlist"][:nmax] + "\n\n(TRUNCATED NETLIST)\n"

            try:
                related = cosine_topk(
                    embeddings=embeddings,
                    metadata=metadata,
                    query_family=mcq["family"],
                    query_circuit_id=mcq["circuit_id"],
                    k=args.k,
                )
            except KeyError:
                skipped_missing_embedding += 1
                continue

            related_kgs: List[Dict[str, Any]] = []
            for r in related:
                kg_txt = load_fun_graph(repo_root, r["family"], r["circuit_id"])
                if kg_txt is None:
                    kg_missing += 1
                if (
                    kg_txt is not None
                    and args.kg_max_chars
                    and int(args.kg_max_chars) > 0
                    and len(kg_txt) > int(args.kg_max_chars)
                ):
                    kg_txt = kg_txt[: int(args.kg_max_chars)] + "\n\n(TRUNCATED)\n"
                related_kgs.append(
                    {
                        "kg_loaded": kg_txt is not None,
                        "kg_text": kg_txt,
                        "kg_path": str(repo_root / "netlists" / r["family"] / str(r["circuit_id"]) / "fun_graph.json"),
                    }
                )

            baseline_prompt = build_baseline_prompt(mcq)
            augmented_prompt = build_augmented_prompt(mcq, related=related, related_kgs=related_kgs)

            if args.apply_chat_template:
                baseline_prompt = _maybe_apply_chat_template(
                    prompt=baseline_prompt,
                    model_id_or_path=resolved_model,
                    trust_remote_code=bool(args.trust_remote_code),
                    download_dir=args.download_dir,
                    dry_run=bool(args.dry_run),
                )
                augmented_prompt = _maybe_apply_chat_template(
                    prompt=augmented_prompt,
                    model_id_or_path=resolved_model,
                    trust_remote_code=bool(args.trust_remote_code),
                    download_dir=args.download_dir,
                    dry_run=bool(args.dry_run),
                )

            eval_rows.append(
                {
                    "mcq": mcq,
                    "related": related,
                    "baseline_prompt": baseline_prompt,
                    "augmented_prompt": augmented_prompt,
                }
            )
            prompts.append(baseline_prompt)
            prompts.append(augmented_prompt)

        # Generate outputs with vLLM
        outputs_text: List[str] = [""] * len(prompts)
        if not args.dry_run and prompts:
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
                enforce_eager=False,

            )

            sampling = SamplingParams(
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                top_p=(float(args.top_p) if args.top_p is not None else 1.0),
            )

            bs = int(args.batch_size)
            bs = max(1, bs)
            for start in range(0, len(prompts), bs):
                batch_prompts = prompts[start : start + bs]
                batch_out = llm.generate(batch_prompts, sampling)
                if len(batch_out) != len(batch_prompts):
                    raise RuntimeError(
                        f"vLLM returned {len(batch_out)} outputs for a batch of {len(batch_prompts)} prompts"
                    )
                for i, req_out in enumerate(batch_out):
                    txt = ""
                    try:
                        txt = str(req_out.outputs[0].text)
                    except Exception:
                        txt = ""
                    outputs_text[start + i] = txt

        # Write results
        for i, row in enumerate(eval_rows):
            mcq = row["mcq"]
            gold = mcq["answer"]
            base_text = outputs_text[2 * i] if (2 * i) < len(outputs_text) else ""
            aug_text = outputs_text[2 * i + 1] if (2 * i + 1) < len(outputs_text) else ""

            base_pred = parse_choice_letter(base_text)
            aug_pred = parse_choice_letter(aug_text)

            base_ok = (base_pred == gold)
            aug_ok = (aug_pred == gold)

            if not args.dry_run:
                if base_pred is None:
                    baseline_parse_fail += 1
                if aug_pred is None:
                    augmented_parse_fail += 1
                baseline_correct += int(base_ok)
                augmented_correct += int(aug_ok)

            num_eval += 1
            rec = {
                "family": mcq["family"],
                "circuit_id": mcq["circuit_id"],
                "answer": gold,
                "related": row["related"],
                "baseline": {
                    "pred": base_pred,
                    "raw": base_text,
                    "correct": bool(base_ok) if base_pred is not None else False,
                },
                "augmented": {
                    "pred": aug_pred,
                    "raw": aug_text,
                    "correct": bool(aug_ok) if aug_pred is not None else False,
                },
                "baseline_prompt": row["baseline_prompt"] if args.save_prompts else None,
                "augmented_prompt": row["augmented_prompt"] if args.save_prompts else None,
                "source_path": mcq["source_path"],
            }
            f_out.write(json.dumps(rec) + "\n")

            if args.dry_run:
                # Only do one item in dry-run unless user explicitly sets limit.
                if args.limit == 0:
                    break

    dt = time.time() - t0
    summary: Dict[str, Any] = {
        "mode": "eval_vllm_local",
        "dry_run": bool(args.dry_run),
        "mcq_root": str(mcq_root),
        "embeddings_json": str(embeddings_path),
        "num_reference": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "k": int(args.k),
        "kg_max_chars": int(args.kg_max_chars) if args.kg_max_chars else None,
        "limit": int(args.limit) if args.limit else 0,
        "num_seen": int(num_seen),
        "num_eval": int(num_eval),
        "skipped_missing_embedding": int(skipped_missing_embedding),
        "kg_missing": int(kg_missing),
        "vllm_local": {
            "model": args.vllm_model,
            "resolved_model": resolved_model,
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
        },
        "timing_s": float(dt),
        "out_jsonl": str(out_jsonl),
        "summary_json": str(summary_path),
    }

    if not args.dry_run and num_eval > 0:
        summary.update(
            {
                "baseline_accuracy": float(baseline_correct / num_eval),
                "augmented_accuracy": float(augmented_correct / num_eval),
                "accuracy_gain": float((augmented_correct - baseline_correct) / num_eval),
                "baseline_correct": int(baseline_correct),
                "augmented_correct": int(augmented_correct),
                "baseline_parse_fail": int(baseline_parse_fail),
                "augmented_parse_fail": int(augmented_parse_fail),
            }
        )

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_sanity(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = _repo_root()
    embeddings_path = (repo_root / args.embeddings_json).resolve() if not os.path.isabs(args.embeddings_json) else Path(args.embeddings_json)

    embeddings, metadata = load_reference_embeddings(embeddings_path)

    related = cosine_topk(
        embeddings=embeddings,
        metadata=metadata,
        query_family=args.target_family,
        query_circuit_id=str(args.target_id),
        k=args.k,
    )

    related_with_kg: List[Dict[str, Any]] = []
    for r in related:
        kg_txt = load_fun_graph(repo_root, r["family"], r["circuit_id"])
        r2 = dict(r)
        r2["kg_path"] = str(repo_root / "netlists" / r["family"] / str(r["circuit_id"]) / "fun_graph.json")
        r2["kg_loaded"] = kg_txt is not None
        r2["kg_num_chars"] = len(kg_txt) if kg_txt is not None else 0
        related_with_kg.append(r2)

    report = {
        "mode": "sanity",
        "query": {"family": args.target_family, "circuit_id": str(args.target_id)},
        "embeddings_json": str(embeddings_path),
        "num_reference": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "k": int(args.k),
        "related": related_with_kg,
    }

    out_dir = repo_root / "QA_Task" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sanity_{args.target_family}_{args.target_id}_k{args.k}.json"
    out_path.write_text(json.dumps(report, indent=2))
    report["out_path"] = str(out_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal embedding-similarity + KG loader for QA-task scaffolding")
    ap.add_argument("--embeddings_json", type=str, default="umap_results_full/gnn_embeddings.json")

    ap.add_argument("--build_prompts", action="store_true", help="Build baseline + KG-augmented prompts from mcq/<family>/*.json")
    ap.add_argument("--eval_vllm", action="store_true", help="Run baseline vs augmented evaluation via a running vLLM server")
    ap.add_argument("--eval_vllm_local", action="store_true", help="Run baseline vs augmented evaluation using in-process vLLM (no server)")
    ap.add_argument("--mcq_root", type=str, default="mcqs", help="Root directory containing per-family MCQ JSON files")
    ap.add_argument("--out_jsonl", type=str, default="prompts.jsonl", help="Output JSONL filename (relative to QA_Task/out/) or absolute path")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of MCQs processed (0 = no limit)")
    ap.add_argument("--kg_max_chars", type=int, default=12000, help="Truncate each related KG to at most this many characters (0 = no truncation)")
    ap.add_argument("--netlist_max_chars", type=int, default=0, help="Truncate the netlist to at most this many characters (0 = no truncation)")

    ap.add_argument("--sanity", action="store_true", help="Run a retrieval+KG-loading sanity check (no MCQs required)")
    ap.add_argument("--target_family", type=str, default="diff_amps")
    ap.add_argument("--target_id", type=str, default="69")
    ap.add_argument("--k", type=int, default=5)

    # vLLM / OpenAI-compatible client options
    ap.add_argument("--vllm_base_url", type=str, default="http://localhost:8000")
    ap.add_argument("--vllm_model", type=str, default="")
    ap.add_argument("--vllm_api_key", type=str, default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--timeout_s", type=float, default=120.0)
    ap.add_argument("--dry_run", action="store_true", help="Do not send HTTP; just exercise prompt assembly and request construction")
    ap.add_argument("--save_prompts", action="store_true", help="Include prompts in the output JSONL (larger files)")
    ap.add_argument("--summary_json", type=str, default="summary.json", help="Summary JSON filename (relative to QA_Task/out/) or absolute path")

    # vLLM local options
    ap.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="",
        help="If set, exports CUDA_VISIBLE_DEVICES before loading vLLM/torch (e.g. '6' or '6,7').",
    )
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--dtype", type=str, default="", help="vLLM dtype, e.g. 'auto', 'half', 'bfloat16'")
    ap.add_argument("--max_model_len", type=int, default=0, help="Optional max model length override (0 = default)")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--download_dir", type=str, default=None, help="Optional HF download/cache directory")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for local vLLM generation")
    ap.add_argument("--apply_chat_template", action="store_true", help="Wrap prompts using the model tokenizer chat template before generation")
    ap.add_argument(
        "--vllm_worker_multiproc_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="vLLM worker multiprocessing method for local mode. 'spawn' avoids fork-after-Torch deadlocks.",
    )

    args = ap.parse_args()

    # Must be set before vLLM/torch initializes CUDA.
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    if args.sanity:
        report = run_sanity(args)
        print(f"[QA_Task] Loaded {report['num_reference']} reference embeddings (D={report['embedding_dim']}).")
        print(f"[QA_Task] Query: family={report['query']['family']} id={report['query']['circuit_id']}")
        print(f"[QA_Task] Wrote: {report['out_path']}")
        missing = [r for r in report["related"] if not r["kg_loaded"]]
        print(f"[QA_Task] Related circuits: {len(report['related'])} (KG missing for {len(missing)})")
        for r in report["related"]:
            flag = "OK" if r["kg_loaded"] else "MISSING"
            print(f"  - {flag} family={r['family']} id={r['circuit_id']} sim={r['similarity']:.4f} chars={r['kg_num_chars']}")
        return

    if args.build_prompts:
        report = build_prompts(args)
        print(
            "[QA_Task] "
            f"Wrote {report['num_written']}/{report['num_seen']} prompts to {report['out_jsonl']} "
            f"(skipped_missing_embedding={report['num_skipped_missing_embedding']}, kg_missing={report['num_kg_missing']})."
        )
        return

    if args.eval_vllm:
        if not args.vllm_model:
            ap.error("--eval_vllm requires --vllm_model (the model name exposed by the server, or the served model alias).")
        summary = eval_with_vllm(args)
        if args.dry_run:
            print(f"[QA_Task] Dry-run wrote: {summary['out_jsonl']}")
        else:
            print(
                "[QA_Task] "
                f"baseline_acc={summary.get('baseline_accuracy')} augmented_acc={summary.get('augmented_accuracy')} "
                f"gain={summary.get('accuracy_gain')} (n={summary.get('num_eval')})\n"
                f"[QA_Task] Details: {summary['out_jsonl']}\n"
                f"[QA_Task] Summary: {summary['summary_json']}"
            )
        return

    if args.eval_vllm_local:
        if not args.vllm_model:
            ap.error("--eval_vllm_local requires --vllm_model (a local model path or a Hugging Face model id).")
        if args.max_model_len == 0:
            args.max_model_len = None
        summary = eval_with_vllm_local(args)
        if args.dry_run:
            print(f"[QA_Task] Dry-run wrote: {summary['out_jsonl']}")
        else:
            print(
                "[QA_Task] "
                f"baseline_acc={summary.get('baseline_accuracy')} augmented_acc={summary.get('augmented_accuracy')} "
                f"gain={summary.get('accuracy_gain')} (n={summary.get('num_eval')})\n"
                f"[QA_Task] Details: {summary['out_jsonl']}\n"
                f"[QA_Task] Summary: {summary['summary_json']}"
            )
        return

    ap.error("Specify one of: --sanity, --build_prompts, --eval_vllm, --eval_vllm_local.")


if __name__ == "__main__":
    main()
