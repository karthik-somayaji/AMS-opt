import argparse
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple


_TAG_SUFFIXES = {
    "directly-proportional": "-directly-proportional",
    "inversely-proportional": "-inversely-proportional",
    "trade-off": "-trade-off",
    "ambiguous": "-ambiguous",
}


def _split_tag(node_id: str) -> Tuple[str, Optional[str]]:
    for tag, suffix in _TAG_SUFFIXES.items():
        if node_id.endswith(suffix):
            return node_id[: -len(suffix)], tag
    return node_id, None


def _stable_shuffle(items: List[str], *, key: str) -> List[str]:
    # Deterministic shuffle based on md5(key).
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items


def _make_options(*, question: str, golden: str, distractors: List[str]) -> Dict[str, str]:
    choices = [golden] + list(distractors)
    # Ensure exactly 4 unique options.
    dedup = []
    seen = set()
    for c in choices:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    if len(dedup) < 4:
        raise ValueError(f"Not enough unique options for question: {question}")
    dedup = dedup[:4]

    shuffled = _stable_shuffle(dedup, key=question)
    return {"A": shuffled[0], "B": shuffled[1], "C": shuffled[2], "D": shuffled[3]}


def _qa_direction(*, subject: str, outcome: str, direction: str, extra_note: Optional[str] = None) -> Dict[str, Any]:
    # direction: "increases" or "decreases"
    note = f" {extra_note}" if extra_note else ""
    question = f"If {subject} increases, what happens to {outcome}{note}?"
    golden = direction
    options = _make_options(
        question=question,
        golden=golden,
        distractors=["increases", "decreases", "no change", "ambiguous"],
    )
    # Make the sentence factual and simple.
    verb = "increases" if direction == "increases" else "decreases"
    sentence = f"Increasing {subject} {verb} {outcome}" + (f" {extra_note}" if extra_note else "") + "."
    return {"question": question, "golden_answer": golden, "options": options, "sentence": sentence}


def _qa_relation_type(*, a: str, b: str, relation: str) -> Dict[str, Any]:
    question = f"What is the relationship between {a} and {b} in the KG?"
    golden = relation
    options = _make_options(
        question=question,
        golden=golden,
        distractors=["directly proportional", "inversely proportional", "trade-off", "ambiguous"],
    )
    sentence = f"The relationship between {a} and {b} is {relation}."
    return {"question": question, "golden_answer": golden, "options": options, "sentence": sentence}


def _qa_tradeoff(*, a: str, b: str, all_metrics: List[str]) -> Dict[str, Any]:
    question = f"{a} has a trade-off with which performance metric?"
    distractors = [m for m in sorted(all_metrics) if m != b]
    options = _make_options(question=question, golden=b, distractors=distractors[:3])
    sentence = f"{a} trades off with {b}."
    return {"question": question, "golden_answer": b, "options": options, "sentence": sentence}


def _qa_param_substructure(*, param: str, sub: str, all_subs: List[str]) -> Dict[str, Any]:
    question = f"Which substructure is {param} associated with in the circuit KG?"
    distractors = [s for s in sorted(all_subs) if s != sub]
    options = _make_options(question=question, golden=sub, distractors=distractors[:3])
    sentence = f"{param} is associated with {sub}."
    return {"question": question, "golden_answer": sub, "options": options, "sentence": sentence}


def _qa_substructure_performance(*, sub: str, metric: str, all_metrics: List[str]) -> Dict[str, Any]:
    question = f"Which performance metric is connected to the {sub} in the KG?"
    distractors = [m for m in sorted(all_metrics) if m != metric]
    options = _make_options(question=question, golden=metric, distractors=distractors[:3])
    sentence = f"{sub} connects to {metric}."
    return {"question": question, "golden_answer": metric, "options": options, "sentence": sentence}


def _qa_id(*, circuit: str, question: str) -> str:
    digest = hashlib.md5(f"{circuit}::{question}".encode("utf-8")).hexdigest()[:10]
    return f"{circuit}_{digest}"


def generate_raw_qa(fun_updated_path: str, *, circuit: str) -> List[Dict[str, Any]]:
    with open(fun_updated_path, "r") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    links = graph.get("links", [])

    node_type: Dict[str, str] = {}
    for n in nodes:
        node_type[str(n.get("id"))] = str(n.get("type"))

    # Base metrics/substructures for distractor sets
    base_metrics = sorted(
        [nid for nid, t in node_type.items() if t == "performance" and _split_tag(nid)[1] is None]
    )
    base_subs = sorted([nid for nid, t in node_type.items() if t == "substructure"])

    qa: List[Dict[str, Any]] = []

    for e in links:
        src = str(e.get("source"))
        tgt = str(e.get("target"))

        src_type = node_type.get(src)
        tgt_type = node_type.get(tgt)

        src_base, src_tag = _split_tag(src)
        tgt_base, tgt_tag = _split_tag(tgt)

        # Parameter -> substructure association
        if src_type == "parameter" and tgt_type == "substructure" and src_tag is None and tgt_tag is None:
            qa.append(_qa_param_substructure(param=src_base, sub=tgt_base, all_subs=base_subs))
            continue

        # Substructure -> performance connection
        if src_type == "substructure" and tgt_type == "performance" and src_tag is None and tgt_tag is None:
            if tgt_base in base_metrics:
                qa.append(_qa_substructure_performance(sub=src_base, metric=tgt_base, all_metrics=base_metrics))
            continue

        # Trade-off between performance metrics (tagged)
        if src_tag == "trade-off" and tgt_tag == "trade-off" and src_base in base_metrics and tgt_base in base_metrics:
            qa.append(_qa_tradeoff(a=src_base, b=tgt_base, all_metrics=base_metrics))
            continue

        # Ambiguous relationship between metrics (tagged)
        if src_tag == "ambiguous" and tgt_tag == "ambiguous" and src_base in base_metrics and tgt_base in base_metrics:
            qa.append(_qa_relation_type(a=src_base, b=tgt_base, relation="ambiguous"))
            continue

        # Direct proportionality between metrics (tagged)
        if src_tag == "directly-proportional" and tgt_tag == "directly-proportional" and src_base in base_metrics and tgt_base in base_metrics:
            qa.append(_qa_direction(subject=src_base, outcome=tgt_base, direction="increases"))
            continue

        # Parameter proportionality -> metric proportionality
        if src_tag in {"directly-proportional", "inversely-proportional"} and tgt_tag in {
            "directly-proportional",
            "trade-off",
        } and tgt_base in base_metrics:
            direction = "increases" if src_tag == "directly-proportional" else "decreases"
            extra_note = "(trade-off)" if tgt_tag == "trade-off" else None
            qa.append(_qa_direction(subject=src_base, outcome=tgt_base, direction=direction, extra_note=extra_note))
            continue

        # Otherwise ignore meta-links like Gain -> Gain-trade-off, param -> param-directly-proportional, etc.

    # Deduplicate by question text (some graphs may contain duplicates)
    seen_q = set()
    unique = []
    for item in qa:
        q = item["question"]
        if q in seen_q:
            continue
        seen_q.add(q)
        item = dict(item)
        item["id"] = _qa_id(circuit=circuit, question=q)
        unique.append(item)

    return unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fun_updated", required=True, help="Path to fun_updated.json")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument(
        "--circuit",
        default=None,
        help="Circuit name used in QA ids (e.g., amp2). Default: inferred from fun_updated path.",
    )
    args = ap.parse_args()

    circuit = args.circuit
    if not circuit:
        # Heuristic: LLMBO/<ckt>_ati_new/fun_updated.json -> <ckt>
        parts = os.path.normpath(args.fun_updated).split(os.sep)
        circuit = "unknown"
        if len(parts) >= 2 and parts[-1] == "fun_updated.json":
            parent = parts[-2]
            if parent.endswith("_ati_new"):
                circuit = parent[: -len("_ati_new")]
            else:
                circuit = parent

    qa = generate_raw_qa(args.fun_updated, circuit=str(circuit))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(qa, f, indent=2)

    print(f"Wrote {len(qa)} QA entries to {args.out}")


if __name__ == "__main__":
    main()
