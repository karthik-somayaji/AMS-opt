import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> str:
    # QA_Task/filtered_QA/filter_with_llmbo.py -> repo root is 2 levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_raw_qa_path(circuit: str) -> str:
    return os.path.join(_repo_root(), "QA_Task", "raw_QA", f"{circuit}_raw_QA.json")


def _default_filtered_out_path(circuit: str) -> str:
    return os.path.join(_repo_root(), "QA_Task", "filtered_QA", f"{circuit}_filtered_QA.json")


def _default_scores_out_path(circuit: str) -> str:
    return os.path.join(_repo_root(), "QA_Task", "filtered_QA", f"{circuit}_filtered_QA_scores.json")


def _default_init_pkl_path(circuit: str) -> str:
    return os.path.join(_repo_root(), "QA_Task", "filtered_QA", f"{circuit}_init_data.pkl")


def _analog_rep_python() -> str:
    # Match the user's known working interpreter.
    return "/home/karthik/miniconda3/envs/analog-rep/bin/python"


def _parse_final_best_fom(stdout: str) -> float:
    m = re.search(r"FINAL_BEST_FOM=([0-9eE+\-\.]+)", stdout)
    if not m:
        raise RuntimeError("Could not find FINAL_BEST_FOM in output")
    return float(m.group(1))


def _run_llmbo_variant(
    *,
    circuit: str,
    task_path: str,
    paragraph_text: str,
    init_data_pkl: str,
    n_init_data: int,
    n_itr: int,
    n_proposal_llm: int,
    n_proposal_bo: int,
    gpt_version: str,
    openai_api_seed: int,
    llm_temperature: float,
) -> float:
    repo = _repo_root()
    runner = os.path.join(repo, "LLMBO", "QA_verification_llmbo.py")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
        tmp.write(paragraph_text)
        tmp_path = tmp.name

    try:
        cmd = [
            _analog_rep_python(),
            runner,
            "--circuit",
            circuit,
            "--task_path",
            task_path,
            "--history",
            "1",
            "--paragraph_file",
            tmp_path,
            "--init_data_pkl",
            init_data_pkl,
            "--n_init_data",
            str(n_init_data),
            "--n_itr",
            str(n_itr),
            "--n_proposal_llm",
            str(n_proposal_llm),
            "--n_proposal_bo",
            str(n_proposal_bo),
            "--gpt_version",
            gpt_version,
            "--openai_api_seed",
            str(openai_api_seed),
            "--llm_temperature",
            str(llm_temperature),
        ]
        proc = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Variant run failed (rc={proc.returncode}). Output:\n{proc.stdout}")
        return _parse_final_best_fom(proc.stdout)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _compile_paragraph(entries: List[Dict[str, Any]]) -> str:
    lines = []
    for e in entries:
        qid = str(e.get("id", ""))
        sent = str(e.get("sentence", "")).strip()
        if not sent:
            continue
        lines.append(f"[{qid}] {sent}" if qid else sent)
    return "\n".join(lines)


def _pick_wrong_option(entry: Dict[str, Any]) -> str:
    golden = str(entry.get("golden_answer"))
    opts = entry.get("options") or {}
    for key in ["A", "B", "C", "D"]:
        v = opts.get(key)
        if v is None:
            continue
        v = str(v)
        if v != golden:
            return v
    # fallback
    return "incorrect"


def _flip_sentence(entry: Dict[str, Any]) -> str:
    sent = str(entry.get("sentence", "")).strip()
    if not sent:
        return sent

    golden = str(entry.get("golden_answer"))
    wrong = _pick_wrong_option(entry)

    # Directional flip
    if "Increasing " in sent and (" increases " in sent or " decreases " in sent):
        if " increases " in sent:
            return sent.replace(" increases ", " decreases ", 1)
        if " decreases " in sent:
            return sent.replace(" decreases ", " increases ", 1)

    # If the golden answer text appears in the sentence, replace it with a wrong option.
    if golden and golden in sent:
        return sent.replace(golden, wrong, 1)

    # Relationship sentence
    if " is ambiguous" in sent:
        return sent.replace("ambiguous", wrong, 1)

    # Fallback: append a contradictory marker (least preferred)
    return f"{sent} (incorrect: {wrong})."


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs))


def _mean(xs: List[float]) -> float:
    # statistics.fmean is stable and avoids float summation pitfalls.
    return float(statistics.fmean(xs))


def _ensure_init_data(
    *,
    circuit: str,
    task_path: str,
    init_data_pkl: str,
    n_init_data: int,
    gpt_version: str,
    openai_api_seed: int,
) -> None:
    if os.path.exists(init_data_pkl):
        return

    repo = _repo_root()
    runner = os.path.join(repo, "LLMBO", "QA_verification_llmbo.py")
    os.makedirs(os.path.dirname(os.path.abspath(init_data_pkl)), exist_ok=True)

    # Empty paragraph is fine; we only want initialization data.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
        tmp.write("")
        tmp_path = tmp.name

    try:
        cmd = [
            _analog_rep_python(),
            runner,
            "--circuit",
            circuit,
            "--task_path",
            task_path,
            "--history",
            "0",
            "--paragraph_file",
            tmp_path,
            "--n_init_data",
            str(n_init_data),
            "--n_itr",
            "0",
            "--n_proposal_llm",
            "0",
            "--n_proposal_bo",
            "0",
            "--gpt_version",
            gpt_version,
            "--openai_api_seed",
            str(openai_api_seed),
            "--save_init_data_pkl",
            init_data_pkl,
        ]
        proc = subprocess.run(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Init-data generation failed (rc={proc.returncode}). Output:\n{proc.stdout}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Budgeted QA filtering using LLMBO paragraph verification.")
    ap.add_argument("--circuit", required=True, choices=["amp2", "FC", "comp", "ldo"])
    ap.add_argument("--task_path", default=None)
    ap.add_argument("--raw_qa", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scores_out", default=None)

    ap.add_argument(
        "--paragraph_k",
        type=int,
        default=20,
        help="How many QA sentences to include in the base paragraph. Use 0 or -1 for all.",
    )
    ap.add_argument("--paragraph_ids", nargs="+", default=None, help="Explicit QA ids for the base paragraph.")

    ap.add_argument("--repeats", type=int, default=3, help="How many repeats (median) per variant.")
    ap.add_argument("--min_delta", type=float, default=0.0, help="Minimum delta required for keep decisions.")

    ap.add_argument("--n_init_data", type=int, default=1)
    ap.add_argument("--n_itr", type=int, default=1)
    ap.add_argument("--n_proposal_llm", type=int, default=1)
    ap.add_argument("--n_proposal_bo", type=int, default=0)

    ap.add_argument("--gpt_version", default="3.5", choices=["3.5", "4"])
    ap.add_argument("--openai_api_seed", type=int, default=514, help="Passed through; does not fully control LLM randomness.")
    ap.add_argument("--llm_temperature", type=float, default=0.0, help="LLM temperature for proposer suggestions (lower is more deterministic).")

    ap.add_argument("--init_data_pkl", default=None, help="Path to reuse initialization data pickle.")

    args = ap.parse_args()

    repo = _repo_root()
    task_path = args.task_path
    if not task_path:
        # Mirror defaults in QA_verification_llmbo.py
        task_path = os.path.join(repo, "LLMBO", "tasks", args.circuit, f"{args.circuit}.json")

    raw_qa = args.raw_qa or _default_raw_qa_path(args.circuit)
    out_path = args.out or _default_filtered_out_path(args.circuit)
    scores_out = args.scores_out or _default_scores_out_path(args.circuit)

    init_data_pkl = args.init_data_pkl or _default_init_pkl_path(args.circuit)
    print(f"[filter] circuit={args.circuit} task_path={task_path}", flush=True)
    print(f"[filter] raw_qa={raw_qa}", flush=True)
    print(f"[filter] init_data_pkl={init_data_pkl} (exists={os.path.exists(init_data_pkl)})", flush=True)
    _ensure_init_data(
        circuit=args.circuit,
        task_path=task_path,
        init_data_pkl=init_data_pkl,
        n_init_data=args.n_init_data,
        gpt_version=args.gpt_version,
        openai_api_seed=args.openai_api_seed,
    )
    print(f"[filter] init_data_pkl ready (exists={os.path.exists(init_data_pkl)})", flush=True)

    with open(raw_qa, "r") as f:
        qa_all: List[Dict[str, Any]] = json.load(f)

    id_to_entry = {str(x.get("id")): x for x in qa_all if x.get("id") is not None}

    if args.paragraph_ids:
        base_ids = [str(x) for x in args.paragraph_ids]
    else:
        k = int(args.paragraph_k)
        all_ids = sorted(id_to_entry.keys())
        if k <= 0:
            base_ids = all_ids
        else:
            base_ids = all_ids[:k]

    base_entries = [id_to_entry[i] for i in base_ids if i in id_to_entry]
    print(
        f"[filter] loaded {len(qa_all)} QA items; base paragraph has {len(base_entries)} sentences",
        flush=True,
    )
    print(
        f"[filter] repeats={args.repeats} n_itr={args.n_itr} n_proposal_llm={args.n_proposal_llm} "
        f"n_proposal_bo={args.n_proposal_bo} llm_temperature={args.llm_temperature}",
        flush=True,
    )

    def eval_paragraph(paragraph_entries: List[Dict[str, Any]], *, label: str) -> Tuple[float, List[float]]:
        paragraph = _compile_paragraph(paragraph_entries)
        foms = []
        for r in range(int(args.repeats)):
            print(f"[filter] {label}: repeat {r + 1}/{int(args.repeats)}", flush=True)
            fom = _run_llmbo_variant(
                circuit=args.circuit,
                task_path=task_path,
                paragraph_text=paragraph,
                init_data_pkl=init_data_pkl,
                n_init_data=args.n_init_data,
                n_itr=args.n_itr,
                n_proposal_llm=args.n_proposal_llm,
                n_proposal_bo=args.n_proposal_bo,
                gpt_version=args.gpt_version,
                openai_api_seed=args.openai_api_seed + r,
                llm_temperature=float(args.llm_temperature),
            )
            foms.append(float(fom))
            print(f"[filter] {label}: FOM={float(fom):.6f}", flush=True)
        return _mean(foms), foms

    baseline_mean, baseline_foms = eval_paragraph(base_entries, label="baseline")
    print(f"[filter] baseline_mean={baseline_mean:.6f}", flush=True)

    results = []
    kept: List[Dict[str, Any]] = []

    for idx, entry in enumerate(base_entries, start=1):
        qid = str(entry.get("id"))
        print(f"[filter] {idx}/{len(base_entries)} id={qid}: evaluating ablate/flip", flush=True)

        ablated_entries = [e for e in base_entries if str(e.get("id")) != qid]
        ablated_mean, ablated_foms = eval_paragraph(ablated_entries, label=f"ablate:{qid}")

        flipped_entry = dict(entry)
        flipped_entry["sentence"] = _flip_sentence(entry)
        flipped_entries = [flipped_entry if str(e.get("id")) == qid else e for e in base_entries]
        flipped_mean, flipped_foms = eval_paragraph(flipped_entries, label=f"flip:{qid}")

        delta_remove = baseline_mean - ablated_mean
        delta_flip = baseline_mean - flipped_mean

        keep = (delta_remove >= float(args.min_delta)) and (delta_flip >= float(args.min_delta))
        print(
            f"[filter] id={qid}: delta_remove={delta_remove:.6f} delta_flip={delta_flip:.6f} keep={keep}",
            flush=True,
        )

        results.append(
            {
                "id": qid,
                "aggregate": "mean",
                "baseline_mean": baseline_mean,
                "ablated_mean": ablated_mean,
                "flipped_mean": flipped_mean,
                # Backward-compatible keys (older files used *median but now store mean).
                "baseline_median": baseline_mean,
                "ablated_median": ablated_mean,
                "flipped_median": flipped_mean,
                "delta_remove": delta_remove,
                "delta_flip": delta_flip,
                "baseline_foms": baseline_foms,
                "ablated_foms": ablated_foms,
                "flipped_foms": flipped_foms,
                "keep": keep,
            }
        )

        if keep:
            out_entry = dict(entry)
            out_entry["verification"] = {
                "aggregate": "mean",
                "baseline_mean_fom": baseline_mean,
                # Backward-compatible key name.
                "baseline_median_fom": baseline_mean,
                "delta_remove": delta_remove,
                "delta_flip": delta_flip,
                "n_itr": int(args.n_itr),
                "repeats": int(args.repeats),
            }
            kept.append(out_entry)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(kept, f, indent=2)

    with open(scores_out, "w") as f:
        json.dump(
            {
                "circuit": args.circuit,
                "task_path": task_path,
                "raw_qa": raw_qa,
                "init_data_pkl": init_data_pkl,
                "paragraph_ids": base_ids,
                "aggregate": "mean",
                "baseline_mean": baseline_mean,
                # Backward-compatible key name.
                "baseline_median": baseline_mean,
                "baseline_foms": baseline_foms,
                "config": {
                    "repeats": int(args.repeats),
                    "n_init_data": int(args.n_init_data),
                    "n_itr": int(args.n_itr),
                    "n_proposal_llm": int(args.n_proposal_llm),
                    "n_proposal_bo": int(args.n_proposal_bo),
                    "min_delta": float(args.min_delta),
                    "gpt_version": args.gpt_version,
                    "llm_temperature": float(args.llm_temperature),
                },
                "per_id": results,
            },
            f,
            indent=2,
        )

    print(
        f"Wrote {len(kept)}/{len(base_entries)} kept QA entries to {os.path.relpath(out_path, repo)}; "
        f"scores in {os.path.relpath(scores_out, repo)} (baseline_mean={baseline_mean:.6f})."
    )


if __name__ == "__main__":
    main()
