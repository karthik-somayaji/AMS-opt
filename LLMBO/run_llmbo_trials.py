import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone


CIRCUITS = ["amp2", "FC", "comp", "ldo"]
MODES = ["topk", "bottomk", "random_family", "ado-kt"]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: str, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _repo_relative_path(path: str, repo_root: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        if os.path.commonpath([repo_root, abs_path]) == repo_root:
            return os.path.relpath(abs_path, repo_root)
    except ValueError:
        pass
    return os.path.basename(abs_path)


def _sanitize_command(command: list[str], repo_root: str) -> list[str]:
    sanitized = []
    for index, token in enumerate(command):
        if os.path.isabs(token):
            if index == 0:
                sanitized.append(os.path.basename(token))
            else:
                sanitized.append(_repo_relative_path(token, repo_root))
            continue
        sanitized.append(token)
    return sanitized


def _seed_bundle(base_seed: int, history: int, circuit: str, mode: str, trial_id: int) -> dict:
    key = f"{base_seed}:{history}:{circuit}:{mode}:{trial_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _chunk(offset: int) -> int:
        value = int(digest[offset:offset + 8], 16) % 2_147_483_647
        return value if value != 0 else 1

    return {
        "numpy_seed": _chunk(0),
        "torch_seed": _chunk(8),
        "openai_api_seed": _chunk(16),
    }


def _build_jobs(args) -> list[dict]:
    jobs = []
    if not args.skip_history1:
        for circuit in args.circuits:
            for mode in args.modes:
                for trial_id in range(1, args.trials_per_combo + 1):
                    jobs.append(
                        {
                            "history": 1,
                            "circuit": circuit,
                            "mode": mode,
                            "trial_id": trial_id,
                        }
                    )

    if not args.skip_history0:
        for circuit in args.circuits:
            for trial_id in range(1, args.trials_per_combo + 1):
                jobs.append(
                    {
                        "history": 0,
                        "circuit": circuit,
                        "mode": None,
                        "trial_id": trial_id,
                    }
                )
    return jobs


def _run_dir(output_root: str, job: dict) -> str:
    history_dir = os.path.join(output_root, f"history_{job['history']}", job["circuit"])
    if job["history"] == 1:
        history_dir = os.path.join(history_dir, job["mode"])
    return os.path.join(history_dir, f"trial_{job['trial_id']:02d}")


def _build_command(args, job: dict, run_dir: str, seeds: dict) -> list[str]:
    cmd = [
        args.python_executable,
        args.llmbo_script,
        "--history",
        str(job["history"]),
        "--refined",
        str(args.refined),
        "--related_k",
        str(args.related_k),
        "--gnn_embedding_mode",
        args.gnn_embedding_mode,
        "--gnn_checkpoint",
        args.gnn_checkpoint,
        "--embeddings_json",
        args.embeddings_json,
        "--target_id",
        job["circuit"],
        "--circuit",
        job["circuit"],
        "--n_itr",
        str(args.n_itr),
        "--numpy_seed",
        str(seeds["numpy_seed"]),
        "--torch_seed",
        str(seeds["torch_seed"]),
        "--openai_api_seed",
        str(seeds["openai_api_seed"]),
        "--early_stop_fom",
        str(args.early_stop_fom),
        "--run_output_dir",
        run_dir,
        "--path_checkpoints",
        os.path.join(run_dir, "checkpoints"),
        "--trial_id",
        str(job["trial_id"]),
        "--gnn_device",
        args.gnn_device,
    ]

    if args.model:
        cmd.extend(["--model", args.model])
    if args.reference_metadata_json:
        cmd.extend(["--reference_metadata_json", args.reference_metadata_json])
    if args.task_path:
        cmd.extend(["--task_path", args.task_path])
    if job["history"] == 1 and job["mode"]:
        cmd.extend(["--related_mode", job["mode"]])
    return cmd


def _flatten_result(repo_root: str, job: dict, run_dir: str, log_path: str, returncode: int | None, duration_sec: float | None, summary, status: str) -> dict:
    row = {
        "history": job["history"],
        "circuit": job["circuit"],
        "mode": job["mode"],
        "trial_id": job["trial_id"],
        "status": status,
        "returncode": returncode,
        "duration_sec": None if duration_sec is None else round(duration_sec, 3),
        "run_dir": _repo_relative_path(run_dir, repo_root),
        "log_path": _repo_relative_path(log_path, repo_root),
        "summary_path": _repo_relative_path(os.path.join(run_dir, "summary.json"), repo_root),
    }
    if summary:
        row.update(
            {
                "recorded_iteration": summary.get("recorded_iteration"),
                "executed_iterations": summary.get("executed_iterations"),
                "iteration_budget": summary.get("iteration_budget"),
                "stopped_early": summary.get("stopped_early"),
                "best_fom": summary.get("best_fom"),
                "best_llm_fom": summary.get("best_llm_fom"),
                "best_bo_fom": summary.get("best_bo_fom"),
                "best_metrics_json": json.dumps(summary.get("best_metrics"), sort_keys=True),
                "best_llm_metrics_json": json.dumps(summary.get("best_llm_metrics"), sort_keys=True),
                "best_bo_metrics_json": json.dumps(summary.get("best_bo_metrics"), sort_keys=True),
                "numpy_seed": summary.get("numpy_seed"),
                "torch_seed": summary.get("torch_seed"),
                "openai_api_seed": summary.get("openai_api_seed"),
            }
        )
    return row


def _write_aggregate(output_root: str, manifest: list[dict], results: list[dict]) -> None:
    _write_json(os.path.join(output_root, "manifest.json"), manifest)
    _write_json(os.path.join(output_root, "aggregate_results.json"), results)

    csv_path = os.path.join(output_root, "aggregate_results.csv")
    fieldnames = [
        "history",
        "circuit",
        "mode",
        "trial_id",
        "status",
        "returncode",
        "duration_sec",
        "recorded_iteration",
        "executed_iterations",
        "iteration_budget",
        "stopped_early",
        "best_fom",
        "best_llm_fom",
        "best_bo_fom",
        "numpy_seed",
        "torch_seed",
        "openai_api_seed",
        "best_metrics_json",
        "best_llm_metrics_json",
        "best_bo_metrics_json",
        "run_dir",
        "log_path",
        "summary_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(description="Run LLMBO sweeps serially across circuits, modes, and trials.")
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--llmbo_script", type=str, default=os.path.join(repo_root, "LLMBO", "llmbo.py"))
    parser.add_argument("--output_root", type=str, default=os.path.join(repo_root, "results", "llmbo_sweep_runs"))
    parser.add_argument("--circuits", nargs="+", choices=CIRCUITS, default=CIRCUITS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--trials_per_combo", type=int, default=5)
    parser.add_argument("--related_k", type=int, default=2)
    parser.add_argument("--n_itr", type=int, default=20)
    parser.add_argument("--early_stop_fom", type=float, default=1.0)
    parser.add_argument("--base_seed", type=int, default=20260414)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--refined", type=int, choices=[0, 1], default=0)
    parser.add_argument("--gnn_embedding_mode", type=str, choices=["skg", "sg"], default="sg")
    parser.add_argument("--gnn_checkpoint", type=str, default=os.path.join(repo_root, "checkpoints_sg_vs_skg", "best_model.pt"))
    parser.add_argument("--embeddings_json", type=str, default=os.path.join(repo_root, "umap_results_full", "gnn_embeddings_sg_sg_vs_skg.json"))
    parser.add_argument("--reference_metadata_json", type=str, default=None)
    parser.add_argument("--gnn_device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--task_path", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_history0", action="store_true")
    parser.add_argument("--skip_history1", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    jobs = _build_jobs(args)
    manifest = []
    results = []

    print(f"[LLMBO Sweep] Repo root: {repo_root}")
    print(f"[LLMBO Sweep] Output root: {output_root}")
    print(f"[LLMBO Sweep] Total jobs: {len(jobs)}")

    for index, job in enumerate(jobs, start=1):
        run_dir = _run_dir(output_root, job)
        os.makedirs(run_dir, exist_ok=True)
        seeds = _seed_bundle(args.base_seed, job["history"], job["circuit"], job["mode"] or "history0", job["trial_id"])
        command = _build_command(args, job, run_dir, seeds)
        command_str = shlex.join(command)
        log_path = os.path.join(run_dir, "stdout.log")
        summary_path = os.path.join(run_dir, "summary.json")
        metadata = {
            "job": job,
            "seeds": seeds,
            "run_dir": _repo_relative_path(run_dir, repo_root),
            "log_path": _repo_relative_path(log_path, repo_root),
            "summary_path": _repo_relative_path(summary_path, repo_root),
            "command": _sanitize_command(command, repo_root),
            "scheduled_at_utc": _utc_now(),
        }
        _write_json(os.path.join(run_dir, "run_metadata.json"), metadata)
        manifest.append(metadata)

        if args.skip_existing and os.path.exists(summary_path):
            summary = _read_json(summary_path)
            row = _flatten_result(repo_root, job, run_dir, log_path, 0, 0.0, summary, "skipped_existing")
            results.append(row)
            _write_aggregate(output_root, manifest, results)
            print(f"[{index}/{len(jobs)}] Skipping existing run: history={job['history']} circuit={job['circuit']} mode={job['mode']} trial={job['trial_id']}")
            continue

        if args.dry_run:
            row = _flatten_result(repo_root, job, run_dir, log_path, None, None, None, "dry_run")
            results.append(row)
            _write_aggregate(output_root, manifest, results)
            print(f"[{index}/{len(jobs)}] DRY RUN {command_str}")
            continue

        print(f"[{index}/{len(jobs)}] Running history={job['history']} circuit={job['circuit']} mode={job['mode']} trial={job['trial_id']}")
        started_at = time.time()
        with open(log_path, "w") as log_file:
            log_file.write(f"[LLMBO Sweep] Started at {_utc_now()}\n")
            log_file.write(f"[LLMBO Sweep] Command: {command_str}\n\n")
            log_file.flush()
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        duration_sec = time.time() - started_at

        summary = _read_json(summary_path) if os.path.exists(summary_path) else None
        status = "ok" if completed.returncode == 0 and summary is not None else "failed"
        row = _flatten_result(repo_root, job, run_dir, log_path, completed.returncode, duration_sec, summary, status)
        results.append(row)
        _write_aggregate(output_root, manifest, results)

        print(
            f"[{index}/{len(jobs)}] Finished status={status} returncode={completed.returncode} "
            f"duration={duration_sec:.1f}s best_fom={row.get('best_fom')}"
        )

        if completed.returncode != 0 and args.fail_fast:
            print("[LLMBO Sweep] Stopping early because --fail_fast is set.")
            return completed.returncode

    ok_count = sum(1 for row in results if row["status"] in {"ok", "skipped_existing"})
    failed_count = sum(1 for row in results if row["status"] == "failed")
    print(f"[LLMBO Sweep] Completed. ok={ok_count} failed={failed_count} output_root={output_root}")
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())