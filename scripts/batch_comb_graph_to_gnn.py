#!/usr/bin/env python3
"""Batch-run comb_graph_to_gnn over netlists families.

By default this writes outputs *in-place* (next to each circuit's comb_graph.json)
which matches running:
  python3 scripts/comb_graph_to_gnn.py --in <circuit_dir>/comb_graph.json

Examples:
  python3 scripts/batch_comb_graph_to_gnn.py --families diff_amps comparators

  # Dry-run (print what would run)
  python3 scripts/batch_comb_graph_to_gnn.py --families diff_amps comparators --dry-run

Notes:
- Supports either `comb_graph.json` or `<id>_comb_graph.json` per circuit dir.
- Skips circuit dirs that don't contain either comb graph file.
- Optionally checks that global meaning files exist before converting.
- Continues on errors but reports failures at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_comb_graph_file(circuit_dir: Path) -> Path | None:
    """Return the comb graph JSON path in `circuit_dir`, or None if missing."""
    primary = circuit_dir / "comb_graph.json"
    if primary.exists():
        return primary
    alt = circuit_dir / f"{circuit_dir.name}_comb_graph.json"
    if alt.exists():
        return alt
    return None


def iter_comb_graphs(netlists_root: Path, family: str):
    fam_dir = netlists_root / family
    if not fam_dir.exists() or not fam_dir.is_dir():
        return
    for circuit_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
        cg = find_comb_graph_file(circuit_dir)
        if cg is not None:
            yield cg


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch run comb_graph_to_gnn.py over families")
    ap.add_argument("--netlists-root", default="netlists", help="Path to netlists directory")
    ap.add_argument("--families", nargs="+", required=True, help="Family dirs under netlists/, e.g. diff_amps comparators")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--dry-run", action="store_true", help="Print commands instead of running")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop immediately on first failure")
    ap.add_argument(
        "--require-global-meanings",
        action="store_true",
        help="Fail fast unless netlists/_performance_meanings.json and netlists/_substructures_ordered.json exist",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    netlists_root = (repo_root / args.netlists_root).resolve()

    # Optional preflight: ensure global vocab files exist so conversion uses a stable mapping.
    if args.require_global_meanings:
        perf_path = netlists_root / "_performance_meanings.json"
        subs_path = netlists_root / "_substructures_ordered.json"
        missing = [p for p in (perf_path, subs_path) if not p.exists()]
        if missing:
            print("Missing required global meaning files:")
            for p in missing:
                print(f"- {p}")
            print("Generate them first, e.g.:")
            print(f"  {args.python} scripts/generate_global_meanings.py --families {' '.join(args.families)}")
            return 2

    processed = 0
    skipped_families = []
    failures = []

    for fam in args.families:
        fam_dir = netlists_root / fam
        if not fam_dir.exists():
            skipped_families.append(fam)
            continue

        for cg in iter_comb_graphs(netlists_root, fam):
            cmd = [args.python, str(repo_root / "scripts" / "comb_graph_to_gnn.py"), "--in", str(cg)]
            if args.dry_run:
                print("DRY:", " ".join(cmd))
                processed += 1
                continue

            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if res.returncode != 0:
                failures.append((str(cg), res.returncode, res.stdout[-4000:]))
                if args.stop_on_error:
                    print(f"FAILED: {cg} (exit={res.returncode})")
                    print(res.stdout)
                    return res.returncode
            processed += 1

    print(f"Processed comb_graph.json files: {processed}")
    if skipped_families:
        print("Skipped missing families:", ", ".join(skipped_families))

    if failures:
        print(f"Failures: {len(failures)}")
        for path, code, tail in failures[:50]:
            print("---")
            print(f"{path} (exit={code})")
            print(tail)
        return 1

    print("All conversions completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
