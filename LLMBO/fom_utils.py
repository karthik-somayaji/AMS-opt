import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent


def _load_objective(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load objective module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.objective


amp2_objective = _load_objective("amp2_objective_module", "backend/spice/objective_new.py")
fc_objective = _load_objective("fc_objective_module", "backend/spice_FC/objective_new.py")
comp_objective = _load_objective("comp_objective_module", "backend/spice_comp/objective_new.py")
ldo_objective = _load_objective("ldo_objective_module", "backend/spice_ldo/objective.py")

TASK_CONFIGS = {
    "amp2": REPO_ROOT / "tasks" / "amp2" / "amp2.json",
    "FC": REPO_ROOT / "tasks" / "FC" / "FC.json",
    "comp": REPO_ROOT / "tasks" / "comp" / "comp.json",
    "ldo": REPO_ROOT / "tasks" / "ldo" / "ldo.json",
}

OBJECTIVES = {
    "amp2": amp2_objective,
    "FC": fc_objective,
    "comp": comp_objective,
    "ldo": ldo_objective,
}


def _load_task_config(circuit: str) -> Dict[str, Any]:
    try:
        config_path = TASK_CONFIGS[circuit]
    except KeyError as exc:
        raise ValueError(f"Unsupported circuit '{circuit}'. Expected one of {sorted(TASK_CONFIGS)}") from exc

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _validate_required_metrics(circuit: str, metrics: Dict[str, Any]) -> None:
    required = {
        "amp2": {"gain", "cmrr", "gbw", "pow"},
        "FC": {"gain", "cmrr", "gbw", "pow"},
        "comp": {"gain", "ugf", "offset", "hyst_err"},
        "ldo": {"q_curr", "stability", "output_voltage_difference"},
    }[circuit]
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Missing required metrics for {circuit}: {missing}")


def compute_fom(circuit: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    _validate_required_metrics(circuit, metrics)
    config = _load_task_config(circuit)
    objective = OBJECTIVES[circuit]

    result = objective({key: float(value) for key, value in metrics.items()}, config)
    return {key: float(value) for key, value in result.items() if isinstance(value, (int, float))}


def compute_amp2_fom(*, gain: float, cmrr: float, gbw: float, pow: float) -> Dict[str, float]:
    return compute_fom("amp2", {"gain": gain, "cmrr": cmrr, "gbw": gbw, "pow": pow})


def compute_fc_fom(*, gain: float, cmrr: float, gbw: float, pow: float) -> Dict[str, float]:
    return compute_fom("FC", {"gain": gain, "cmrr": cmrr, "gbw": gbw, "pow": pow})


def compute_comp_fom(*, gain: float, ugf: float, offset: float, hyst_err: float) -> Dict[str, float]:
    return compute_fom(
        "comp",
        {"gain": gain, "ugf": ugf, "offset": offset, "hyst_err": hyst_err},
    )


def compute_ldo_fom(*, q_curr: float, stability: float, output_voltage_difference: float) -> Dict[str, float]:
    return compute_fom(
        "ldo",
        {
            "q_curr": q_curr,
            "stability": stability,
            "output_voltage_difference": output_voltage_difference,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate circuit FOM directly from metrics.")
    parser.add_argument("circuit", choices=sorted(TASK_CONFIGS), help="Circuit name")
    parser.add_argument(
        "--metrics-json",
        required=True,
        help="JSON object with raw metrics, for example '{\"gain\": 55, \"cmrr\": 80, \"gbw\": 7.4e6, \"pow\": 1.9e-5}'",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = json.loads(args.metrics_json)
    result = compute_fom(args.circuit, metrics)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()