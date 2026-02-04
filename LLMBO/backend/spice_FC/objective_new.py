"""FC objective with spec-aware clipped normalization.

This mirrors the amp2 `objective_new.py` behavior:
- For maximize metrics: 0 below norm_min, 1 at/above spec, linear between.
- For minimize metric (power): 1 at/below spec, 0 at/above norm_max, linear between.

With this scheme, all normalized metrics are in [0, 1] and higher is better.
Therefore `pow_weight` should be positive.
"""

from typing import Dict


def read_results(f_val, val_name, args, check_specs: bool = False):
    """Parse a float value from HSPICE output with optional spec checking."""
    try:
        parsed = float(f_val)

        if check_specs:
            if val_name == "pow":
                if parsed > args[f"{val_name}_spec"]:
                    return args[f"{val_name}_failed"]
            else:
                if parsed < args[f"{val_name}_spec"]:
                    return args[f"{val_name}_failed"]

        return parsed
    except (TypeError, ValueError):
        print(f"Failed to read {val_name} from the file")
        return args[f"{val_name}_failed"]


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x


def _norm_maximize(value: float, norm_min: float, spec: float) -> float:
    """Higher is better; normalize to [0,1] with saturation at spec."""
    if spec <= norm_min:
        return 1.0 if value >= spec else 0.0
    if value <= norm_min:
        return 0.0
    if value >= spec:
        return 1.0
    return _clamp01((value - norm_min) / (spec - norm_min))


def _norm_minimize(value: float, spec: float, norm_max: float) -> float:
    """Lower is better; normalize to [0,1] with saturation at spec."""
    if norm_max <= spec:
        return 1.0 if value <= spec else 0.0
    if value <= spec:
        return 1.0
    if value >= norm_max:
        return 0.0
    return _clamp01((norm_max - value) / (norm_max - spec))


def normalize_fvals(f_vals: Dict, ckt: Dict) -> Dict:
    """Attach `*_normed` keys to f_vals for FC metrics."""
    metrics_list = ckt.get("metrics_list", []) or []

    gbw = float(f_vals.get("gbw", 0.0))
    gain = float(f_vals.get("gain", 0.0))
    cmrr = float(f_vals.get("cmrr", 0.0))
    pow_v = float(f_vals.get("pow", 0.0))
    pm = float(f_vals.get("pm", 0.0))

    if "gbw" in metrics_list:
        f_vals["gbw_normed"] = _norm_maximize(gbw, ckt["gbw_norm"][0], ckt["gbw_spec"])
    if "gain" in metrics_list:
        f_vals["gain_normed"] = _norm_maximize(gain, ckt["gain_norm"][0], ckt["gain_spec"])
    if "cmrr" in metrics_list:
        f_vals["cmrr_normed"] = _norm_maximize(cmrr, ckt["cmrr_norm"][0], ckt["cmrr_spec"])
    if "pm" in metrics_list and "pm_spec" in ckt and "pm_norm" in ckt:
        f_vals["pm_normed"] = _norm_maximize(pm, ckt["pm_norm"][0], ckt["pm_spec"])

    # Power is minimize.
    if "pow" in metrics_list:
        f_vals["pow_normed"] = _norm_minimize(pow_v, ckt["pow_spec"], ckt["pow_norm"][1])

    return f_vals


def objective(f_vals: Dict, ckt: Dict) -> Dict:
    """Compute FC FOM using clipped, spec-aware normalized metrics."""
    f_vals = normalize_fvals(f_vals, ckt)

    merits = list(ckt.get("metrics_list", []) or [])

    fom = 0.0
    for merit in merits:
        weight_key = f"{merit}_weight"
        if weight_key not in ckt:
            continue
        fom += float(ckt[weight_key]) * float(f_vals.get(f"{merit}_normed", 0.0))

    f_vals["fom"] = float(fom)
    return f_vals
