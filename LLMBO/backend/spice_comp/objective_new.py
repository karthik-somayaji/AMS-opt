"""Comparator objective with spec-aware clipped normalization.

For `comp.json` metrics:
- Maximize: `gain`, `ugf`
- Minimize: `offset`, `hyst_err` (treated as absolute values)

All normalized metrics are mapped to [0, 1] where higher is better.
"""

from typing import Dict


def read_results(f_val, val_name, args, check_specs: bool = False):
    """Parse a float from HSPICE output with optional spec checking."""
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
        return args.get(f"{val_name}_failed", 0.0)


def _clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x


def _norm_maximize(value: float, norm_min: float, spec: float) -> float:
    """Higher is better; 0 at/below norm_min, 1 at/above spec."""
    if spec <= norm_min:
        return 1.0 if value >= spec else 0.0
    if value <= norm_min:
        return 0.0
    if value >= spec:
        return 1.0
    return _clamp01((value - norm_min) / (spec - norm_min))


def _norm_minimize(value: float, spec: float, norm_max: float) -> float:
    """Lower is better; 1 at/below spec, 0 at/above norm_max."""
    if norm_max <= spec:
        return 1.0 if value <= spec else 0.0
    if value <= spec:
        return 1.0
    if value >= norm_max:
        return 0.0
    return _clamp01((norm_max - value) / (norm_max - spec))


def normalize_fvals(f_vals: Dict, ckt: Dict) -> Dict:
    """Attach `*_normed` keys to f_vals for comparator metrics."""
    metrics_list = ckt.get("metrics_list", []) or []

    gain = float(f_vals.get("gain", 0.0))
    ugf = float(f_vals.get("ugf", 0.0))
    offset = abs(float(f_vals.get("offset", 0.0)))
    hyst_err = abs(float(f_vals.get("hyst_err", 0.0)))

    if "gain" in metrics_list:
        f_vals["gain_normed"] = _norm_maximize(gain, ckt["gain_norm"][0], ckt["gain_spec"])

    if "ugf" in metrics_list:
        # Guard against invalid values (e.g. negative/zero) by flooring to norm_min.
        ugf_floor = float(ckt["ugf_norm"][0])
        ugf_val = ugf if ugf > 0.0 else ugf_floor
        f_vals["ugf_normed"] = _norm_maximize(ugf_val, ugf_floor, ckt["ugf_spec"])

    if "offset" in metrics_list:
        # Interpret spec/norm bounds in absolute terms.
        offset_spec = abs(float(ckt["offset_spec"]))
        offset_max = abs(float(ckt["offset_norm"][1]))
        f_vals["offset_normed"] = _norm_minimize(offset, offset_spec, offset_max)

    if "hyst_err" in metrics_list:
        hyst_spec = abs(float(ckt["hyst_err_spec"]))
        hyst_max = abs(float(ckt["hyst_err_norm"][1]))
        f_vals["hyst_err_normed"] = _norm_minimize(hyst_err, hyst_spec, hyst_max)

    return f_vals


def objective(f_vals: Dict, ckt: Dict) -> Dict:
    """Compute comparator FOM using clipped, spec-aware normalized metrics."""
    f_vals = normalize_fvals(f_vals, ckt)

    merits = list(ckt.get("metrics_list", []) or [])
    fom = 0.0
    weight_total = 0.0
    for merit in merits:
        weight_key = f"{merit}_weight"
        if weight_key not in ckt:
            continue
        weight = float(ckt[weight_key])
        fom += weight * float(f_vals.get(f"{merit}_normed", 0.0))
        weight_total += weight

    if weight_total > 0.0:
        fom /= weight_total

    f_vals["fom"] = float(fom)
    return f_vals
