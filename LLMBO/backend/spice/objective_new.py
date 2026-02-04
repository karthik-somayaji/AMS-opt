"""Objective / metric normalization (new spec-based scheme).

This module introduces a spec-aware normalization:
- Maximizable metrics (gain/cmrr/gbw):
  - 0 if <= norm_min
  - 1 if >= spec
  - linear between norm_min and spec otherwise
- Minimizable metrics (pow):
  - 1 if <= spec
  - 0 if >= norm_max
  - linear (inverted) between spec and norm_max otherwise

The resulting normalized values are always in [0, 1].

FOM uses the same weighted-sum structure as the original implementation, but for
Option 1 we treat `pow_normed` as "higher is better" so the expected weight for
`pow_weight` should be positive.
"""

from typing import Dict


def read_results(f_val, val_name, args, check_specs: bool = False):
    """Parse a float from raw simulator output, optionally failing on spec violations.

    NOTE: This function mirrors the original API in `objective.py` but fixes a bug
    in the original implementation.

    Args:
        f_val: value as string
        val_name: name of metric
        args: task/circuit dict
        check_specs: if True, return `<metric>_failed` when spec violated
    """

    try:
        f_val_f = float(f_val)
        if check_specs:
            if val_name == "pow":
                # Minimizable: violation if above spec
                if f_val_f > args[f"{val_name}_spec"]:
                    return args[f"{val_name}_failed"]
            else:
                # Maximizable: violation if below spec
                if f_val_f < args[f"{val_name}_spec"]:
                    return args[f"{val_name}_failed"]
        return f_val_f
    except Exception:
        print(f"Failed to read {val_name} from the file")
        return args[f"{val_name}_failed"]


def _clip01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def _norm_maximize(value: float, norm_min: float, norm_max: float, spec: float) -> float:
    """Spec-aware normalization for metrics we want to maximize."""
    # Handle degenerate configuration
    if spec <= norm_min:
        # If spec is at/below min, then anything >= spec returns 1 by definition.
        return 1.0 if value >= spec else 0.0

    if value <= norm_min:
        return 0.0
    if value >= spec:
        return 1.0

    # If value is between norm_min and spec, map linearly to [0,1]
    return _clip01((value - norm_min) / (spec - norm_min))


def _norm_minimize(value: float, norm_min: float, norm_max: float, spec: float) -> float:
    """Spec-aware normalization for metrics we want to minimize."""
    # For minimize, we only need spec and norm_max for the shape; norm_min is kept
    # for signature symmetry and potential future use.
    if norm_max <= spec:
        # Degenerate: if norm_max at/below spec, then anything <= spec is 1 else 0.
        return 1.0 if value <= spec else 0.0

    if value >= norm_max:
        return 0.0
    if value <= spec:
        return 1.0

    # Between spec and norm_max, invert linearly
    return _clip01((norm_max - value) / (norm_max - spec))


def normalize_fvals(f_vals: Dict, ckt: Dict) -> Dict:
    """Normalize objective metrics using spec-aware rules."""

    # Raw values
    gbw = float(f_vals.get("gbw", 0.0))
    gain = float(f_vals.get("gain", 0.0))
    cmrr = float(f_vals.get("cmrr", 0.0))
    pm = float(f_vals.get("pm", 0.0))
    pow_v = float(f_vals.get("pow", 0.0))

    # Ranges
    gbw_min, gbw_max = ckt["gbw_norm"]
    gain_min, gain_max = ckt["gain_norm"]
    cmrr_min, cmrr_max = ckt["cmrr_norm"]
    pm_min, pm_max = ckt.get("pm_norm", [0.0, 1.0])
    pow_min, pow_max = ckt["pow_norm"]

    # Specs
    gbw_spec = ckt["gbw_spec"]
    gain_spec = ckt["gain_spec"]
    cmrr_spec = ckt["cmrr_spec"]
    pm_spec = ckt.get("pm_spec", pm_max)
    pow_spec = ckt["pow_spec"]

    # Spec-aware normalization
    f_vals["gbw_normed"] = _norm_maximize(gbw, gbw_min, gbw_max, gbw_spec)
    f_vals["gain_normed"] = _norm_maximize(gain, gain_min, gain_max, gain_spec)
    f_vals["cmrr_normed"] = _norm_maximize(cmrr, cmrr_min, cmrr_max, cmrr_spec)

    # PM not requested in new FOM, but keep a sensible normalized version
    f_vals["pm_normed"] = _norm_maximize(pm, pm_min, pm_max, pm_spec)

    # Power: lower is better -> normalized is higher-is-better
    f_vals["pow_normed"] = _norm_minimize(pow_v, pow_min, pow_max, pow_spec)

    return f_vals


def objective(f_vals: Dict, ckt: Dict) -> Dict:
    """Compute spec-aware weighted FOM.

    Option 1 semantics: `pow_normed` is "higher is better" (1 means meets/exceeds
    power spec), so `pow_weight` should be positive.
    """

    f_vals = normalize_fvals(f_vals, ckt)

    # Use same merit set as original amp2 configuration (pm excluded per request).
    merits = ["gbw", "gain", "cmrr", "pow"]

    fom = 0.0
    for merit in merits:
        weight = float(ckt.get(f"{merit}_weight", 0.0))
        fom += weight * float(f_vals.get(f"{merit}_normed", 0.0))

    f_vals["fom"] = float(fom)
    return f_vals
