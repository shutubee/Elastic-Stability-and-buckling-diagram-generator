from __future__ import annotations

from dataclasses import dataclass, asdict
from math import pi, sqrt, isfinite
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


LENGTH_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
FORCE_TO_N = {"N": 1.0, "kN": 1e3, "MN": 1e6}
STRESS_TO_PA = {"Pa": 1.0, "MPa": 1e6, "GPa": 1e9}

BOUNDARY_K = {
    "Fixed-Fixed": 0.5,
    "Fixed-Pinned": 0.7,
    "Pinned-Pinned": 1.0,
    "Fixed-Free": 2.0,
    "Guided-Guided": 1.0,
    "Custom / Override": 1.0,
}

MATERIAL_PRESETS = {
    "Structural Steel": {"E_GPa": 200.0, "yield_MPa": 250.0, "nu": 0.30, "density": 7850.0},
    "Aluminum Alloy": {"E_GPa": 70.0, "yield_MPa": 240.0, "nu": 0.33, "density": 2700.0},
    "Concrete (Simplified)": {"E_GPa": 30.0, "yield_MPa": 35.0, "nu": 0.20, "density": 2400.0},
    "Composite (Baseline)": {"E_GPa": 120.0, "yield_MPa": 600.0, "nu": 0.25, "density": 1600.0},
    "Custom": {"E_GPa": 200.0, "yield_MPa": 250.0, "nu": 0.30, "density": 7850.0},
}

PRESETS = {
    "P1 - Slender steel Euler benchmark": {
        "length_m": 2.0,
        "shape": "Rectangular",
        "dims_m": {"width": 0.05, "depth": 0.05},
        "axis": "Auto governing/min I",
        "material": "Structural Steel",
        "boundary": "Pinned-Pinned",
        "load_N": 100e3,
        "ecc_m": 0.0,
        "method": "Auto regime",
    },
    "P2 - Fixed-fixed steel column": {
        "length_m": 2.0,
        "shape": "Rectangular",
        "dims_m": {"width": 0.05, "depth": 0.05},
        "axis": "Auto governing/min I",
        "material": "Structural Steel",
        "boundary": "Fixed-Fixed",
        "load_N": 100e3,
        "ecc_m": 0.0,
        "method": "Auto regime",
    },
    "P3 - Intermediate Johnson case": {
        "length_m": 1.25,
        "shape": "Circular Solid",
        "dims_m": {"diameter": 0.06},
        "axis": "Auto governing/min I",
        "material": "Structural Steel",
        "boundary": "Pinned-Pinned",
        "load_N": 200e3,
        "ecc_m": 0.0,
        "method": "Auto regime",
    },
    "P4 - Stocky plastic-dominated column": {
        "length_m": 0.75,
        "shape": "Rectangular",
        "dims_m": {"width": 0.10, "depth": 0.10},
        "axis": "Auto governing/min I",
        "material": "Structural Steel",
        "boundary": "Fixed-Fixed",
        "load_N": 900e3,
        "ecc_m": 0.0,
        "method": "Johnson",
    },
    "P5 - Cantilever eccentric aluminum": {
        "length_m": 1.8,
        "shape": "Hollow Circular",
        "dims_m": {"outer_diameter": 0.08, "inner_diameter": 0.05},
        "axis": "Auto governing/min I",
        "material": "Aluminum Alloy",
        "boundary": "Fixed-Free",
        "load_N": 40e3,
        "ecc_m": 0.010,
        "method": "Auto regime",
    },
    "P8 - Fully custom": {
        "length_m": 2.0,
        "shape": "Rectangular",
        "dims_m": {"width": 0.10, "depth": 0.10},
        "axis": "Auto governing/min I",
        "material": "Custom",
        "boundary": "Pinned-Pinned",
        "load_N": 100e3,
        "ecc_m": 0.0,
        "method": "Auto regime",
    },
}


@dataclass
class AnalysisInput:
    length_m: float
    k_factor: float
    shape: str
    dims_m: Dict[str, float]
    axis: str
    E_Pa: float
    yield_Pa: float
    nu: float
    density_kg_m3: float
    load_N: float
    eccentricity_m: float
    method: str = "Auto regime"
    imperfection_m: float = 0.0
    design_factor: float = 1.0


@dataclass
class SectionProperties:
    area_m2: float
    Ix_m4: float
    Iy_m4: float
    Igov_m4: float
    rgov_m: float
    axis_used: str
    c_m: float


@dataclass
class AnalysisResult:
    inputs: Dict[str, Any]
    section: Dict[str, Any]
    derived: Dict[str, Any]
    results: Dict[str, Any]
    warnings: List[Dict[str, str]]
    errors: List[Dict[str, str]]


def length_to_m(value: float, unit: str) -> float:
    return float(value) * LENGTH_TO_M[unit]


def force_to_n(value: float, unit: str) -> float:
    return float(value) * FORCE_TO_N[unit]


def stress_to_pa(value: float, unit: str) -> float:
    return float(value) * STRESS_TO_PA[unit]


def format_si(value: float, units: str, precision: int = 3) -> str:
    if value is None or not isfinite(float(value)):
        return "â"
    return f"{value:.{precision}g} {units}"


def shear_modulus(E_Pa: float, nu: float) -> float:
    return E_Pa / (2.0 * (1.0 + nu))


def compute_section(shape: str, dims: Dict[str, float], axis: str = "Auto governing/min I") -> Tuple[SectionProperties | None, List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []

    def err(code: str, msg: str, fix: str = "Revise the section dimensions.") -> None:
        errors.append({"code": code, "severity": "error", "message": msg, "suggested_fix": fix})

    shape = str(shape)
    area = Ix = Iy = c = np.nan

    try:
        if shape == "Rectangular":
            b = float(dims.get("width", 0.0))
            h = float(dims.get("depth", 0.0))
            if b <= 0 or h <= 0:
                err("W1", "Rectangular width and depth must be positive.")
                return None, errors
            area = b * h
            Ix = b * h**3 / 12.0
            Iy = h * b**3 / 12.0
            c = max(b, h) / 2.0
        elif shape == "Circular Solid":
            d = float(dims.get("diameter", 0.0))
            if d <= 0:
                err("W1", "Circular diameter must be positive.")
                return None, errors
            area = pi * d**2 / 4.0
            Ix = Iy = pi * d**4 / 64.0
            c = d / 2.0
        elif shape == "Hollow Circular":
            D = float(dims.get("outer_diameter", 0.0))
            d = float(dims.get("inner_diameter", 0.0))
            if D <= 0:
                err("W1", "Outer diameter must be positive.")
                return None, errors
            if d < 0 or d >= D:
                err("W1", "Invalid hollow section geometry: inner diameter must be smaller than outer diameter.", "Ensure inner diameter < outer diameter.")
                return None, errors
            area = pi * (D**2 - d**2) / 4.0
            Ix = Iy = pi * (D**4 - d**4) / 64.0
            c = D / 2.0
        elif shape == "I-Section":
            bf = float(dims.get("flange_width", 0.0))
            tf = float(dims.get("flange_thickness", 0.0))
            tw = float(dims.get("web_thickness", 0.0))
            h = float(dims.get("total_depth", 0.0))
            if min(bf, tf, tw, h) <= 0:
                err("W1", "I-section dimensions must be positive.")
                return None, errors
            if h <= 2 * tf or bf <= tw:
                err("W15", "I-section is geometrically inconsistent: require h > 2tf and bf > tw.")
                return None, errors
            area = 2.0 * bf * tf + tw * (h - 2.0 * tf)
            # Strong x-axis through centroidal mid-depth.
            Ix = 2.0 * ((bf * tf**3 / 12.0) + bf * tf * ((h - tf) / 2.0) ** 2) + tw * (h - 2.0 * tf) ** 3 / 12.0
            # Weak y-axis.
            Iy = 2.0 * (tf * bf**3 / 12.0) + (h - 2.0 * tf) * tw**3 / 12.0
            c = h / 2.0
        else:
            err("W15", f"Unsupported section shape: {shape}")
            return None, errors
    except (TypeError, ValueError, OverflowError) as exc:
        err("W15", f"Could not compute section properties: {exc}")
        return None, errors

    if min(area, Ix, Iy) <= 0 or not all(isfinite(v) for v in [area, Ix, Iy]):
        err("W15", "Computed non-physical section property.")
        return None, errors

    if axis == "Strong x-axis":
        Igov, axis_used = Ix, "x / strong"
    elif axis == "Weak y-axis":
        Igov, axis_used = Iy, "y / weak"
    else:
        if Ix <= Iy:
            Igov, axis_used = Ix, "x / governing minimum"
        else:
            Igov, axis_used = Iy, "y / governing minimum"

    rgov = sqrt(Igov / area)
    return SectionProperties(area, Ix, Iy, Igov, rgov, axis_used, c), errors


def euler_capacity(E_Pa: float, I_m4: float, K: float, L_m: float) -> Tuple[float, float]:
    Pcr = pi**2 * E_Pa * I_m4 / (K * L_m) ** 2
    return Pcr, Pcr


def johnson_stress(E_Pa: float, yield_Pa: float, slenderness: float) -> float:
    sigma = yield_Pa * (1.0 - (yield_Pa / (4.0 * pi**2 * E_Pa)) * slenderness**2)
    return max(0.0, sigma)


def transition_slenderness(E_Pa: float, yield_Pa: float) -> float:
    return sqrt(2.0 * pi**2 * E_Pa / yield_Pa)


def classify_regime(slenderness: float, lambda_c: float, eccentricity_m: float, imperfection_m: float, method: str) -> Tuple[str, str]:
    if method == "Euler":
        return "Euler", "Forced analytical Euler path."
    if method == "Johnson":
        return "Johnson", "Forced Johnson inelastic column path."
    if method == "Conservative min(Euler, Johnson)":
        return "Conservative", "Uses the lower of Euler and Johnson predictions."
    if eccentricity_m > 0:
        return "Interaction / eccentric", "Axial compression includes first-order eccentric bending."
    if imperfection_m > 0:
        return "Imperfection-sensitive", "Initial imperfection reduces the nominal capacity."
    if slenderness > lambda_c:
        return "Euler", "Elastic buckling governs because slenderness is above the Johnson transition threshold."
    if slenderness < 0.35 * lambda_c:
        return "Plastic / crushing check", "Very stocky member: yielding or crushing may govern before elastic buckling."
    return "Johnson", "Intermediate slenderness: Johnson inelastic transition estimate governs."


def analyze(inp: AnalysisInput) -> AnalysisResult:
    warnings: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    def warn(code: str, message: str, fix: str = "Review the input and interpretation.") -> None:
        warnings.append({"code": code, "severity": "warning", "message": message, "suggested_fix": fix})

    def err(code: str, message: str, fix: str = "Correct the input before solving.") -> None:
        errors.append({"code": code, "severity": "error", "message": message, "suggested_fix": fix})

    if inp.length_m <= 0:
        err("W1", "Column length must be positive.")
    if inp.k_factor <= 0:
        err("W5", "Effective length factor K must be positive.")
    if inp.E_Pa <= 0:
        err("W2", "Young's modulus must be positive.", "Provide a valid elastic modulus.")
    if inp.yield_Pa <= 0:
        err("W3", "Yield stress must be positive.", "Provide a valid yield stress.")
    if not (0.0 <= inp.nu <= 0.5):
        warn("W4", "Poisson ratio is outside the usual 0 to 0.5 range.", "Verify the material model.")
    if inp.load_N <= 0:
        err("W6", "Load magnitude must be positive.", "Provide a positive compression load.")
    if inp.eccentricity_m < 0:
        err("W7", "Eccentricity cannot be negative.")
    if inp.design_factor <= 0:
        err("W15", "Design/load factor must be positive.")

    sec, sec_errors = compute_section(inp.shape, inp.dims_m, inp.axis)
    errors.extend(sec_errors)

    if errors or sec is None:
        return AnalysisResult(asdict(inp), {}, {}, {}, warnings, errors)

    slenderness = inp.k_factor * inp.length_m / sec.rgov_m
    G = shear_modulus(inp.E_Pa, inp.nu)
    lambda_c = transition_slenderness(inp.E_Pa, inp.yield_Pa)

    P_euler, _ = euler_capacity(inp.E_Pa, sec.Igov_m4, inp.k_factor, inp.length_m)
    sigma_euler = P_euler / sec.area_m2

    sigma_johnson = johnson_stress(inp.E_Pa, inp.yield_Pa, slenderness)
    P_johnson = sigma_johnson * sec.area_m2

    if inp.method == "Euler":
        P_nom = P_euler
        sigma_nom = sigma_euler
    elif inp.method == "Johnson":
        P_nom = P_johnson
        sigma_nom = sigma_johnson
    elif inp.method == "Conservative min(Euler, Johnson)":
        P_nom = min(P_euler, P_johnson if P_johnson > 0 else P_euler)
        sigma_nom = P_nom / sec.area_m2
    else:
        if slenderness > lambda_c:
            P_nom, sigma_nom = P_euler, sigma_euler
        elif slenderness < 0.35 * lambda_c:
            P_nom, sigma_nom = min(P_johnson, inp.yield_Pa * sec.area_m2), min(sigma_johnson, inp.yield_Pa)
        else:
            P_nom, sigma_nom = P_johnson, sigma_johnson

    # Imperfection/eccentricity knock-down: simple first-order teaching approximation, not a substitute for code/FEM design.
    imperfection_factor = 1.0
    if inp.imperfection_m > 0:
        imperfection_factor += 5.0 * inp.imperfection_m / max(sec.rgov_m, 1e-12)
    if inp.eccentricity_m > 0:
        imperfection_factor += 2.0 * inp.eccentricity_m / max(sec.c_m, 1e-12)

    P_capacity = P_nom / imperfection_factor
    sigma_capacity = P_capacity / sec.area_m2

    design_load = inp.load_N * inp.design_factor
    safety_factor = P_capacity / design_load if design_load > 0 else np.nan
    utilization = design_load / P_capacity if P_capacity > 0 else np.inf

    M_ecc = design_load * inp.eccentricity_m
    bending_stress = M_ecc * sec.c_m / sec.Igov_m4 if sec.Igov_m4 > 0 else 0.0
    axial_stress = design_load / sec.area_m2
    combined_stress = axial_stress + bending_stress
    combined_util_yield = combined_stress / inp.yield_Pa if inp.yield_Pa > 0 else np.nan

    regime, regime_note = classify_regime(slenderness, lambda_c, inp.eccentricity_m, inp.imperfection_m, inp.method)

    if slenderness > 200:
        warn("W8", "Very slender column; response is sensitive to imperfections.", "Consider imperfection-sensitive or nonlinear analysis.")
    if slenderness < 30:
        warn("W9", "Stocky column; buckling may not govern.", "Check crushing/plastic capacity and local buckling.")
    if inp.eccentricity_m > 0.15 * sec.c_m:
        warn("W7", "Large eccentricity relative to section size; linear estimate may be unrealistic.", "Enable nonlinear analysis or reduce eccentricity for comparison.")
    if utilization >= 1.0:
        warn("W13", "Applied design load reaches or exceeds estimated critical capacity.", "Increase member size, reduce effective length, or reduce load.")
    elif utilization >= 0.8:
        warn("W13", "Load is approaching critical capacity.", "Increase safety margin before design use.")
    if combined_util_yield >= 1.0:
        warn("W15", "Combined axial + eccentric bending stress exceeds yield estimate.", "Check P-M interaction or run nonlinear/FEM verification.")

    failure = "elastic buckling" if regime == "Euler" else "inelastic buckling" if "Johnson" in regime else "interaction / plastic check"
    if combined_util_yield >= 1.0:
        failure = "interaction or yielding"

    derived = {
        "G_Pa": G / 1e9,
        "slenderness_lambda": slenderness,
        "lambda_transition": lambda_c,
        "effective_length_m": inp.k_factor * inp.length_m,
        "euler_load_kN": P_euler / 1e3,
        "euler_stress_MPa": sigma_euler / 1e6,
        "johnson_load_kN": P_johnson / 1e3,
        "johnson_stress_MPa": sigma_johnson / 1e6,
    }
    results = {
        "critical_load_kN": P_capacity / 1e3,
        "critical_stress_MPa": sigma_capacity / 1e6,
        "nominal_load_kN_before_knockdown": P_nom / 1e3,
        "safety_factor": safety_factor,
        "utilization": utilization,
        "regime": regime,
        "regime_note": regime_note,
        "failure_classification": failure,
        "design_load_kN": design_load / 1e3,
        "axial_stress_MPa": axial_stress / 1e6,
        "bending_stress_MPa": bending_stress / 1e6,
        "combined_stress_MPa": combined_stress / 1e6,
        "combined_yield_utilization": combined_util_yield,
        "imperfection_eccentricity_knockdown_factor": imperfection_factor,
    }
    section = {
        "area_m2": sec.area_m2,
        "Ix_m4": sec.Ix_m4,
        "Iy_m4": sec.Iy_m4,
        "Igov_m4": sec.Igov_m4,
        "rgov_m": sec.rgov_m,
        "axis_used": sec.axis_used,
        "c_m": sec.c_m,
    }
    return AnalysisResult(asdict(inp), section, derived, results, warnings, errors)


def stress_slenderness_dataframe(E_Pa: float, yield_Pa: float, lam_min: float = 5.0, lam_max: float = 250.0, n: int = 250) -> pd.DataFrame:
    lam = np.linspace(lam_min, lam_max, n)
    euler = (pi**2 * E_Pa / lam**2) / 1e6
    johnson = np.maximum(0.0, yield_Pa * (1.0 - (yield_Pa / (4.0 * pi**2 * E_Pa)) * lam**2)) / 1e6
    return pd.DataFrame({"slenderness": lam, "Euler stress MPa": euler, "Johnson stress MPa": johnson, "Yield stress MPa": np.full_like(lam, yield_Pa / 1e6)})


def capacity_vs_length_dataframe(inp: AnalysisInput, n: int = 200) -> pd.DataFrame:
    sec, errors = compute_section(inp.shape, inp.dims_m, inp.axis)
    if errors or sec is None:
        return pd.DataFrame()
    Ls = np.linspace(max(inp.length_m * 0.25, 0.05), max(inp.length_m * 2.5, inp.length_m + 0.05), n)
    P_e = pi**2 * inp.E_Pa * sec.Igov_m4 / (inp.k_factor * Ls) ** 2
    lam = inp.k_factor * Ls / sec.rgov_m
    P_j = np.maximum(0.0, inp.yield_Pa * (1.0 - (inp.yield_Pa / (4.0 * pi**2 * inp.E_Pa)) * lam**2)) * sec.area_m2
    return pd.DataFrame({"Length m": Ls, "Euler Pcr kN": P_e / 1e3, "Johnson Pcr kN": P_j / 1e3})


def imperfection_sensitivity_dataframe(base_result: AnalysisResult, inp: AnalysisInput, n: int = 60) -> pd.DataFrame:
    sec_data = base_result.section
    if not sec_data:
        return pd.DataFrame()
    r = sec_data["rgov_m"]
    amps = np.linspace(0.0, max(10e-3, 0.04 * r), n)
    rows = []
    for amp in amps:
        tmp = AnalysisInput(**{**asdict(inp), "imperfection_m": float(amp)})
        res = analyze(tmp)
        rows.append({"Imperfection mm": amp * 1e3, "Critical load kN": res.results.get("critical_load_kN", np.nan), "Utilization": res.results.get("utilization", np.nan)})
    return pd.DataFrame(rows)


def result_tables(result: AnalysisResult) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section_df = pd.DataFrame(
        [
            ("Area A", result.section.get("area_m2", np.nan), "mÂ²"),
            ("Ix", result.section.get("Ix_m4", np.nan), "mâ´"),
            ("Iy", result.section.get("Iy_m4", np.nan), "mâ´"),
            ("I governing", result.section.get("Igov_m4", np.nan), "mâ´"),
            ("r governing", result.section.get("rgov_m", np.nan), "m"),
            ("Axis used", result.section.get("axis_used", "â"), "â"),
        ],
        columns=["Quantity", "Value", "Unit"],
    )
    derived_df = pd.DataFrame([(k, v) for k, v in result.derived.items()], columns=["Derived quantity", "Value"])
    results_df = pd.DataFrame([(k, v) for k, v in result.results.items()], columns=["Result", "Value"])
    return section_df, derived_df, results_df


def build_markdown_report(result: AnalysisResult) -> str:
    section_df, derived_df, results_df = result_tables(result)
    lines = [
        "# Elastic Stability and Buckling Report",
        "",
        "## Primary Results",
        results_df.to_markdown(index=False),
        "",
        "## Derived Geometry and Material Quantities",
        section_df.to_markdown(index=False),
        "",
        derived_df.to_markdown(index=False),
        "",
        "## Validation Messages",
    ]
    if result.errors:
        lines.append("### Errors")
        for item in result.errors:
            lines.append(f"- **{item['code']}** {item['message']} Suggested fix: {item['suggested_fix']}")
    if result.warnings:
        lines.append("### Warnings")
        for item in result.warnings:
            lines.append(f"- **{item['code']}** {item['message']} Suggested fix: {item['suggested_fix']}")
    if not result.errors and not result.warnings:
        lines.append("No validation messages.")
    lines.extend([
        "",
        "## Note",
        "This app is an educational/pre-design calculator. Do not use it as the sole basis for code-compliant design.",
    ])
    return "\n".join(lines)