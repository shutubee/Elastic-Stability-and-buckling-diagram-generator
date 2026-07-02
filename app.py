from __future__ import annotations

import json
from dataclasses import asdict
from math import pi
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from stability_core import (
    BOUNDARY_K,
    LENGTH_TO_M,
    MATERIAL_PRESETS,
    PRESETS,
    AnalysisInput,
    analyze,
    build_markdown_report,
    capacity_vs_length_dataframe,
    force_to_n,
    imperfection_sensitivity_dataframe,
    length_to_m,
    result_tables,
    stress_slenderness_dataframe,
    stress_to_pa,
)


st.set_page_config(
    page_title="Elastic Stability & Buckling Diagram Generator",
    page_icon="📐",
    layout="wide",
)


APP_NOTE = "Educational/pre-design tool. Verify final engineering design with applicable codes, FEM, and a qualified engineer."


# -----------------------------
# Plot helpers
# -----------------------------

def _clean_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_section(shape: str, dims_m: Dict[str, float], axis_used: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    _clean_axes(ax)

    if shape == "Rectangular":
        b = dims_m["width"]
        h = dims_m["depth"]
        ax.add_patch(plt.Rectangle((-b / 2, -h / 2), b, h, fill=False, linewidth=2))
        ax.annotate("b", (0, -h / 2), xytext=(0, -h / 2 - 0.12 * max(b, h)), ha="center", arrowprops=dict(arrowstyle="<->"))
        ax.annotate("h", (b / 2, 0), xytext=(b / 2 + 0.12 * max(b, h), 0), va="center", arrowprops=dict(arrowstyle="<->"))
        lim = max(b, h) * 0.9
    elif shape == "Circular Solid":
        d = dims_m["diameter"]
        ax.add_patch(plt.Circle((0, 0), d / 2, fill=False, linewidth=2))
        ax.annotate("d", (-d / 2, 0), xytext=(d / 2, 0), ha="center", va="center", arrowprops=dict(arrowstyle="<->"))
        lim = d * 0.7
    elif shape == "Hollow Circular":
        D = dims_m["outer_diameter"]
        d = dims_m["inner_diameter"]
        ax.add_patch(plt.Circle((0, 0), D / 2, fill=False, linewidth=2))
        ax.add_patch(plt.Circle((0, 0), d / 2, fill=False, linewidth=2))
        ax.annotate("D", (-D / 2, 0), xytext=(D / 2, 0), ha="center", va="center", arrowprops=dict(arrowstyle="<->"))
        lim = D * 0.7
    else:
        bf = dims_m["flange_width"]
        tf = dims_m["flange_thickness"]
        tw = dims_m["web_thickness"]
        h = dims_m["total_depth"]
        ax.add_patch(plt.Rectangle((-bf / 2, h / 2 - tf), bf, tf, fill=False, linewidth=2))
        ax.add_patch(plt.Rectangle((-bf / 2, -h / 2), bf, tf, fill=False, linewidth=2))
        ax.add_patch(plt.Rectangle((-tw / 2, -h / 2 + tf), tw, h - 2 * tf, fill=False, linewidth=2))
        lim = max(bf, h) * 0.65

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"Section sketch\nAxis used: {axis_used}")
    fig.tight_layout()
    return fig


def plot_boundary_load(boundary: str, K: float, load_kN: float, ecc_mm: float):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    _clean_axes(ax)
    x0 = 0.0
    y0, y1 = 0.0, 1.0
    ax.plot([x0, x0], [y0, y1], linewidth=4)

    # Load arrow. Eccentricity shifts arrow line.
    ex = min(max(ecc_mm / 50.0, -0.35), 0.35) if ecc_mm else 0.0
    ax.annotate("", xy=(ex, y1), xytext=(ex, y1 + 0.28), arrowprops=dict(arrowstyle="-|>", lw=2))
    ax.text(ex + 0.04, y1 + 0.18, f"P = {load_kN:.2g} kN", va="center")
    if abs(ex) > 1e-9:
        ax.plot([0, ex], [y1 + 0.05, y1 + 0.05], linestyle="--")
        ax.text(ex / 2, y1 + 0.08, f"e = {ecc_mm:.1f} mm", ha="center")

    # Support icons.
    lower = boundary.split("-")[-1] if "-" in boundary else boundary
    upper = boundary.split("-")[0] if "-" in boundary else boundary

    def draw_support(y: float, kind: str, top: bool = False):
        name = kind.lower()
        if "fixed" in name:
            ax.plot([-0.25, 0.25], [y, y], linewidth=3)
            for i in np.linspace(-0.22, 0.22, 6):
                dy = -0.05 if not top else 0.05
                ax.plot([i, i + 0.05], [y, y + dy], linewidth=1)
        elif "pinned" in name:
            tri_y = y - 0.08 if not top else y + 0.08
            ax.plot([-0.12, 0.0, 0.12, -0.12], [tri_y, y, tri_y, tri_y], linewidth=2)
        elif "free" in name:
            ax.text(0.12, y, "free", va="center")
        elif "guided" in name:
            ax.plot([-0.18, 0.18], [y, y], linewidth=2)
            ax.plot([-0.18, 0.18], [y + (0.05 if top else -0.05), y + (0.05 if top else -0.05)], linewidth=1)
        else:
            ax.text(0.12, y, "custom", va="center")

    draw_support(y0, lower, top=False)
    draw_support(y1, upper, top=True)
    ax.text(-0.48, 0.50, f"{boundary}\nK = {K:.3g}", va="center")
    ax.set_xlim(-0.6, 0.8)
    ax.set_ylim(-0.25, 1.40)
    ax.set_title("Boundary and load schematic")
    fig.tight_layout()
    return fig


def plot_mode_shape(boundary: str, mode_no: int = 1, scale: float = 1.0):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    y = np.linspace(0, 1, 300)
    n = max(int(mode_no), 1)
    if boundary == "Fixed-Free":
        x = (1 - np.cos((2 * n - 1) * pi * y / 2.0))
    elif boundary == "Fixed-Fixed":
        x = 1 - np.cos(2 * n * pi * y)
        x -= x.mean()
    else:
        x = np.sin(n * pi * y)
    x = 0.25 * scale * x / max(np.max(np.abs(x)), 1e-12)
    ax.plot(np.zeros_like(y), y, linestyle="--", label="undeformed")
    ax.plot(x, y, linewidth=2, label=f"mode {n}")
    ax.set_xlabel("relative lateral displacement")
    ax.set_ylabel("normalized length")
    ax.set_title("Schematic buckling mode shape")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_stress_slenderness(df: pd.DataFrame, lam_op: float, sigma_op: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(df["slenderness"], df["Euler stress MPa"], label="Euler")
    ax.plot(df["slenderness"], df["Johnson stress MPa"], label="Johnson")
    ax.plot(df["slenderness"], df["Yield stress MPa"], linestyle="--", label="yield")
    ax.scatter([lam_op], [sigma_op], s=60, label="current point")
    ax.set_xlabel("Slenderness λ = KL/r")
    ax.set_ylabel("Critical stress (MPa)")
    ax.set_title("Stress-slenderness comparison")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_capacity_length(df: pd.DataFrame, L_op: float, P_op: float):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(df["Length m"], df["Euler Pcr kN"], label="Euler")
    ax.plot(df["Length m"], df["Johnson Pcr kN"], label="Johnson")
    ax.scatter([L_op], [P_op], s=60, label="current point")
    ax.set_xlabel("Column length (m)")
    ax.set_ylabel("Critical load (kN)")
    ax.set_title("Capacity sensitivity to length")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_interaction(axial_util: float, bend_util: float):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    x = np.linspace(0, 1, 100)
    ax.plot(x, 1 - x, label="linear P-M envelope")
    ax.scatter([axial_util], [bend_util], s=70, label="current point")
    ax.set_xlabel("P / Pcr")
    ax.set_ylabel("bending stress / yield stress")
    ax.set_title("Approximate axial-bending interaction")
    ax.set_xlim(0, max(1.2, axial_util * 1.2))
    ax.set_ylim(0, max(1.2, bend_util * 1.2))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


# -----------------------------
# Sidebar and input handling
# -----------------------------

st.title("📐 Elastic Stability and Buckling Diagram Generator")
st.caption(APP_NOTE)

with st.sidebar:
    st.header("Preset")
    preset_name = st.selectbox("Configuration preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]
    st.write("Use presets for quick initialization, then override any field.")

    st.header("Visualization toggles")
    show_section = st.checkbox("Section sketch", value=True)
    show_boundary = st.checkbox("Boundary/load sketch", value=True)
    show_mode = st.checkbox("Mode shape", value=True)
    show_curves = st.checkbox("Euler/Johnson curves", value=True)
    show_capacity = st.checkbox("Capacity vs length", value=True)
    show_imperf = st.checkbox("Imperfection sensitivity", value=False)
    show_interaction = st.checkbox("P-M interaction", value=True)


def value_in_unit(value_m: float, unit: str) -> float:
    return float(value_m) / LENGTH_TO_M[unit]


def default_dim(name: str, fallback_m: float, unit: str) -> float:
    return value_in_unit(preset.get("dims_m", {}).get(name, fallback_m), unit)


tab_inputs, tab_results, tab_visuals, tab_export = st.tabs(["Inputs", "Results", "Visuals", "Export"])

with tab_inputs:
    left, mid, right = st.columns(3)

    with left:
        st.subheader("1) Geometry")
        length_unit = st.selectbox("Column length unit", ["m", "cm", "mm"], index=0)
        length_value = st.number_input(
            "Column length L",
            min_value=0.000001,
            value=value_in_unit(preset["length_m"], length_unit),
            step=0.1 if length_unit == "m" else 10.0,
            format="%.6g",
            key=f"length_{preset_name}_{length_unit}",
        )
        length_m = length_to_m(length_value, length_unit)

        shape_options = ["Rectangular", "Circular Solid", "Hollow Circular", "I-Section"]
        shape_index = shape_options.index(preset["shape"]) if preset["shape"] in shape_options else 0
        shape = st.selectbox("Section shape", shape_options, index=shape_index)
        dim_unit = st.selectbox("Section dimension unit", ["mm", "cm", "m"], index=0)
        axis = st.selectbox("Buckling axis", ["Auto governing/min I", "Strong x-axis", "Weak y-axis"], index=0)

        dims_m: Dict[str, float] = {}
        if shape == "Rectangular":
            width = st.number_input("Width b", min_value=0.000001, value=default_dim("width", 0.10, dim_unit), step=1.0, format="%.6g", key=f"b_{preset_name}_{dim_unit}_{shape}")
            depth = st.number_input("Depth h", min_value=0.000001, value=default_dim("depth", 0.10, dim_unit), step=1.0, format="%.6g", key=f"h_{preset_name}_{dim_unit}_{shape}")
            dims_m = {"width": length_to_m(width, dim_unit), "depth": length_to_m(depth, dim_unit)}
        elif shape == "Circular Solid":
            diameter = st.number_input("Diameter d", min_value=0.000001, value=default_dim("diameter", 0.06, dim_unit), step=1.0, format="%.6g", key=f"d_{preset_name}_{dim_unit}_{shape}")
            dims_m = {"diameter": length_to_m(diameter, dim_unit)}
        elif shape == "Hollow Circular":
            outer_d = st.number_input("Outer diameter D", min_value=0.000001, value=default_dim("outer_diameter", 0.08, dim_unit), step=1.0, format="%.6g", key=f"D_{preset_name}_{dim_unit}_{shape}")
            inner_d = st.number_input("Inner diameter d", min_value=0.0, value=default_dim("inner_diameter", 0.04, dim_unit), step=1.0, format="%.6g", key=f"di_{preset_name}_{dim_unit}_{shape}")
            dims_m = {"outer_diameter": length_to_m(outer_d, dim_unit), "inner_diameter": length_to_m(inner_d, dim_unit)}
        else:
            bf = st.number_input("Flange width bf", min_value=0.000001, value=default_dim("flange_width", 0.10, dim_unit), step=1.0, format="%.6g", key=f"bf_{preset_name}_{dim_unit}_{shape}")
            tf = st.number_input("Flange thickness tf", min_value=0.000001, value=default_dim("flange_thickness", 0.01, dim_unit), step=1.0, format="%.6g", key=f"tf_{preset_name}_{dim_unit}_{shape}")
            tw = st.number_input("Web thickness tw", min_value=0.000001, value=default_dim("web_thickness", 0.008, dim_unit), step=1.0, format="%.6g", key=f"tw_{preset_name}_{dim_unit}_{shape}")
            total_depth = st.number_input("Total depth h", min_value=0.000001, value=default_dim("total_depth", 0.16, dim_unit), step=1.0, format="%.6g", key=f"itotal_{preset_name}_{dim_unit}_{shape}")
            dims_m = {
                "flange_width": length_to_m(bf, dim_unit),
                "flange_thickness": length_to_m(tf, dim_unit),
                "web_thickness": length_to_m(tw, dim_unit),
                "total_depth": length_to_m(total_depth, dim_unit),
            }

    with mid:
        st.subheader("2) Material")
        material_name = st.selectbox("Material preset", list(MATERIAL_PRESETS.keys()), index=list(MATERIAL_PRESETS.keys()).index(preset.get("material", "Structural Steel")))
        mat = MATERIAL_PRESETS[material_name]
        E_value = st.number_input("Young's modulus E", min_value=0.000001, value=float(mat["E_GPa"]), step=1.0, format="%.6g")
        E_unit = st.selectbox("E unit", ["GPa", "MPa", "Pa"], index=0)
        yield_value = st.number_input("Yield stress σy", min_value=0.000001, value=float(mat["yield_MPa"]), step=1.0, format="%.6g")
        yield_unit = st.selectbox("Yield stress unit", ["MPa", "GPa", "Pa"], index=0)
        nu = st.number_input("Poisson ratio ν", min_value=0.0, max_value=0.5, value=float(mat["nu"]), step=0.01, format="%.4f")
        density = st.number_input("Density kg/m³", min_value=1.0, value=float(mat["density"]), step=10.0, format="%.6g")
        mat_model = st.selectbox("Material model label", ["linear_elastic", "elastic_perfectly_plastic", "bilinear_hardening", "custom_curve"], index=0)
        E_Pa = stress_to_pa(E_value, E_unit)
        yield_Pa = stress_to_pa(yield_value, yield_unit)

    with right:
        st.subheader("3) Boundary, load, solver")
        boundary_options = list(BOUNDARY_K.keys())
        boundary = st.selectbox("End condition preset", boundary_options, index=boundary_options.index(preset.get("boundary", "Pinned-Pinned")))
        auto_K = BOUNDARY_K[boundary]
        override_K = st.checkbox("Override K", value=(boundary == "Custom / Override"))
        if override_K:
            K = st.number_input("Effective length factor K", min_value=0.000001, value=float(auto_K), step=0.05, format="%.6g")
        else:
            K = auto_K
            st.info(f"Preset K = {K:.3g}")

        load_type = st.selectbox("Load type", ["Axial compression", "Eccentric compression", "Combined P + M label"], index=1 if preset.get("ecc_m", 0) > 0 else 0)
        load_unit = st.selectbox("Load unit", ["kN", "N", "MN"], index=0)
        load_value = st.number_input("Compression load P", min_value=0.000001, value=preset.get("load_N", 100e3) / 1e3, step=10.0, format="%.6g")
        load_N = force_to_n(load_value, load_unit)

        ecc_unit = st.selectbox("Eccentricity unit", ["mm", "cm", "m"], index=0)
        ecc_default = value_in_unit(float(preset.get("ecc_m", 0.0)), ecc_unit)
        ecc_value = st.number_input("Eccentricity e", min_value=0.0, value=ecc_default if load_type != "Axial compression" else 0.0, step=1.0, format="%.6g")
        eccentricity_m = length_to_m(ecc_value, ecc_unit)

        imperf_value = st.number_input("Initial imperfection amplitude", min_value=0.0, value=0.0, step=0.5, format="%.6g")
        imperf_unit = st.selectbox("Imperfection unit", ["mm", "cm", "m"], index=0)
        imperfection_m = length_to_m(imperf_value, imperf_unit)
        design_factor = st.number_input("Load/design factor γ", min_value=0.000001, value=1.0, step=0.05, format="%.6g")
        method = st.selectbox("Solver method", ["Auto regime", "Euler", "Johnson", "Conservative min(Euler, Johnson)"], index=0)

inp = AnalysisInput(
    length_m=length_m,
    k_factor=K,
    shape=shape,
    dims_m=dims_m,
    axis=axis,
    E_Pa=E_Pa,
    yield_Pa=yield_Pa,
    nu=nu,
    density_kg_m3=density,
    load_N=load_N,
    eccentricity_m=eccentricity_m,
    method=method,
    imperfection_m=imperfection_m,
    design_factor=design_factor,
)
result = analyze(inp)

# -----------------------------
# Results tab
# -----------------------------
with tab_results:
    if result.errors:
        st.error("Inputs contain errors. Fix them before using the results.")
        st.dataframe(pd.DataFrame(result.errors), use_container_width=True)
    else:
        r = result.results
        d = result.derived
        s = result.section

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Critical load", f"{r['critical_load_kN']:.3g} kN")
        c2.metric("Critical stress", f"{r['critical_stress_MPa']:.3g} MPa")
        c3.metric("Safety factor", f"{r['safety_factor']:.3g}")
        c4.metric("Utilization", f"{r['utilization']:.3g}")

        st.subheader("Regime and interpretation")
        st.write(f"**Governing regime:** {r['regime']}")
        st.write(f"**Why:** {r['regime_note']}")
        st.write(f"**Failure classification:** {r['failure_classification']}")

        st.subheader("Derived properties")
        section_df, derived_df, results_df = result_tables(result)
        st.dataframe(section_df, use_container_width=True)
        st.dataframe(derived_df, use_container_width=True)

        st.subheader("Primary result table")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Validation messages")
        if result.warnings:
            st.warning("Review warnings before using the output.")
            st.dataframe(pd.DataFrame(result.warnings), use_container_width=True)
        else:
            st.success("No warnings from the current validation checks.")

# -----------------------------
# Visuals tab
# -----------------------------
with tab_visuals:
    if result.errors:
        st.error("Visuals are hidden because the current input set has errors.")
    else:
        s = result.section
        r = result.results
        d = result.derived

        col_a, col_b = st.columns(2)
        if show_section:
            with col_a:
                st.pyplot(plot_section(shape, dims_m, s["axis_used"]), use_container_width=True)
        if show_boundary:
            with col_b:
                st.pyplot(plot_boundary_load(boundary, K, r["design_load_kN"], eccentricity_m * 1000), use_container_width=True)

        col_c, col_d = st.columns(2)
        if show_mode:
            with col_c:
                mode_no = st.slider("Mode number for schematic", min_value=1, max_value=5, value=1)
                mode_scale = st.slider("Mode shape scale", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
                st.pyplot(plot_mode_shape(boundary, mode_no, mode_scale), use_container_width=True)
        if show_interaction:
            with col_d:
                bend_util = r["bending_stress_MPa"] / max(yield_Pa / 1e6, 1e-12)
                st.pyplot(plot_interaction(r["utilization"], bend_util), use_container_width=True)

        if show_curves:
            df_curve = stress_slenderness_dataframe(E_Pa, yield_Pa)
            st.pyplot(plot_stress_slenderness(df_curve, d["slenderness_lambda"], r["critical_stress_MPa"]), use_container_width=True)
            with st.expander("Curve data"):
                st.dataframe(df_curve, use_container_width=True)

        if show_capacity:
            df_len = capacity_vs_length_dataframe(inp)
            if not df_len.empty:
                st.pyplot(plot_capacity_length(df_len, inp.length_m, r["critical_load_kN"]), use_container_width=True)
                with st.expander("Capacity-length data"):
                    st.dataframe(df_len, use_container_width=True)

        if show_imperf:
            st.subheader("Imperfection sensitivity")
            df_imp = imperfection_sensitivity_dataframe(result, inp)
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            ax.plot(df_imp["Imperfection mm"], df_imp["Critical load kN"], label="critical load")
            ax.set_xlabel("Imperfection amplitude (mm)")
            ax.set_ylabel("Critical load (kN)")
            ax.set_title("Imperfection sensitivity")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.dataframe(df_imp, use_container_width=True)

# -----------------------------
# Export tab
# -----------------------------
with tab_export:
    st.subheader("Exportable artifacts")
    config = asdict(inp)
    payload = {
        "configuration": config,
        "section": result.section,
        "derived": result.derived,
        "results": result.results,
        "warnings": result.warnings,
        "errors": result.errors,
        "note": APP_NOTE,
    }
    st.download_button(
        "Download configuration + result JSON",
        data=json.dumps(payload, indent=2),
        file_name="elastic_stability_result.json",
        mime="application/json",
    )

    if not result.errors:
        section_df, derived_df, results_df = result_tables(result)
        csv_bundle = {
            "section_properties": section_df.to_csv(index=False),
            "derived_values": derived_df.to_csv(index=False),
            "primary_results": results_df.to_csv(index=False),
        }
        st.download_button(
            "Download primary results CSV",
            data=results_df.to_csv(index=False),
            file_name="elastic_stability_primary_results.csv",
            mime="text/csv",
        )
        report_md = build_markdown_report(result)
        st.download_button(
            "Download markdown report",
            data=report_md,
            file_name="elastic_stability_report.md",
            mime="text/markdown",
        )

    st.subheader("How to run locally")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
