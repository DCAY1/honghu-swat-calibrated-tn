"""Extended pipeline that augments the base model dataset with external data.

Usage:
    from honghu_nps.extended_pipeline import build_extended_model_dataset
    extended_dataset = build_extended_model_dataset(base_model_dataset)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .external_data import (
    build_county_outlet_features,
    build_extended_meteorology,
    build_extended_water_quality_context,
    build_longterm_flow_features,
    DEFAULT_AUTO_MONITORING,
    DEFAULT_DISCHARGE_DIR,
    DEFAULT_LONGTERM_FLOW,
    DEFAULT_WX_DIR,
)

# Extended feature columns to be added to the model dataset
EXTENDED_FEATURE_COLUMNS = [
    # Climatology from national weather station (1957-2015 daily means)
    "evap_mm_clim",
    "evap_mm_std",
    "gst_avg_clim",
    "sunshine_hours_clim",
    "sunshine_hours_std",
    "temp_mean_wx_clim",
    "temp_mean_wx_std",
    "temp_range_wx_clim",
    "precip_wx_clim",
    "precip_wx_std",
    "precip_freq_clim",
    "rhum_avg_clim",
    "rhum_avg_std",
    "wind_avg_clim",
    "wind_avg_std",
    "prs_avg_clim",
    # Hourly monitoring daily statistics
    "hourly_wt_mean",
    "hourly_wt_range",
    "hourly_do_mean",
    "hourly_do_range",
    "hourly_ph_mean",
    "hourly_cond_mean",
    "hourly_turb_mean",
    "hourly_tn_mean",
    "hourly_tn_range",
    "hourly_completeness",
    # County outlet spatial aggregation
    "cso_tn_mean",
    "cso_tn_std",
    "cso_tn_count",
    "cso_tp_mean",
    "cso_nh3n_mean",
    "cso_cod_mean",
    # Long-term flow climatology
    "flow_clim_mean",
    "flow_clim_std",
    "flow_clim_p10",
    "flow_clim_p50",
    "flow_clim_p90",
    # Cross-derived features
    "temp_consensus",
    "evap_precip_ratio",
    "photo_activity_proxy",
    "flow_clim_iqr",
    "tn_diurnal_range",
]


def build_extended_model_dataset(
    base_dataset: pd.DataFrame,
    wx_dir: str | Path | None = None,
    auto_mon_path: str | Path | None = None,
    discharge_dir: str | Path | None = None,
    longterm_flow_path: str | Path | None = None,
) -> pd.DataFrame:
    """Augment the base model_dataset_daily with external features.

    Parameters
    ----------
    base_dataset : pd.DataFrame
        The model_dataset_daily from run_stage_prepare.
    wx_dir : path, optional
        National weather station directory.
    auto_mon_path : path, optional
        Automatic hourly monitoring Excel file.
    discharge_dir : path, optional
        County discharge outlet directory.
    longterm_flow_path : path, optional
        Long-term daily flow Excel file.

    Returns
    -------
    pd.DataFrame
        base_dataset with additional feature columns merged on date.
    """
    if base_dataset.empty:
        return base_dataset.copy()

    wx_dir = Path(wx_dir) if wx_dir else DEFAULT_WX_DIR
    auto_mon_path = Path(auto_mon_path) if auto_mon_path else DEFAULT_AUTO_MONITORING
    discharge_dir = Path(discharge_dir) if discharge_dir else DEFAULT_DISCHARGE_DIR
    longterm_flow_path = Path(longterm_flow_path) if longterm_flow_path else DEFAULT_LONGTERM_FLOW

    result = base_dataset.copy()

    # --- Layer 1: Extended meteorology from national weather station ---
    if wx_dir.exists():
        met_cols = [c for c in result.columns if c in {"precip", "precipitation", "temp_mean", "tmin", "tmax"}]
        met_frame = result[["date"] + met_cols].drop_duplicates(subset=["date"]).copy()
        if "precipitation" not in met_frame.columns and "precip" in met_frame.columns:
            met_frame["precipitation"] = met_frame["precip"]
        elif "precip" in met_frame.columns and "precipitation" not in met_frame.columns:
            met_frame["precipitation"] = met_frame["precip"]
        ext_met = build_extended_meteorology(met_frame, wx_dir)
        wx_cols = [c for c in ext_met.columns if c not in result.columns and c != "date"]
        if wx_cols:
            result = result.merge(ext_met[["date"] + wx_cols], on="date", how="left")
            print(f"[extended_pipeline] Added {len(wx_cols)} weather station columns")

    # --- Layer 2: Hourly monitoring daily statistics ---
    if auto_mon_path.exists():
        hourly = build_extended_water_quality_context(
            result[["date", "target_tn_day"]].drop_duplicates(subset=["date"]),
            auto_mon_path,
        )
        hourly_cols = [c for c in hourly.columns if c not in result.columns and c != "date"]
        if hourly_cols:
            result = result.merge(hourly[["date"] + hourly_cols], on="date", how="left")
            print(f"[extended_pipeline] Added {len(hourly_cols)} hourly monitoring columns")

    # --- Layer 3: County outlet aggregation ---
    if discharge_dir.exists():
        outlet_features = build_county_outlet_features(result[["date"]], discharge_dir)
        outlet_cols = [c for c in outlet_features.columns if c not in result.columns]
        if outlet_cols:
            result = result.merge(outlet_features, on="date", how="left")
            print(f"[extended_pipeline] Added {len(outlet_cols)} county outlet columns")

    # --- Layer 4: Long-term flow climatology ---
    if longterm_flow_path.exists():
        flow_features = build_longterm_flow_features(result[["date"]], longterm_flow_path)
        flow_cols = [c for c in flow_features.columns if c not in result.columns]
        if flow_cols:
            result = result.merge(flow_features, on="date", how="left")
            print(f"[extended_pipeline] Added {len(flow_cols)} flow climatology columns")

    # --- Derived cross-features ---
    result = _add_cross_derived_features(result)

    return result


def _add_cross_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add features that combine external data with existing pipeline features."""
    out = frame.copy()

    # Temperature consensus: blend model temp with station climatology
    if "temp_mean_wx_clim" in out.columns and "temp_mean" in out.columns:
        out["temp_consensus"] = out["temp_mean"].fillna(out["temp_mean_wx_clim"])
    elif "temp_mean_wx_clim" in out.columns:
        out["temp_consensus"] = out["temp_mean_wx_clim"]
    elif "temp_mean" in out.columns:
        out["temp_consensus"] = out["temp_mean"]

    # Evaporation-driven concentration proxy
    if "evap_mm_clim" in out.columns and "precip" in out.columns:
        out["evap_precip_ratio"] = np.where(
            out["precip"].fillna(0) > 1.0,
            out["evap_mm_clim"].fillna(0) / out["precip"].clip(lower=0.1),
            out["evap_mm_clim"].fillna(0),
        )

    # Photodegradation proxy: sunshine climatology × temperature
    if "sunshine_hours_clim" in out.columns and "temp_consensus" in out.columns:
        out["photo_activity_proxy"] = (
            out["sunshine_hours_clim"].fillna(0) * out["temp_consensus"].fillna(0) / 24.0
        )

    # Long-term flow IQR as uncertainty indicator
    if all(c in out.columns for c in ["flow_clim_p50", "flow_clim_p10", "flow_clim_p90"]):
        out["flow_clim_iqr"] = out["flow_clim_p90"].fillna(0) - out["flow_clim_p10"].fillna(0)

    # County outlet TN as supplementary loading indicator (sparse, fill with NaN)
    if "cso_tn_mean" in out.columns and "L0" in out.columns:
        out["cso_tn_normalized"] = out["cso_tn_mean"].fillna(np.nan) / (out["L0"].fillna(1.0) + 1e-6)

    # Hourly TN range as diurnal variation indicator
    if "hourly_tn_range" in out.columns:
        out["tn_diurnal_range"] = out["hourly_tn_range"].fillna(0)

    # Precipitation anomaly: diff from climatology
    if "precip_wx_clim" in out.columns and "precip" in out.columns:
        out["precip_anomaly"] = out["precip"].fillna(0) - out["precip_wx_clim"].fillna(0)

    return out


def print_data_summary(frame: pd.DataFrame) -> None:
    """Print a summary of the extended dataset."""
    print("=" * 60)
    print("Extended Dataset Summary")
    print("=" * 60)
    print(f"Rows: {len(frame)}")
    print(f"Columns: {len(frame.columns)}")

    # Column categories
    base_cols = [
        "date", "target_tn_day", "mask_y", "site",
        "L0", "L_aux_1", "L_aux_2", "L_eff",
        "precip", "api3", "api7",
        "state_label", "state_prob_s1", "state_prob_s2", "state_prob_s3", "state_prob_s4",
        "delta_h", "Qg_sgn", "Qg_abs", "connect", "mix", "res", "mem", "exchange_strength",
    ]
    clim_cols = [c for c in frame.columns if c.endswith("_clim") or c.endswith("_std") or c == "precip_freq_clim"]
    hourly_cols = [c for c in frame.columns if c.startswith("hourly_")]
    outlet_cols = [c for c in frame.columns if c.startswith("cso_")]
    flow_cols = [c for c in frame.columns if c.startswith("flow_clim")]
    derived_cols = [c for c in frame.columns if any(c.startswith(p) for p in [
        "temp_consensus", "evap_precip_ratio", "photo_activity_proxy",
        "flow_clim_iqr", "cso_tn_normalized", "tn_diurnal_range", "precip_anomaly",
    ])]

    present_base = [c for c in base_cols if c in frame.columns]
    print(f"\nBase features: {len(present_base)} present of {len(base_cols)} defined")
    print(f"Weather climatology: {len(clim_cols)} columns — {clim_cols}")
    print(f"Hourly monitoring: {len(hourly_cols)} columns — {hourly_cols}")
    print(f"County outlets: {len(outlet_cols)} columns — {outlet_cols}")
    print(f"Flow climatology: {len(flow_cols)} columns — {flow_cols}")
    print(f"Cross-derived: {len(derived_cols)} columns — {derived_cols}")

    # Date range
    if "date" in frame.columns:
        print(f"\nDate range: {frame['date'].min()} ~ {frame['date'].max()}")

    # Target coverage
    if "target_tn_day" in frame.columns:
        valid_target = frame["target_tn_day"].notna().sum()
        print(f"Valid target observations: {valid_target} / {len(frame)} ({100*valid_target/max(len(frame),1):.1f}%)")

    print("=" * 60)
