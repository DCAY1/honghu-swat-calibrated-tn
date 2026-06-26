from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from honghu_nps.data_loading import SourcePaths, load_hydrology, load_meteorology, load_swat, load_water_quality
from honghu_nps.hydraulics import build_daily_gate_state, build_hydraulic_proxies
from honghu_nps.modeling import build_model_upgrade_benchmarks, fit_predictive_system
from honghu_nps.prior import build_boundary_prior, build_boundary_prior_identifiability
from honghu_nps.reporting import summarize_prediction_intervals


@dataclass(frozen=True)
class PipelineOutputs:
    water_quality: pd.DataFrame
    gate_state: pd.DataFrame
    hydraulic_proxies: pd.DataFrame
    boundary_prior: pd.DataFrame
    boundary_prior_identifiability: pd.DataFrame
    response_dataset: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    benchmark_metrics: pd.DataFrame
    interval_summary: pd.DataFrame


def _canonical_site(site: str) -> str:
    if site.startswith("湖心A"):
        return "湖心A(洪湖湖心A)"
    if site.startswith("湖心B"):
        return "湖心B(洪湖湖心B)"
    return site


def _assign_gate(site: str) -> str:
    if site == "排水闸":
        return "新滩口闸"
    return "新滩口闸"


def _add_site_lags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["site", "date"]).copy()
    out["site_lagged_tn_1d"] = out.groupby("site")["target_tn"].shift(1)
    out["site_lagged_tp_1d"] = out.groupby("site")["target_tp"].shift(1)
    out["site_lagged_tn_3d"] = out.groupby("site")["target_tn"].shift(3)
    out["site_lagged_tp_3d"] = out.groupby("site")["target_tp"].shift(3)
    return out


def build_dynamic_site_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["site", "date"]).copy()
    group = out.groupby("site", group_keys=False)
    if "directionality_index" not in out.columns:
        out["directionality_index"] = out.get("direction", pd.Series(index=out.index, dtype=object)).map(
            {"inflow": -1.0, "stagnant": 0.0, "outflow": 1.0}
        ).fillna(0.0)

    for days in (1, 3, 7):
        out[f"prior_tn_central_lag_{days}"] = group["prior_tn_central"].shift(days)
        out[f"prior_tp_central_lag_{days}"] = group["prior_tp_central"].shift(days)

    out["storm_amplified_tn_roll3"] = group["storm_amplified_tn"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["storm_amplified_tp_roll3"] = group["storm_amplified_tp"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["storm_amplified_tn_roll7"] = group["storm_amplified_tn"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["storm_amplified_tp_roll7"] = group["storm_amplified_tp"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["event_count_7d"] = group["event_flag"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    out["event_precip_7d"] = group["precip"].transform(lambda s: s.where(out.loc[s.index, "event_flag"].eq(1), 0.0).rolling(7, min_periods=1).sum())

    def _lag_adjusted(series_name: str) -> pd.Series:
        adjusted = pd.Series(index=out.index, dtype=float)
        for site, site_frame in out.groupby("site"):
            lag = int(round(site_frame["travel_lag_days"].dropna().median())) if site_frame["travel_lag_days"].notna().any() else 1
            lag = max(min(lag, 14), 1)
            adjusted.loc[site_frame.index] = site_frame[series_name].shift(lag)
        return adjusted

    out["site_travel_lag_adjusted_prior_tn_central"] = _lag_adjusted("prior_tn_central")
    out["site_travel_lag_adjusted_prior_tp_central"] = _lag_adjusted("prior_tp_central")
    trimmed_columns = [
        "hydraulic_state_index",
        "gate_transition_index",
        "backwater_index",
        "stagnation_index",
        "lake_level_memory",
        "exchange_intensity",
        "flow",
        "head_diff",
    ]
    out = out.drop(columns=[column for column in trimmed_columns if column in out.columns])
    return out


def build_daily_response_dataset(
    water_quality: pd.DataFrame,
    boundary_prior: pd.DataFrame,
    hydraulic_proxies: pd.DataFrame,
    meteorology: pd.DataFrame,
) -> pd.DataFrame:
    wq = water_quality.copy()
    wq["site"] = wq["site"].map(_canonical_site)
    wq["gate"] = wq["site"].map(_assign_gate)

    hydraulics = hydraulic_proxies.copy()
    prior = boundary_prior.copy()
    meteo = meteorology.rename(columns={"precipitation": "precip"}).copy()
    merged = (
        wq.merge(prior, on=["date", "gate"], how="left")
        .merge(hydraulics, on=["date", "gate"], how="left", suffixes=("", "_hyd"))
        .merge(meteo, on="date", how="left")
    )
    merged["temp_mean"] = merged["temp_mean"].ffill()
    merged["precip"] = merged["precip"].fillna(0.0)
    merged = _add_site_lags(merged)
    merged = build_dynamic_site_features(merged)
    merged = merged.dropna(
        subset=[
            "target_tn",
            "target_tp",
            "prior_tn_central",
            "prior_tp_central",
            "prior_tn_central_lag_1",
            "prior_tp_central_lag_1",
            "site_travel_lag_adjusted_prior_tn_central",
            "site_travel_lag_adjusted_prior_tp_central",
            "connectivity_factor",
            "mixing_intensity",
            "hydraulic_memory",
            "residence_time_proxy",
            "site_lagged_tn_1d",
            "site_lagged_tp_1d",
        ]
    ).reset_index(drop=True)
    return merged


def run_full_pipeline(paths: SourcePaths, output_dir: str, split_date: str = "2023-10-01") -> PipelineOutputs:
    meteorology = load_meteorology(paths.meteorology)
    water_quality = load_water_quality(paths.water_quality)
    swat_reach, swat_sub = load_swat(paths.swat)
    hydrology = load_hydrology(paths.hydrology)

    gate_state = build_daily_gate_state(hydrology)
    hydraulic_proxies = build_hydraulic_proxies(gate_state)
    boundary_prior = build_boundary_prior(swat_sub, swat_reach, gate_state, meteorology)
    boundary_prior_identifiability = build_boundary_prior_identifiability(swat_sub, swat_reach, gate_state, meteorology)
    response_dataset = build_daily_response_dataset(water_quality, boundary_prior, hydraulic_proxies, meteorology)
    predictions, metrics = fit_predictive_system(response_dataset, split_date=split_date)
    benchmark_metrics = build_model_upgrade_benchmarks(response_dataset, split_date=split_date)
    interval_summary = summarize_prediction_intervals(predictions)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    water_quality.to_csv(out_dir / "water_quality_daily.csv", index=False)
    gate_state.to_csv(out_dir / "daily_gate_state.csv", index=False)
    hydraulic_proxies.to_csv(out_dir / "daily_hydraulic_proxies.csv", index=False)
    boundary_prior.to_csv(out_dir / "daily_boundary_prior.csv", index=False)
    boundary_prior_identifiability.to_csv(out_dir / "effective_prior_identifiability.csv", index=False)
    response_dataset.to_csv(out_dir / "daily_response_dataset.csv", index=False)
    predictions.to_csv(out_dir / "site_daily_prediction.csv", index=False)
    metrics.to_csv(out_dir / "site_model_metrics.csv", index=False)
    benchmark_metrics.to_csv(out_dir / "model_upgrade_benchmark_metrics.csv", index=False)
    interval_summary.to_csv(out_dir / "prediction_interval_regime_summary.csv", index=False)

    return PipelineOutputs(
        water_quality=water_quality,
        gate_state=gate_state,
        hydraulic_proxies=hydraulic_proxies,
        boundary_prior=boundary_prior,
        boundary_prior_identifiability=boundary_prior_identifiability,
        response_dataset=response_dataset,
        predictions=predictions,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        interval_summary=interval_summary,
    )
