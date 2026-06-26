from __future__ import annotations

import numpy as np
import pandas as pd


def _sub_daily_totals(swat_sub: pd.DataFrame) -> pd.DataFrame:
    frame = swat_sub.copy()
    area_ha = frame["sub_area_km2"].fillna(0.0) * 100.0
    frame["source_tn_kg"] = area_ha * (frame["org_n_kgha"].fillna(0.0) + frame["no3_surq"].fillna(0.0))
    frame["source_tp_kg"] = area_ha * (
        frame["org_p_kgha"].fillna(0.0) + frame["sol_p"].fillna(0.0) + frame["sedp_kgha"].fillna(0.0)
    )
    return (
        frame.groupby("date", as_index=False)[["source_tn_kg", "source_tp_kg", "precip_mm", "wyld_mm"]]
        .sum()
        .sort_values("date")
    )


def _reach_daily_totals(swat_reach: pd.DataFrame) -> pd.DataFrame:
    frame = swat_reach.copy()
    return (
        frame.groupby("date", as_index=False)[["flow_out", "total_n_kg", "total_p_kg"]]
        .sum()
        .sort_values("date")
    )


def _fractional_lag_route(series: pd.Series, lag_days: pd.Series) -> pd.Series:
    routed = pd.Series(index=series.index, dtype=float)
    for idx in range(len(series)):
        lag = float(lag_days.iloc[idx])
        lag = max(lag, 0.0)
        lower = int(np.floor(lag))
        upper = int(np.ceil(lag))
        frac = lag - lower
        lower_idx = idx - lower
        upper_idx = idx - upper
        lower_val = float(series.iloc[lower_idx]) if lower_idx >= 0 else 0.0
        upper_val = float(series.iloc[upper_idx]) if upper_idx >= 0 else 0.0
        routed.iloc[idx] = (1.0 - frac) * lower_val + frac * upper_val
    return routed.fillna(0.0)


def _route_by_gate(frame: pd.DataFrame, value_col: str) -> pd.Series:
    routed = pd.Series(index=frame.index, dtype=float)
    for _, gate_frame in frame.groupby("gate"):
        routed.loc[gate_frame.index] = _fractional_lag_route(gate_frame[value_col], gate_frame["travel_lag_days"]).to_numpy()
    return routed.sort_index().fillna(0.0)


def _scenario_columns(central: pd.Series, uncertainty: np.ndarray) -> tuple[pd.Series, pd.Series, pd.Series]:
    conservative = central * (1.0 - uncertainty)
    responsive = central * (1.0 + uncertainty)
    return conservative.clip(lower=0.0), central.clip(lower=0.0), responsive.clip(lower=0.0)


def _prepare_boundary_generator_inputs(
    swat_sub: pd.DataFrame,
    swat_reach: pd.DataFrame,
    gate_state: pd.DataFrame,
    meteorology: pd.DataFrame,
) -> pd.DataFrame:
    source = _sub_daily_totals(swat_sub)
    reach = _reach_daily_totals(swat_reach)
    meteo = meteorology.rename(columns={"precipitation": "meteo_precip"}).copy()
    gate = gate_state.copy()

    merged = source.merge(reach, on="date", how="outer").merge(meteo, on="date", how="left")
    merged = gate.merge(merged, on="date", how="left").sort_values(["gate", "date"]).reset_index(drop=True)
    merged["meteo_precip"] = merged["meteo_precip"].fillna(merged["precip_mm"]).fillna(0.0)
    merged["wyld_mm"] = merged["wyld_mm"].fillna(0.0)
    merged["source_tn_kg"] = merged["source_tn_kg"].fillna(0.0)
    merged["source_tp_kg"] = merged["source_tp_kg"].fillna(0.0)
    merged["total_n_kg"] = merged["total_n_kg"].fillna(0.0)
    merged["total_p_kg"] = merged["total_p_kg"].fillna(0.0)
    return merged


def _compute_boundary_generator(frame: pd.DataFrame) -> pd.DataFrame:
    merged = frame.copy()
    merged["L_swat_tn"] = 0.5 * merged["source_tn_kg"] + 0.5 * merged["total_n_kg"]
    merged["L_swat_tp"] = 0.5 * merged["source_tp_kg"] + 0.5 * merged["total_p_kg"]
    merged["swat_tn_raw"] = merged["L_swat_tn"]
    merged["swat_tp_raw"] = merged["L_swat_tp"]

    storm_score = np.clip((merged["meteo_precip"] / 20.0) + (merged["wyld_mm"] / 5.0), 0.0, None)
    merged["event_factor"] = 1.0 + 0.25 * storm_score
    merged["event_flag"] = (storm_score >= 1.0).astype(int)
    merged["travel_lag_days"] = np.select(
        [
            merged["event_flag"].eq(1) & merged["connectivity_factor"].fillna(0.0).ge(0.6),
            merged["event_flag"].eq(1),
            merged["connectivity_factor"].fillna(0.0).ge(0.7),
        ],
        [1.0, 2.0, 2.5],
        default=3.5,
    )
    merged["storm_amplified_tn"] = merged["L_swat_tn"] * merged["event_factor"]
    merged["storm_amplified_tp"] = merged["L_swat_tp"] * merged["event_factor"]

    conn = merged["connectivity_factor"].fillna(0.3)
    merged["hydraulic_connectivity_state"] = conn
    direction_factor = np.select(
        [
            merged["direction"].eq("inflow"),
            merged["direction"].eq("stagnant"),
        ],
        [1.0, 0.45],
        default=0.7,
    )
    merged["hydraulic_directionality_factor"] = direction_factor
    seasonal_factor = np.where(merged["date"].dt.month.isin([6, 7, 8, 9]), 1.1, 0.95)
    merged["hydraulic_seasonal_factor"] = seasonal_factor
    merged["routed_tn_kg"] = _route_by_gate(merged, "storm_amplified_tn")
    merged["routed_tp_kg"] = _route_by_gate(merged, "storm_amplified_tp")
    merged["transit_storage_tn"] = (
        merged.groupby("gate")["storm_amplified_tn"].transform(lambda s: s.fillna(0.0).ewm(alpha=0.35).mean())
        * (1.0 - conn)
    )
    merged["transit_storage_tp"] = (
        merged.groupby("gate")["storm_amplified_tp"].transform(lambda s: s.fillna(0.0).ewm(alpha=0.35).mean())
        * (1.0 - conn)
    )
    merged["available_tn_kg"] = merged["routed_tn_kg"] + 0.25 * merged["transit_storage_tn"]
    merged["available_tp_kg"] = merged["routed_tp_kg"] + 0.25 * merged["transit_storage_tp"]
    raw_effective_tn = merged["available_tn_kg"] * conn * direction_factor * seasonal_factor
    raw_effective_tp = merged["available_tp_kg"] * conn * direction_factor * seasonal_factor
    merged["effective_input_tn"] = np.minimum(raw_effective_tn, merged["available_tn_kg"]).clip(lower=0.0)
    merged["effective_input_tp"] = np.minimum(raw_effective_tp, merged["available_tp_kg"]).clip(lower=0.0)
    merged["effective_boundary_tn_central"] = merged["effective_input_tn"]
    merged["effective_boundary_tp_central"] = merged["effective_input_tp"]
    merged["mass_balance_tn_ratio"] = merged["effective_input_tn"] / merged["available_tn_kg"].replace(0.0, np.nan)
    merged["mass_balance_tp_ratio"] = merged["effective_input_tp"] / merged["available_tp_kg"].replace(0.0, np.nan)
    merged["mass_balance_tn_ratio"] = merged["mass_balance_tn_ratio"].fillna(0.0).clip(0.0, 1.0)
    merged["mass_balance_tp_ratio"] = merged["mass_balance_tp_ratio"].fillna(0.0).clip(0.0, 1.0)
    uncertainty = np.where(merged["event_flag"].eq(1), 0.35, 0.2)
    uncertainty += np.where(merged["gate_mode"].eq("backflow_recharge"), 0.1, 0.0)
    tn_conservative, tn_central, tn_responsive = _scenario_columns(merged["effective_boundary_tn_central"], uncertainty)
    tp_conservative, tp_central, tp_responsive = _scenario_columns(merged["effective_boundary_tp_central"], uncertainty)
    merged["effective_boundary_tn_conservative"] = tn_conservative
    merged["effective_boundary_tn_responsive"] = tn_responsive
    merged["effective_boundary_tp_conservative"] = tp_conservative
    merged["effective_boundary_tp_responsive"] = tp_responsive
    merged["prior_tn_conservative"] = tn_conservative
    merged["prior_tn_central"] = tn_central
    merged["prior_tn_responsive"] = tn_responsive
    merged["prior_tp_conservative"] = tp_conservative
    merged["prior_tp_central"] = tp_central
    merged["prior_tp_responsive"] = tp_responsive
    return merged


def build_boundary_prior(
    swat_sub: pd.DataFrame,
    swat_reach: pd.DataFrame,
    gate_state: pd.DataFrame,
    meteorology: pd.DataFrame,
) -> pd.DataFrame:
    """Generate state-dependent effective boundary priors from SWAT load, hydraulics, and event forcing."""
    merged = _compute_boundary_generator(_prepare_boundary_generator_inputs(swat_sub, swat_reach, gate_state, meteorology))

    return merged.loc[
        :,
        [
            "date",
            "gate",
            "L_swat_tn",
            "L_swat_tp",
            "swat_tn_raw",
            "swat_tp_raw",
            "event_factor",
            "storm_amplified_tn",
            "storm_amplified_tp",
            "travel_lag_days",
            "connectivity_factor",
            "event_flag",
            "hydraulic_connectivity_state",
            "hydraulic_directionality_factor",
            "hydraulic_seasonal_factor",
            "routed_tn_kg",
            "routed_tp_kg",
            "effective_input_tn",
            "effective_input_tp",
            "effective_boundary_tn_conservative",
            "effective_boundary_tn_central",
            "effective_boundary_tn_responsive",
            "effective_boundary_tp_conservative",
            "effective_boundary_tp_central",
            "effective_boundary_tp_responsive",
            "mass_balance_tn_ratio",
            "mass_balance_tp_ratio",
            "prior_tn_conservative",
            "prior_tn_central",
            "prior_tn_responsive",
            "prior_tp_conservative",
            "prior_tp_central",
            "prior_tp_responsive",
        ],
    ]


def build_boundary_prior_identifiability(
    swat_sub: pd.DataFrame,
    swat_reach: pd.DataFrame,
    gate_state: pd.DataFrame,
    meteorology: pd.DataFrame,
) -> pd.DataFrame:
    scenarios = {
        "baseline": (
            swat_sub.copy(),
            swat_reach.copy(),
            gate_state.copy(),
            meteorology.copy(),
        ),
        "swat_weakened": (
            swat_sub.assign(
                org_n_kgha=swat_sub["org_n_kgha"] * 0.5,
                org_p_kgha=swat_sub["org_p_kgha"] * 0.5,
                no3_surq=swat_sub["no3_surq"] * 0.5,
                sol_p=swat_sub["sol_p"] * 0.5,
                sedp_kgha=swat_sub["sedp_kgha"] * 0.5,
            ),
            swat_reach.assign(total_n_kg=swat_reach["total_n_kg"] * 0.5, total_p_kg=swat_reach["total_p_kg"] * 0.5),
            gate_state.copy(),
            meteorology.copy(),
        ),
        "connectivity_switch": (
            swat_sub.copy(),
            swat_reach.copy(),
            gate_state.assign(connectivity_factor=1.0, direction="outflow", gate_mode="high_exchange_flush"),
            meteorology.copy(),
        ),
        "event_off": (
            swat_sub.assign(precip_mm=0.0, wyld_mm=0.0),
            swat_reach.copy(),
            gate_state.copy(),
            meteorology.assign(precipitation=0.0),
        ),
    }
    frames: list[pd.DataFrame] = []
    for scenario, (scenario_sub, scenario_reach, scenario_gate, scenario_meteo) in scenarios.items():
        frame = build_boundary_prior(scenario_sub, scenario_reach, scenario_gate, scenario_meteo)
        frame["scenario"] = scenario
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
