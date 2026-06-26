from __future__ import annotations

import numpy as np
import pandas as pd


def _gate_name(station_name: str) -> str:
    if "闸" in station_name:
        return station_name.split("闸")[0] + "闸"
    if "上" in station_name:
        return station_name.split("上")[0]
    if "下" in station_name:
        return station_name.split("下")[0]
    return station_name


def build_daily_gate_state(hydrology: pd.DataFrame) -> pd.DataFrame:
    frame = hydrology.copy()
    frame["gate"] = frame["station_name"].map(_gate_name)
    frame["position"] = np.where(frame["station_name"].str.contains("上"), "up", "down")

    levels = (
        frame.pivot_table(index=["date", "gate"], columns="position", values="level_value", aggfunc="mean")
        .reset_index()
        .rename(columns={"up": "up_level", "down": "down_level"})
    )
    flow = frame.groupby(["date", "gate"], as_index=False)["flow"].max()
    merged = levels.merge(flow, on=["date", "gate"], how="outer").sort_values(["gate", "date"]).reset_index(drop=True)
    merged["head_diff"] = merged["up_level"] - merged["down_level"]
    merged["flow"] = merged["flow"].fillna(0.0)
    merged["month"] = merged["date"].dt.month
    merged["seasonal_stage"] = np.where(merged["month"].isin([6, 7, 8, 9]), "flood", "non_flood")

    def classify_direction(row: pd.Series) -> str:
        if pd.notna(row["head_diff"]) and row["head_diff"] < -0.05:
            return "inflow"
        if pd.notna(row["head_diff"]) and row["head_diff"] > 0.05:
            return "outflow"
        return "stagnant"

    merged["direction"] = merged.apply(classify_direction, axis=1)
    merged["exchange_intensity"] = np.tanh((merged["flow"].abs() / 80.0) + merged["head_diff"].abs().fillna(0.0) / 3.0)
    merged["flow_memory"] = merged.groupby("gate")["flow"].transform(lambda s: s.fillna(0.0).ewm(alpha=0.25).mean())
    merged["head_memory"] = merged.groupby("gate")["head_diff"].transform(lambda s: s.fillna(0.0).ewm(alpha=0.25).mean())
    merged["flow_variation"] = merged.groupby("gate")["flow"].transform(lambda s: s.fillna(0.0).diff().abs().fillna(0.0))

    def classify_mode(row: pd.Series) -> str:
        if row["direction"] == "inflow":
            return "backflow_recharge"
        if row["exchange_intensity"] >= 0.7 and row["direction"] == "outflow":
            return "high_exchange_flush"
        if row["exchange_intensity"] < 0.35 and row["flow_memory"] < 20.0:
            return "low_exchange_stagnation"
        return "outflow_dominant"

    merged["gate_mode"] = merged.apply(classify_mode, axis=1)
    merged["connectivity_factor"] = np.clip(0.2 + 0.6 * merged["exchange_intensity"], 0.0, 1.0)
    return merged


def build_hydraulic_proxies(gate_state: pd.DataFrame) -> pd.DataFrame:
    frame = gate_state.copy().sort_values(["gate", "date"]).reset_index(drop=True)
    frame["exchange_intensity"] = np.tanh((frame["flow"].abs() / 80.0) + frame["head_diff"].abs().fillna(0.0) / 3.0)
    frame["flow_memory"] = frame.groupby("gate")["flow"].transform(lambda s: s.fillna(0.0).ewm(alpha=0.25).mean())
    frame["head_memory"] = frame.groupby("gate")["head_diff"].transform(lambda s: s.fillna(0.0).ewm(alpha=0.25).mean())
    frame["historical_state"] = np.tanh(frame["flow_memory"].abs() / 100.0 + frame["head_memory"].abs() / 2.5)
    frame["seasonal_gate_factor"] = np.where(frame["date"].dt.month.isin([6, 7, 8, 9]), 1.0, 0.7)
    frame["gate_transition_index"] = frame.groupby("gate")["gate_mode"].transform(lambda s: s.ne(s.shift(1)).astype(float))
    frame["hydraulic_state_index"] = np.tanh(
        0.45 * frame["exchange_intensity"]
        + 0.25 * np.tanh(frame["head_diff"].abs().fillna(0.0))
        + 0.20 * frame["historical_state"]
        + 0.10 * frame["seasonal_gate_factor"]
    )
    frame["mixing_intensity"] = np.clip(
        0.55 * frame["exchange_intensity"] + 0.25 * frame["historical_state"] + 0.20 * frame["seasonal_gate_factor"],
        0.0,
        1.0,
    )
    frame["hydraulic_memory"] = frame.groupby("gate")["hydraulic_state_index"].transform(
        lambda s: s.fillna(0.0).ewm(alpha=0.3).mean()
    )
    frame["residence_time_proxy"] = 1.0 / (frame["mixing_intensity"] + 0.08)
    frame["backwater_index"] = np.where(frame["direction"].eq("inflow"), frame["head_diff"].abs(), 0.0)
    frame["stagnation_index"] = 1.0 - frame["mixing_intensity"]
    frame["lake_level_memory"] = frame.groupby("gate")["head_diff"].transform(lambda s: s.fillna(0.0).ewm(alpha=0.35).mean())
    return frame.loc[
        :,
        [
            "date",
            "gate",
            "flow",
            "head_diff",
            "direction",
            "gate_mode",
            "connectivity_factor",
            "exchange_intensity",
            "mixing_intensity",
            "hydraulic_state_index",
            "hydraulic_memory",
            "gate_transition_index",
            "residence_time_proxy",
            "backwater_index",
            "stagnation_index",
            "lake_level_memory",
        ],
    ]
