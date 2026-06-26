from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd


SITE_TARGETS = [
    "排水闸",
    "湖心A(洪湖湖心A)",
    "湖心B(洪湖湖心B)",
    "杨柴湖",
]


@dataclass(frozen=True)
class SourcePaths:
    meteorology: str
    water_quality: str
    swat: str
    hydrology: str


def normalize_water_quality_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={
            "日期": "date",
            "断面名称": "site",
            "总氮(mg/L)": "target_tn",
            "总磷(mg/L)": "target_tp",
        }
    ).loc[:, ["date", "site", "target_tn", "target_tp"]]
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    return normalized.dropna(subset=["date", "site"]).sort_values(["site", "date"]).reset_index(drop=True)


def normalize_meteorology_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={
            "date": "date",
            "precipitation": "precipitation",
            "tmin": "tmin",
            "tmax": "tmax",
            "wind_speed": "wind_speed",
            "solar_radiation": "solar_radiation",
        }
    ).copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["temp_mean"] = (normalized["tmin"] + normalized["tmax"]) / 2.0
    daily = (
        normalized.groupby("date", as_index=False)[
            ["precipitation", "temp_mean", "wind_speed", "solar_radiation"]
        ]
        .mean()
        .sort_values("date")
    )
    return daily


def _extract_level(raw: object) -> tuple[float | None, str | None]:
    if pd.isna(raw):
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    level = float(match.group(1)) if match else None
    trend = None
    if "↑" in text:
        trend = "up"
    elif "↓" in text:
        trend = "down"
    return level, trend


def normalize_hydrology_frame(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.rename(columns={"时间": "date", "站名": "station_name", "总过闸流量": "flow"}).copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()

    up = base.loc[:, ["date", "station_name", "闸上水位", "flow"]].rename(columns={"闸上水位": "raw_level"})
    down = base.loc[:, ["date", "station_name", "闸下水位", "flow"]].rename(columns={"闸下水位": "raw_level"})
    merged = pd.concat([up, down], ignore_index=True)
    merged = merged.dropna(subset=["raw_level", "flow"], how="all").copy()
    levels = merged["raw_level"].apply(_extract_level)
    merged["level_value"] = [item[0] for item in levels]
    merged["level_trend"] = [item[1] for item in levels]
    merged["station_name"] = merged["station_name"].astype(str).str.strip()
    merged = merged.dropna(subset=["level_value"]).drop(columns=["raw_level"]).reset_index(drop=True)
    return merged


def load_water_quality(path: str, sites: Iterable[str] = SITE_TARGETS) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="日平均值")
    normalized = normalize_water_quality_frame(frame)
    return normalized[normalized["site"].isin(list(sites))].reset_index(drop=True)


def load_meteorology(path: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="weather_2023")
    return normalize_meteorology_frame(frame)


def load_swat(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    reach = pd.read_excel(path, sheet_name="Reach")
    sub = pd.read_excel(path, sheet_name="Sub")

    reach = reach.rename(
        columns={
            "DATE": "date",
            "RCH_ID": "rch_id",
            "FLOW_OUT": "flow_out",
            "TOTAL_N_kg": "total_n_kg",
            "TOTAL_P_kg": "total_p_kg",
            "AREA_km2": "area_km2",
        }
    )
    reach["date"] = pd.to_datetime(reach["date"]).dt.normalize()
    reach = reach.loc[:, ["date", "rch_id", "area_km2", "flow_out", "total_n_kg", "total_p_kg"]]

    sub = sub.rename(
        columns={
            "DATE": "date",
            "SUB_ID": "sub_id",
            "V_3": "sub_area_km2",
            "PRECIP_mm": "precip_mm",
            "WYLD_mm": "wyld_mm",
            "ORG_N_kgha": "org_n_kgha",
            "ORG_P_kgha": "org_p_kgha",
            "NO3_SURQ": "no3_surq",
            "SOL_P": "sol_p",
            "SEDP_kgha": "sedp_kgha",
        }
    )
    sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()
    sub = sub.loc[
        :,
        [
            "date",
            "sub_id",
            "sub_area_km2",
            "precip_mm",
            "wyld_mm",
            "org_n_kgha",
            "org_p_kgha",
            "no3_surq",
            "sol_p",
            "sedp_kgha",
        ],
    ]
    return reach, sub


def load_hydrology(path: str) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    frames = [pd.read_excel(path, sheet_name=sheet) for sheet in workbook.sheet_names]
    combined = pd.concat(frames, ignore_index=True)
    return normalize_hydrology_frame(combined)
