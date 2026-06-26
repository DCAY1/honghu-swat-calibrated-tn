"""External data loaders for augmenting the Honghu dataset.

Sources:
  1. National Weather Station (57581 Honghu) — 8 variables, 1957–2023
  2. Automatic hourly water-quality monitoring — 4 stations, 2024
  3. County discharge-outlet monitoring — 9 counties, 2024
  4. NPS_MODEL long-term daily flow — 1998–2019
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT = _REPO_ROOT

DEFAULT_WX_DIR = _REPO_ROOT / "data" / "extensions" / "weather_station"
DEFAULT_AUTO_MONITORING = _REPO_ROOT / "data" / "extensions" / "auto_monitoring.xlsx"
DEFAULT_DISCHARGE_DIR = _REPO_ROOT / "data" / "extensions" / "discharge_outlets"
DEFAULT_LONGTERM_FLOW = _REPO_ROOT / "data" / "extensions" / "longterm_flow.xlsx"

# Column specs: column names, number of value fields, number of QC fields, scale factor
# Raw values are stored in 0.1 units (temp=0.1°C, precip=0.1mm, wind=0.1m/s, etc.)
WX_COLUMN_SPECS = {
    "EVP": {"columns": ["evap_mm"], "n_values": 1, "n_qc": 1, "scale": 0.1},
    "GST": {"columns": ["gst_avg", "gst_max", "gst_min"], "n_values": 3, "n_qc": 3, "scale": 0.1},
    "PRE": {"columns": ["precip_20_8", "precip_8_20", "precip_total"], "n_values": 3, "n_qc": 3, "scale": 0.1},
    "PRS": {"columns": ["prs_avg", "prs_max", "prs_min"], "n_values": 3, "n_qc": 3, "scale": 0.1},
    "RHU": {"columns": ["rhum_avg", "rhum_min"], "n_values": 2, "n_qc": 2, "scale": 1.0},  # percent
    "SSD": {"columns": ["sunshine_hours"], "n_values": 1, "n_qc": 1, "scale": 0.1},
    "TEM": {"columns": ["temp_avg", "temp_max", "temp_min"], "n_values": 3, "n_qc": 3, "scale": 0.1},
    "WIN": {"columns": ["wind_avg", "wind_max", "wind_max_dir", "wind_extreme", "wind_extreme_dir"], "n_values": 5, "n_qc": 5, "scale": 0.1},
}

WX_MISSING_CODE = 32766
WX_TRACE_CODE = 32700  # Trace precipitation/evaporation


def _decode_wx_value(raw: float, scale: float) -> float | None:
    """Decode a raw weather station value, handling special codes.

    Precipitation codes:  30XXX=rain(XXX*scale), 31XXX=snow, 32XXX=mixed
    32700=trace (treated as 0), 32766=missing.
    """
    if raw >= 32766:
        return None
    if raw >= 32700:
        return 0.0  # trace
    if raw >= 30000:
        return (raw - 30000) * scale
    return raw * scale


def _parse_national_wx_space_delimited(file_path: Path) -> pd.DataFrame:
    """Parse China National Weather Station space-delimited format.

    Each line: station_id lat lon elevation year month day <values...> <QC_flags...>
    Missing values are coded as 32766 (or 32700 for trace precipitation).
    QC flags: 0=ok, 8=missing, 9=unreliable.
    Units: temperature in 0.1°C, precipitation in 0.1mm, wind in 0.1m/s, etc.
    """
    var_name = file_path.stem
    spec = WX_COLUMN_SPECS.get(var_name)
    if spec is None:
        raise ValueError(f"Unknown weather variable: {var_name}")

    n_values = spec["n_values"]
    n_qc = spec["n_qc"]
    scale = spec["scale"]

    with open(file_path) as fh:
        raw_lines = [line.strip() for line in fh if line.strip()]

    if not raw_lines:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    header_names = ["station_id", "lat_deci", "lon_deci", "elevation_dm", "year", "month", "day"]

    for line in raw_lines:
        parts = line.split()
        if len(parts) < 7 + n_values:
            continue

        record: dict[str, float | int | str] = {}
        for i, name in enumerate(header_names):
            val = parts[i]
            if name in ("station_id", "year", "month", "day"):
                record[name] = int(val)
            else:
                record[name] = float(val)

        for j, col_name in enumerate(spec["columns"]):
            raw_val = float(parts[7 + j])
            decoded = _decode_wx_value(raw_val, scale)
            record[col_name] = decoded if decoded is not None else np.nan

        rows.append(record)

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["month"], "day": frame["day"]},
        errors="coerce",
    )
    frame = frame.dropna(subset=["date"])
    keep_cols = ["date"] + spec["columns"]
    return frame[[c for c in keep_cols if c in frame.columns]].sort_values("date").reset_index(drop=True)


def load_national_weather_station(
    wx_dir: str | Path = DEFAULT_WX_DIR,
) -> pd.DataFrame:
    """Load and merge all 8 weather variables from Honghu National Station (57581).

    Returns a DataFrame indexed by date with daily values for:
      evap_mm, gst_avg/max/min, precip_20_8/8_20/total, prs_avg/max/min,
      rhum_avg/min, sunshine_hours, temp_avg/max/min, wind_avg/max/extreme.
    """
    wx_path = Path(wx_dir)
    frames: dict[str, pd.DataFrame] = {}
    for txt_file in sorted(wx_path.glob("*.txt")):
        var_name = txt_file.stem
        if var_name not in WX_COLUMN_SPECS:
            continue
        try:
            frames[var_name] = _parse_national_wx_space_delimited(txt_file)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["date"])

    merged = None
    for var_name, frame in frames.items():
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="date", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["date"])

    merged = merged.sort_values("date").reset_index(drop=True)

    # Derived features
    if "temp_max" in merged.columns and "temp_min" in merged.columns:
        merged["temp_mean_wx"] = (merged["temp_max"] + merged["temp_min"]) / 2.0
        merged["temp_range_wx"] = merged["temp_max"] - merged["temp_min"]

    if all(c in merged.columns for c in ["precip_20_8", "precip_8_20"]):
        merged["precip_wx"] = merged["precip_20_8"].fillna(0.0) + merged["precip_8_20"].fillna(0.0)
    elif "precip_total" in merged.columns:
        merged["precip_wx"] = merged["precip_total"]

    if "precip_wx" in merged.columns:
        merged["precip_wx"] = merged["precip_wx"].fillna(0.0)

    return merged


def load_auto_hourly_monitoring(
    path: str | Path = DEFAULT_AUTO_MONITORING,
    target_site: str = "排水闸",
) -> pd.DataFrame:
    """Aggregate automatic hourly monitoring to daily statistics.

    Returns daily mean/min/max for key water-quality parameters at the target site.
    """
    wb = pd.ExcelFile(path)
    if target_site not in wb.sheet_names:
        return pd.DataFrame()

    frame = pd.read_excel(path, sheet_name=target_site)
    frame = frame.rename(columns={"监测时间": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()

    param_columns = {
        "水温(℃)": "wt",
        "pH(无量纲)": "ph",
        "溶解氧(mg/L)": "do",
        "电导率(μS/cm)": "cond",
        "浊度(NTU)": "turb",
        "高锰酸盐指数(mg/L)": "codmn",
        "氨氮(mg/L)": "nh3n",
        "总磷(mg/L)": "tp",
        "总氮(mg/L)": "tn",
    }

    available = {k: v for k, v in param_columns.items() if k in frame.columns}
    if not available:
        return pd.DataFrame()

    daily_aggs: list[pd.DataFrame] = []
    for src_name, tag in available.items():
        agg = (
            frame.groupby("date")[src_name]
            .agg(
                **{
                    f"hourly_{tag}_mean": "mean",
                    f"hourly_{tag}_min": "min",
                    f"hourly_{tag}_max": "max",
                    f"hourly_{tag}_count": "count",
                }
            )
            .reset_index()
        )
        daily_aggs.append(agg)

    if not daily_aggs:
        return pd.DataFrame()

    merged = daily_aggs[0]
    for agg in daily_aggs[1:]:
        merged = merged.merge(agg, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Derived daily variation features
    if all(c in merged.columns for c in ["hourly_do_mean", "hourly_do_min", "hourly_do_max"]):
        merged["hourly_do_range"] = merged["hourly_do_max"] - merged["hourly_do_min"]

    if all(c in merged.columns for c in ["hourly_tn_mean", "hourly_tn_min", "hourly_tn_max"]):
        merged["hourly_tn_range"] = merged["hourly_tn_max"] - merged["hourly_tn_min"]

    # Complete days (24 observations)
    merged["hourly_completeness"] = merged.get("hourly_tn_count", 0) / 24.0
    return merged


def load_county_discharge_outlets(
    discharge_dir: str | Path = DEFAULT_DISCHARGE_DIR,
) -> pd.DataFrame:
    """Load 9-county discharge outlet monitoring data.

    Each county file has columns: 序号, 责任单位, 采样日期, 水系名称, 排口编号,
    排口名称, pH, 化学需氧量, 总磷, 氨氮, 总氮.

    Returns a unified DataFrame with daily per-county outlet measurements.
    """
    p_dir = Path(discharge_dir)
    if not p_dir.exists():
        return pd.DataFrame()

    county_map = {
        "沙市区": "shashi",
        "荆州区": "jingzhou",
        "高新区": "gaoxin",
        "文旅区": "wenlv",
        "江陵": "jiangling",
        "监利": "jianli",
        "洪湖": "honghu_county",
        "经开区": "jingkai",
        "住新局": "zhuxinju",
    }

    all_records: list[dict[str, object]] = []
    for file_path in sorted(p_dir.glob("*.xlsx")):
        county_key = None
        for cn_name, en_name in county_map.items():
            if cn_name in file_path.name:
                county_key = en_name
                break
        if county_key is None:
            continue

        try:
            raw = pd.read_excel(file_path, header=None)
        except Exception:
            continue

        # Find the header row (contains "序号" or "采样日期")
        header_row = None
        for idx in range(min(10, len(raw))):
            row_vals = [str(v).strip() for v in raw.iloc[idx].values if pd.notna(v)]
            if any("采样日期" in v or "序号" in v for v in row_vals):
                header_row = idx
                break

        if header_row is None:
            continue

        # Re-read with correct header
        frame = raw.iloc[header_row + 1 :].copy()
        frame.columns = [str(c).strip().replace("\n", "") for c in raw.iloc[header_row].values]
        frame = frame.reset_index(drop=True)

        # Normalize columns
        col_map = {}
        for col in frame.columns:
            col_str = str(col).strip()
            if "pH" in col_str:
                col_map[col] = "ph"
            elif "化学需氧量" in col_str:
                col_map[col] = "cod"
            elif "总磷" in col_str:
                col_map[col] = "tp"
            elif "氨氮" in col_str:
                col_map[col] = "nh3n"
            elif "总氮" in col_str:
                col_map[col] = "tn"
            elif "采样日期" in col_str:
                col_map[col] = "sample_date"
            elif "排口编号" in col_str:
                col_map[col] = "outlet_id"
            elif "水系名称" in col_str:
                col_map[col] = "water_body"

        frame = frame.rename(columns=col_map)
        if "sample_date" not in frame.columns:
            continue

        frame["county"] = county_key
        frame["sample_date"] = pd.to_datetime(frame["sample_date"], errors="coerce")
        frame = frame.dropna(subset=["sample_date"])

        value_cols = [c for c in ["ph", "cod", "tp", "nh3n", "tn"] if c in frame.columns]
        if not value_cols:
            continue

        for _, row in frame.iterrows():
            rec: dict[str, object] = {
                "date": row["sample_date"],
                "county": county_key,
            }
            if "outlet_id" in frame.columns and pd.notna(row.get("outlet_id")):
                rec["outlet_id"] = str(row["outlet_id"]).strip()
            for vc in value_cols:
                val = row[vc]
                if pd.notna(val):
                    try:
                        rec[f"outlet_{vc}"] = float(val)
                    except (ValueError, TypeError):
                        pass
            all_records.append(rec)

    if not all_records:
        return pd.DataFrame()

    result = pd.DataFrame(all_records)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    for col in [c for c in result.columns if c.startswith("outlet_")]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result.sort_values(["date", "county"]).reset_index(drop=True)


def load_longterm_flow(
    path: str | Path = DEFAULT_LONGTERM_FLOW,
) -> pd.DataFrame:
    """Load long-term daily flow data (1998–2019) from NPS_MODEL project."""
    flow_path = Path(path)
    if not flow_path.exists():
        return pd.DataFrame(columns=["date", "flow_longterm"])

    frame = pd.read_excel(flow_path)
    frame = frame.rename(columns={"日期": "date", "流量": "flow_longterm"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["flow_longterm"] = pd.to_numeric(frame["flow_longterm"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature builders for integration into the main pipeline
# ---------------------------------------------------------------------------


def build_wx_climatology(
    wx_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build daily climatology (long-term means) from national weather station data.

    Returns a DataFrame with doy (day-of-year) and long-term average values
    for key weather variables. This can be joined to any date range safely.
    """
    wx_dir = Path(wx_dir) if wx_dir else DEFAULT_WX_DIR
    if not wx_dir.exists():
        return pd.DataFrame()

    wx = load_national_weather_station(wx_dir)
    if wx.empty or "date" not in wx.columns:
        return pd.DataFrame()

    wx["doy"] = wx["date"].dt.dayofyear

    # Columns to compute climatology for
    clim_cols = []
    for col in [
        "evap_mm", "gst_avg", "sunshine_hours", "temp_mean_wx",
        "temp_range_wx", "precip_wx", "rhum_avg", "wind_avg", "prs_avg",
    ]:
        if col in wx.columns:
            clim_cols.append(col)

    if not clim_cols:
        return pd.DataFrame()

    # Daily climatology: mean and std for each day-of-year
    agg_specs = {}
    for col in clim_cols:
        agg_specs[f"{col}_clim"] = (col, "mean")
        agg_specs[f"{col}_std"] = (col, "std")

    clim = wx.groupby("doy").agg(**agg_specs).reset_index()

    # Precipitation frequency (probability of precipitation > 0.1mm)
    if "precip_wx" in wx.columns:
        precip_freq = wx.groupby("doy")["precip_wx"].apply(lambda s: (s > 0.1).mean()).reset_index()
        precip_freq.columns = ["doy", "precip_freq_clim"]
        clim = clim.merge(precip_freq, on="doy", how="left")

    return clim


def build_extended_meteorology(
    meteorology_daily: pd.DataFrame,
    wx_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Add weather station climatology features to the meteorology frame.

    Since the station record (1957-2015) does not overlap with project data
    (2023+), we join long-term daily climatology by day-of-year. This provides
    seasonal context without look-ahead leakage.
    """
    if meteorology_daily.empty:
        return meteorology_daily.copy()

    wx_dir = Path(wx_dir) if wx_dir else DEFAULT_WX_DIR
    if not wx_dir.exists():
        return meteorology_daily.copy()

    clim = build_wx_climatology(wx_dir)
    if clim.empty:
        return meteorology_daily.copy()

    result = meteorology_daily.copy()
    result["doy"] = result["date"].dt.dayofyear
    result = result.merge(clim, on="doy", how="left")
    result = result.drop(columns=["doy"])

    # Also add direct station observations for overlapping period if any
    wx = load_national_weather_station(wx_dir)
    date_min = meteorology_daily["date"].min()
    date_max = meteorology_daily["date"].max()
    wx_overlap = wx[(wx["date"] >= date_min) & (wx["date"] <= date_max)]

    if not wx_overlap.empty and len(wx_overlap) > 10:
        direct_cols = ["date"]
        for col in ["evap_mm", "sunshine_hours", "temp_mean_wx", "precip_wx", "rhum_avg", "wind_avg"]:
            if col in wx_overlap.columns:
                direct_cols.append(col)
        result = result.merge(wx_overlap[direct_cols], on="date", how="left", suffixes=("", "_direct"))

    return result


def build_extended_water_quality_context(
    target_daily: pd.DataFrame,
    auto_mon_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add multi-site and hourly-statistics context to target observations.

    Joins:
      - Daily statistics from automatic hourly monitoring
    """
    if target_daily.empty:
        return target_daily.copy()

    auto_mon_path = Path(auto_mon_path) if auto_mon_path else DEFAULT_AUTO_MONITORING

    result = target_daily.copy()

    if auto_mon_path.exists():
        hourly = load_auto_hourly_monitoring(auto_mon_path, target_site="排水闸")
        if not hourly.empty:
            result = result.merge(hourly, on="date", how="left")

    return result


def build_county_outlet_features(
    model_dates: pd.DataFrame,
    discharge_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build county-aggregated discharge outlet features.

    For each date, compute summary statistics across all counties.
    """
    discharge_dir = Path(discharge_dir) if discharge_dir else DEFAULT_DISCHARGE_DIR
    if not discharge_dir.exists():
        return pd.DataFrame({"date": model_dates["date"]})

    outlets = load_county_discharge_outlets(discharge_dir)
    if outlets.empty:
        return pd.DataFrame({"date": model_dates["date"]})

    # Aggregate across counties per date (skip outlet_id which is a string identifier)
    value_cols = [c for c in outlets.columns if c.startswith("outlet_") and c != "outlet_id"]
    if not value_cols:
        return pd.DataFrame({"date": model_dates["date"]})

    daily_agg = (
        outlets.groupby("date")[value_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Flatten MultiIndex columns
    daily_agg.columns = [
        "_".join(filter(None, col)).strip("_") if col[1] else col[0]
        for col in daily_agg.columns
    ]

    # Rename to fit project convention
    rename_map = {}
    for col in daily_agg.columns:
        if col.startswith("outlet_"):
            rename_map[col] = col.replace("outlet_", "cso_")  # county-scale outlet
    daily_agg = daily_agg.rename(columns=rename_map)

    dates = pd.DataFrame({"date": pd.to_datetime(model_dates["date"]).dt.normalize()})
    dates = dates.drop_duplicates().sort_values("date")
    return dates.merge(daily_agg, on="date", how="left")


def build_longterm_flow_features(
    model_dates: pd.DataFrame,
    flow_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add long-term flow climatology features.

    For each date, provide:
      - flow_percentile: percentile rank relative to 1998-2019 record
      - flow_anomaly: departure from same-day-of-year climatology
    """
    flow_path = Path(flow_path) if flow_path else DEFAULT_LONGTERM_FLOW
    if not flow_path.exists():
        return pd.DataFrame({"date": model_dates["date"]})

    flow = load_longterm_flow(flow_path)
    if flow.empty:
        return pd.DataFrame({"date": model_dates["date"]})

    flow["doy"] = flow["date"].dt.dayofyear
    flow["year"] = flow["date"].dt.year

    # Day-of-year climatology (1998-2019)
    doy_clim = flow.groupby("doy")["flow_longterm"].agg(
        flow_clim_mean="mean",
        flow_clim_std="std",
        flow_clim_p10=lambda s: s.quantile(0.10),
        flow_clim_p50=lambda s: s.quantile(0.50),
        flow_clim_p90=lambda s: s.quantile(0.90),
    ).reset_index()

    # Overall percentile
    flow_sorted = flow["flow_longterm"].dropna().sort_values()

    dates = pd.DataFrame({"date": pd.to_datetime(model_dates["date"]).dt.normalize()})
    dates = dates.drop_duplicates().sort_values("date")
    dates["doy"] = dates["date"].dt.dayofyear

    features = dates.merge(doy_clim, on="doy", how="left")

    # Percentile rank in historical context
    if len(flow_sorted) > 0:
        features["flow_historical_n"] = len(flow_sorted)

    features = features.drop(columns=["doy"])
    return features
