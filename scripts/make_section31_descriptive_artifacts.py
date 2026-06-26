#!/usr/bin/env python3
"""Create Section 3.1 descriptive tables and publication-style figures.

The figures support one bounded claim: SWAT simulated total nitrogen output
contains watershed non-point-source process information, but it is not a direct
substitute for observed inlet load or outlet-gate concentration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"
MODEL_DATA = OUT / "processed" / "model_dataset_daily.csv"

COLORS = {
    "ink": "#1A1A1A",
    "gray": "#6B6B6B",
    "light_gray": "#F2F2F2",
    "axis": "#333333",
    "grid": "#D9D9D9",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.unicode_minus": False,
        }
    )


def _fmt_date(value: pd.Timestamp | str) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * 0.0
    return (values - values.mean()) / std


def _log_prior(frame: pd.DataFrame, col: str = "L_corr") -> pd.Series:
    return np.log10(frame[col].clip(lower=0.0) + 1.0)


def load_analysis_frame() -> pd.DataFrame:
    frame = pd.read_csv(MODEL_DATA, parse_dates=["date"])
    frame = frame[(frame["date"] >= "2023-01-01") & (frame["date"] <= "2023-12-26")].copy()
    frame = frame.sort_values("date").reset_index(drop=True)
    required = ["date", "L_corr", "L_key", "L_all", "target_tn_day", "precip", "api3", "storm_window_3d"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {MODEL_DATA}: {missing}")
    frame["month"] = frame["date"].dt.month
    frame["log_L_corr"] = _log_prior(frame, "L_corr")
    frame["z_log_L_corr"] = _zscore(frame["log_L_corr"])
    frame["z_total_nitrogen"] = _zscore(frame["target_tn_day"])
    return frame


def lag_diagnostics(frame: pd.DataFrame, prior_col: str = "L_corr", max_lag: int = 14) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lag in range(max_lag + 1):
        aligned = pd.DataFrame({"prior": frame[prior_col].shift(lag), "total_nitrogen": frame["target_tn_day"]}).dropna()
        if len(aligned) < 3:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = aligned["prior"].corr(aligned["total_nitrogen"], method="pearson")
            spearman = aligned["prior"].corr(aligned["total_nitrogen"], method="spearman")
        rows.append({"lag_days": lag, "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows)


def monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("month", as_index=False)
        .agg(
            n_days=("date", "count"),
            L_corr_mean=("L_corr", "mean"),
            L_corr_q90=("L_corr", lambda s: s.quantile(0.9)),
            target_tn_mean=("target_tn_day", "mean"),
            target_tn_q90=("target_tn_day", lambda s: s.quantile(0.9)),
            precip_sum=("precip", "sum"),
        )
        .sort_values("month")
    )


def monthly_lag_matrix(frame: pd.DataFrame, method: str = "pearson", max_lag: int = 14) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, monthly in frame.groupby("month"):
        for lag in range(max_lag + 1):
            aligned = pd.DataFrame(
                {
                    "prior": monthly["L_corr"].shift(lag),
                    "total_nitrogen": monthly["target_tn_day"],
                }
            ).dropna()
            corr = np.nan if len(aligned) < 5 else aligned["prior"].corr(aligned["total_nitrogen"], method=method)
            rows.append({"month": int(month), "lag_days": lag, "correlation": corr})
    return pd.DataFrame(rows).pivot(index="month", columns="lag_days", values="correlation")


def build_diagnostics_table(frame: pd.DataFrame) -> pd.DataFrame:
    lags = lag_diagnostics(frame)
    monthly = monthly_summary(frame)
    peak_lcorr = frame.loc[frame["L_corr"].idxmax()]
    peak_lkey = frame.loc[frame["L_key"].idxmax()]
    peak_total_n = frame.loc[frame["target_tn_day"].idxmax()]
    peak_precip = frame.loc[frame["precip"].idxmax()]
    max_abs_spearman = lags.loc[lags["spearman"].abs().idxmax()]
    max_pearson = lags.loc[lags["pearson"].idxmax()]
    lcorr_month = monthly.loc[monthly["L_corr_mean"].idxmax()]
    total_n_month = monthly.loc[monthly["target_tn_mean"].idxmax()]
    storm = frame[frame["storm_window_3d"].fillna(0).astype(int).eq(1)]
    nonstorm = frame[frame["storm_window_3d"].fillna(0).astype(int).eq(0)]

    rows = [
        {
            "诊断项": "分析窗口",
            "指标": "日期范围",
            "结果": f"{_fmt_date(frame['date'].min())} 至 {_fmt_date(frame['date'].max())}",
            "说明": "聚焦 2023 年建模窗口，避免使用 SWAT 外推补齐期作为主描述证据。",
        },
        {
            "诊断项": "SWAT 总氮输出峰值",
            "指标": "L_corr 最大日",
            "结果": f"{_fmt_date(peak_lcorr['date'])}; {_fmt_num(peak_lcorr['L_corr'], 1)}",
            "说明": "L_corr 为 SWAT 源区/河段总氮输出经训练期滞后正相关权重聚合后的过程变量，非实测通量。",
        },
        {
            "诊断项": "SWAT 关键区总氮输出峰值",
            "指标": "L_key 最大日",
            "结果": f"{_fmt_date(peak_lkey['date'])}; {_fmt_num(peak_lkey['L_key'], 1)}",
            "说明": "作为 L_corr 以外的候选口径，用于检查事件性高值是否受单一聚合方式控制。",
        },
        {
            "诊断项": "排水闸断面总氮峰值",
            "指标": "target_tn_day 最大日",
            "结果": f"{_fmt_date(peak_total_n['date'])}; {_fmt_num(peak_total_n['target_tn_day'], 3)} mg/L",
            "说明": "排水闸日均总氮为浓度响应，不是入湖负荷。",
        },
        {
            "诊断项": "最大降雨日",
            "指标": "precip 最大日",
            "结果": f"{_fmt_date(peak_precip['date'])}; {_fmt_num(peak_precip['precip'], 2)} mm/d",
            "说明": "用于辅助解释 SWAT 总氮输出的降雨径流驱动特征。",
        },
        {
            "诊断项": "月尺度高值期",
            "指标": "L_corr 月均值最高月份",
            "结果": f"{int(lcorr_month['month'])} 月; {_fmt_num(lcorr_month['L_corr_mean'], 1)}",
            "说明": "反映 SWAT 面源氮输出过程的季节性高值期。",
        },
        {
            "诊断项": "月尺度高值期",
            "指标": "排水闸总氮月均值最高月份",
            "结果": f"{int(total_n_month['month'])} 月; {_fmt_num(total_n_month['target_tn_mean'], 3)} mg/L",
            "说明": "反映断面浓度响应的季节性高值期。",
        },
        {
            "诊断项": "0-14 d 滞后相关",
            "指标": "最大绝对 Spearman",
            "结果": f"lag={int(max_abs_spearman['lag_days'])} d; rho={_fmt_num(max_abs_spearman['spearman'], 3)}",
            "说明": "直接单变量秩相关为弱到中等且方向不稳定，不能作为同步负荷-浓度证据。",
        },
        {
            "诊断项": "0-14 d 滞后相关",
            "指标": "最大 Pearson",
            "结果": f"lag={int(max_pearson['lag_days'])} d; r={_fmt_num(max_pearson['pearson'], 3)}",
            "说明": "线性相关最高值仍较弱，支持后续响应校准而非简单同步解释。",
        },
        {
            "诊断项": "降雨窗口波动",
            "指标": "L_corr 均值/标准差",
            "结果": f"窗口内 {_fmt_num(storm['L_corr'].mean(), 1)}/{_fmt_num(storm['L_corr'].std(ddof=0), 1)}; 窗口外 {_fmt_num(nonstorm['L_corr'].mean(), 1)}/{_fmt_num(nonstorm['L_corr'].std(ddof=0), 1)}",
            "说明": "SWAT 总氮输出在降雨影响窗口内表现出更强事件性波动。",
        },
        {
            "诊断项": "降雨窗口波动",
            "指标": "排水闸总氮均值/标准差",
            "结果": f"窗口内 {_fmt_num(storm['target_tn_day'].mean(), 3)}/{_fmt_num(storm['target_tn_day'].std(ddof=0), 3)}; 窗口外 {_fmt_num(nonstorm['target_tn_day'].mean(), 3)}/{_fmt_num(nonstorm['target_tn_day'].std(ddof=0), 3)}",
            "说明": "断面总氮也有波动差异，但幅度和峰值时间不与 SWAT 总氮输出一一同步。",
        },
    ]
    return pd.DataFrame(rows)


def _finish_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.7, colors=COLORS["axis"])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color(COLORS["axis"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COLORS["grid"], lw=0.45, alpha=0.65)


def _save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    svg = FIG_DIR / f"{stem}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf, svg]


def make_compact_diagnostic_figure(frame: pd.DataFrame) -> list[Path]:
    """Create the compact main-text diagnostic figure for Section 3.1."""
    fig = plt.figure(figsize=(7.2, 2.65))
    spec = fig.add_gridspec(
        1,
        2,
        width_ratios=[2.65, 1.0],
        wspace=0.22,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.22)
    ax_annual = fig.add_subplot(spec[0, 0])
    ax_lag = fig.add_subplot(spec[0, 1])

    swat_peak = frame.loc[frame["L_corr"].idxmax()]
    total_n_peak = frame.loc[frame["target_tn_day"].idxmax()]
    lags = lag_diagnostics(frame)
    best = lags.loc[lags["pearson"].idxmax()]

    rain_ax = ax_annual.twinx()
    rain_ax.bar(frame["date"], frame["precip"], width=1.05, color=COLORS["sky"], alpha=0.42, edgecolor="none", zorder=0)
    rain_ax.set_ylim(0, max(frame["precip"].max() * 1.55, 1.0))
    rain_ax.set_yticks([])
    for spine in rain_ax.spines.values():
        spine.set_visible(False)

    ax_annual.axhline(0, color=COLORS["grid"], lw=0.65, zorder=1)
    ax_annual.plot(frame["date"], frame["z_log_L_corr"], color=COLORS["green"], lw=1.05, label="SWAT-based TN export response index", zorder=3)
    ax_annual.plot(frame["date"], frame["z_total_nitrogen"], color=COLORS["blue"], lw=1.05, label="Outlet-gate TN, z-score", zorder=3)
    ax_annual.axvline(swat_peak["date"], color=COLORS["green"], lw=0.75, ls="--", alpha=0.70)
    ax_annual.axvline(total_n_peak["date"], color=COLORS["blue"], lw=0.75, ls="--", alpha=0.70)
    ax_annual.scatter([swat_peak["date"]], [swat_peak["z_log_L_corr"]], color=COLORS["green"], s=18, zorder=5)
    ax_annual.scatter([total_n_peak["date"]], [total_n_peak["z_total_nitrogen"]], color=COLORS["blue"], s=18, zorder=5)
    ax_annual.annotate(
        "TN export index peak\n30 Jun",
        xy=(swat_peak["date"], swat_peak["z_log_L_corr"]),
        xytext=(pd.Timestamp("2023-07-17"), 2.15),
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": COLORS["green"]},
        color=COLORS["green"],
        fontsize=6.5,
    )
    ax_annual.annotate(
        "TN peak\n19 Dec",
        xy=(total_n_peak["date"], total_n_peak["z_total_nitrogen"]),
        xytext=(pd.Timestamp("2023-10-30"), 2.15),
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": COLORS["blue"]},
        color=COLORS["blue"],
        fontsize=6.5,
    )
    ax_annual.set_ylim(-2.55, 2.75)
    ax_annual.set_ylabel("Standardized value")
    ax_annual.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_annual.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_annual.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["sky"], alpha=0.20, edgecolor="none", label="Daily precipitation"),
            Line2D([0], [0], color=COLORS["green"], lw=1.05, label="SWAT-based TN export response index"),
            Line2D([0], [0], color=COLORS["blue"], lw=1.05, label="Outlet-gate TN"),
        ],
        frameon=False,
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0.0, 1.18),
        handlelength=1.35,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    _finish_axes(ax_annual, "y")

    ax_lag.axhline(0.0, color=COLORS["grid"], lw=0.65)
    ax_lag.plot(lags["lag_days"], lags["pearson"], marker="o", ms=2.6, lw=0.95, color=COLORS["vermillion"], label="Pearson r")
    ax_lag.plot(lags["lag_days"], lags["spearman"], marker="s", ms=2.4, lw=0.95, color=COLORS["purple"], label="Spearman rho")
    ax_lag.scatter([best["lag_days"]], [best["pearson"]], color=COLORS["vermillion"], s=18, zorder=3)
    ax_lag.annotate(
        f"max r={best['pearson']:.2f}\nlag={int(best['lag_days'])} d",
        xy=(best["lag_days"], best["pearson"]),
        xytext=(3.7, 0.255),
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": COLORS["vermillion"]},
        color=COLORS["vermillion"],
        fontsize=6.4,
    )
    ax_lag.set_xlim(-0.4, 14.4)
    ax_lag.set_ylim(-0.27, 0.30)
    ax_lag.set_xlabel("Lag (days)")
    ax_lag.set_ylabel("Correlation with TN")
    ax_lag.legend(frameon=False, loc="lower right", handlelength=1.15, fontsize=6.2)
    _finish_axes(ax_lag, "y")

    for label, ax in zip(("a", "b"), (ax_annual, ax_lag), strict=True):
        ax.text(-0.075, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, weight="bold")

    return _save_figure(fig, "fig31_compact_swat_tn_diagnostic")


def make_process_relationship_figure(frame: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 7.7))
    spec = fig.add_gridspec(4, 3, height_ratios=[0.70, 0.70, 0.78, 1.22], hspace=0.48, wspace=0.45)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.075)
    ax_rain = fig.add_subplot(spec[0, :])
    ax_prior = fig.add_subplot(spec[1, :], sharex=ax_rain)
    ax_total_n = fig.add_subplot(spec[2, :], sharex=ax_rain)
    ax_month = fig.add_subplot(spec[3, 0])
    ax_lag = fig.add_subplot(spec[3, 1])
    ax_box = fig.add_subplot(spec[3, 2])

    ax_rain.bar(frame["date"], frame["precip"], width=1.0, color=COLORS["sky"], alpha=0.42, edgecolor="none")
    ax_rain.plot(frame["date"], frame["api3"], color=COLORS["blue"], lw=1.05)
    ax_rain.set_ylabel("Rain/API3\n(mm)")
    ax_rain.set_ylim(bottom=0)

    ax_prior.plot(frame["date"], frame["log_L_corr"], color=COLORS["green"], lw=1.18)
    ax_prior.set_ylabel("log10\n(Lcorr+1)")

    ax_total_n.plot(frame["date"], frame["target_tn_day"], color=COLORS["blue"], lw=1.12)
    ax_total_n.set_ylabel("Total nitrogen\n(mg/L)")

    swat_peak = frame.loc[frame["L_corr"].idxmax()]
    total_n_peak = frame.loc[frame["target_tn_day"].idxmax()]
    ax_prior.scatter([swat_peak["date"]], [swat_peak["log_L_corr"]], color=COLORS["vermillion"], s=23, zorder=4)
    ax_total_n.scatter([total_n_peak["date"]], [total_n_peak["target_tn_day"]], color=COLORS["orange"], s=23, zorder=4)
    ax_prior.annotate(
        f"SWAT peak\n{_fmt_date(swat_peak['date'])}",
        xy=(swat_peak["date"], swat_peak["log_L_corr"]),
        xytext=(swat_peak["date"] + pd.Timedelta(days=16), swat_peak["log_L_corr"] - 0.52),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS["vermillion"]},
        color=COLORS["vermillion"],
        fontsize=7,
    )
    ax_total_n.annotate(
        f"Total nitrogen peak\n{_fmt_date(total_n_peak['date'])}",
        xy=(total_n_peak["date"], total_n_peak["target_tn_day"]),
        xytext=(total_n_peak["date"] - pd.Timedelta(days=70), total_n_peak["target_tn_day"] - 0.23),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS["orange"]},
        color=COLORS["orange"],
        fontsize=7,
    )

    for ax in (ax_rain, ax_prior, ax_total_n):
        ax.axvline(swat_peak["date"], color=COLORS["vermillion"], lw=0.7, alpha=0.45, ls="--")
        ax.axvline(total_n_peak["date"], color=COLORS["orange"], lw=0.7, alpha=0.45, ls="--")
        ax.axvspan(pd.Timestamp("2023-11-07"), pd.Timestamp("2023-12-26"), color=COLORS["light_gray"], alpha=0.60, zorder=0)
        _finish_axes(ax, None)
    ax_rain.text(pd.Timestamp("2023-11-10"), ax_rain.get_ylim()[1] * 0.78, "hold-out period", color=COLORS["gray"], fontsize=7)
    ax_total_n.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_total_n.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_total_n.set_xlabel("2023")
    for ax in (ax_rain, ax_prior):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_rain.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["sky"], alpha=0.42, edgecolor="none", label="Daily precipitation"),
            Line2D([0], [0], color=COLORS["blue"], lw=1.05, label="API3"),
            Line2D([0], [0], color=COLORS["green"], lw=1.18, label="SWAT total nitrogen output"),
            Line2D([0], [0], color=COLORS["blue"], lw=1.12, label="Outlet-gate total nitrogen"),
        ],
        frameon=False,
        loc="upper left",
        ncol=4,
        handlelength=1.5,
        bbox_to_anchor=(0.0, 1.31),
        borderaxespad=0.0,
    )

    monthly = monthly_summary(frame)
    months = monthly["month"].astype(int)
    prior_norm = monthly["L_corr_mean"] / monthly["L_corr_mean"].max()
    total_n_norm = monthly["target_tn_mean"] / monthly["target_tn_mean"].max()
    ax_month.plot(months, prior_norm, marker="o", ms=3.2, lw=1.05, color=COLORS["green"], label="SWAT output")
    ax_month.plot(months, total_n_norm, marker="s", ms=3.0, lw=1.05, color=COLORS["blue"], label="Total nitrogen")
    ax_month.set_xlim(0.6, 12.9)
    ax_month.set_ylim(0.0, 1.15)
    ax_month.set_xticks(range(1, 13, 2))
    ax_month.set_xlabel("Month")
    ax_month.set_ylabel("Normalized\nmonthly mean")
    ax_month.legend(frameon=False, loc="upper left", handlelength=1.4)
    _finish_axes(ax_month, "y")

    lags = lag_diagnostics(frame)
    ax_lag.axhline(0.0, color=COLORS["grid"], lw=0.7)
    ax_lag.plot(lags["lag_days"], lags["pearson"], marker="o", ms=3.0, lw=1.0, color=COLORS["vermillion"], label="Pearson r")
    ax_lag.plot(lags["lag_days"], lags["spearman"], marker="s", ms=2.8, lw=1.0, color=COLORS["purple"], label="Spearman rho")
    best = lags.loc[lags["pearson"].idxmax()]
    ax_lag.scatter([best["lag_days"]], [best["pearson"]], color=COLORS["vermillion"], s=21, zorder=3)
    ax_lag.annotate(
        f"max r={best['pearson']:.2f}\nlag={int(best['lag_days'])} d",
        xy=(best["lag_days"], best["pearson"]),
        xytext=(7.1, 0.22),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS["vermillion"]},
        color=COLORS["vermillion"],
        fontsize=7,
    )
    ax_lag.set_xlim(-0.4, 14.4)
    ax_lag.set_ylim(-0.27, 0.29)
    ax_lag.set_xlabel("Lag of SWAT output (days)")
    ax_lag.set_ylabel("Correlation\nwith total nitrogen")
    ax_lag.legend(frameon=False, loc="lower right", handlelength=1.4)
    _finish_axes(ax_lag, "y")

    storm_mask = frame["storm_window_3d"].fillna(0).astype(int).eq(1)
    box_data = [
        frame.loc[storm_mask, "z_log_L_corr"].dropna(),
        frame.loc[~storm_mask, "z_log_L_corr"].dropna(),
        frame.loc[storm_mask, "z_total_nitrogen"].dropna(),
        frame.loc[~storm_mask, "z_total_nitrogen"].dropna(),
    ]
    bp = ax_box.boxplot(
        box_data,
        positions=[0.86, 1.16, 1.86, 2.16],
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "lw": 0.8},
        boxprops={"lw": 0.7},
        whiskerprops={"lw": 0.7},
        capprops={"lw": 0.7},
    )
    for patch, color in zip(bp["boxes"], [COLORS["green"], COLORS["gray"], COLORS["blue"], COLORS["gray"]], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.52)
    ax_box.axhline(0.0, color=COLORS["grid"], lw=0.7)
    ax_box.set_xlim(0.58, 2.43)
    ax_box.set_ylim(-2.35, 2.75)
    ax_box.set_xticks([0.86, 1.16, 1.86, 2.16])
    ax_box.set_xticklabels(["Rain", "Other", "Rain", "Other"])
    ax_box.text(1.01, -0.23, "SWAT output", transform=ax_box.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax_box.text(2.01, -0.23, "Total nitrogen", transform=ax_box.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax_box.set_ylabel("Standardized\nvalue")
    _finish_axes(ax_box, "y")

    ax_rain.text(-0.065, 1.28, "a", transform=ax_rain.transAxes, ha="left", va="bottom", fontsize=10, weight="bold")
    for label, ax in zip(("b", "c", "d"), (ax_month, ax_lag, ax_box), strict=True):
        ax.text(-0.20, 1.09, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, weight="bold")
    return _save_figure(fig, "fig31_swat_tn_process_relationship")


def _select_event_dates(frame: pd.DataFrame, min_gap_days: int = 35, n_events: int = 3) -> list[pd.Timestamp]:
    candidates = frame.sort_values("L_corr", ascending=False)["date"].tolist()
    selected: list[pd.Timestamp] = []
    for date in candidates:
        ts = pd.Timestamp(date)
        if all(abs((ts - other).days) >= min_gap_days for other in selected):
            selected.append(ts)
        if len(selected) == n_events:
            break
    return sorted(selected)


def make_event_cases_figure(frame: pd.DataFrame) -> list[Path]:
    event_dates = _select_event_dates(frame)
    fig, axes = plt.subplots(3, len(event_dates), figsize=(7.2, 5.7), sharex=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.105, wspace=0.25, hspace=0.18)

    row_labels = ["Rain/API3\n(mm)", "SWAT output\nz-score", "Total nitrogen\nz-score"]
    for col, event_date in enumerate(event_dates):
        window = frame[(frame["date"] >= event_date - pd.Timedelta(days=10)) & (frame["date"] <= event_date + pd.Timedelta(days=21))].copy()
        rel_days = (window["date"] - event_date).dt.days

        axes[0, col].bar(rel_days, window["precip"], width=0.85, color=COLORS["sky"], alpha=0.42, edgecolor="none")
        axes[0, col].plot(rel_days, window["api3"], color=COLORS["blue"], lw=0.95)
        axes[1, col].plot(rel_days, _zscore(window["log_L_corr"]), color=COLORS["green"], lw=1.10)
        axes[2, col].plot(rel_days, _zscore(window["target_tn_day"]), color=COLORS["blue"], lw=1.10)

        for row in range(3):
            ax = axes[row, col]
            ax.axvline(0, color=COLORS["vermillion"], ls="--", lw=0.75, alpha=0.75)
            ax.set_xlim(-10, 21)
            ax.set_xticks([-10, 0, 10, 20])
            _finish_axes(ax, "y")
            if col == 0:
                ax.set_ylabel(row_labels[row])
            else:
                ax.set_yticklabels([])
            if row < 2:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Days from SWAT peak")

        local_total_n_peak = window.loc[window["target_tn_day"].idxmax()]
        peak_lag = int((local_total_n_peak["date"] - event_date).days)
        axes[0, col].set_title(f"SWAT event {col + 1}\n{_fmt_date(event_date)}", pad=5)
        axes[2, col].annotate(
            f"local total nitrogen peak\n{peak_lag:+d} d",
            xy=(peak_lag, _zscore(window["target_tn_day"]).loc[local_total_n_peak.name]),
            xytext=(max(-8, peak_lag - 8), 1.35),
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": COLORS["orange"]},
            color=COLORS["orange"],
            fontsize=7,
        )

    axes[0, 0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["sky"], alpha=0.42, edgecolor="none", label="Daily precipitation"),
            Line2D([0], [0], color=COLORS["blue"], lw=0.95, label="API3"),
            Line2D([0], [0], color=COLORS["green"], lw=1.10, label="SWAT total nitrogen output"),
            Line2D([0], [0], color=COLORS["blue"], lw=1.10, label="Outlet-gate total nitrogen"),
        ],
        frameon=False,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.0, 1.62),
        handlelength=1.4,
        borderaxespad=0.0,
    )
    for label, ax in zip(("a", "b", "c"), axes[:, 0], strict=True):
        ax.text(-0.26, 1.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, weight="bold")
    return _save_figure(fig, "fig32_rainfall_swat_tn_event_cases")


def make_month_lag_summary_figure(frame: pd.DataFrame) -> list[Path]:
    pearson = monthly_lag_matrix(frame, "pearson")
    spearman = monthly_lag_matrix(frame, "spearman")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.925, top=0.90, bottom=0.18, wspace=0.10)

    for ax, matrix, title in zip(axes, (pearson, spearman), ("Pearson r", "Spearman rho"), strict=True):
        image = ax.imshow(matrix.values, aspect="auto", cmap="RdBu_r", vmin=-0.55, vmax=0.55, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Lag of SWAT output (days)")
        ax.set_xticks([0, 3, 7, 10, 14])
        ax.set_xticklabels([0, 3, 7, 10, 14])
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index.astype(int))
        _finish_axes(ax, None)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Month")
    cbar_ax = fig.add_axes([0.94, 0.23, 0.016, 0.58])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Correlation")
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6)
    axes[0].text(-0.17, 1.08, "a", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=10, weight="bold")
    axes[1].text(-0.11, 1.08, "b", transform=axes[1].transAxes, ha="left", va="bottom", fontsize=10, weight="bold")
    return _save_figure(fig, "fig33_month_lag_summary")


def main() -> None:
    setup_style()
    frame = load_analysis_frame()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLE_DIR / "tableS6_swat_tn_descriptive_diagnostics.csv"
    build_diagnostics_table(frame).to_csv(table_path, index=False)

    figure_paths = []
    figure_paths.extend(make_compact_diagnostic_figure(frame))

    print(f"Wrote {table_path}")
    for path in figure_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
