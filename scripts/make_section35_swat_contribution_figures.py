#!/usr/bin/env python3
"""Create Section 3.5 figures for SWAT NPS-TN contribution analysis.

The figures defend a bounded claim: SWAT simulated TN outputs are not direct
load truth, but their event timing becomes useful for outlet-section TN
prediction after response calibration and gate-state conditioning.
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

COLORS = {
    "ink": "#202124",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "grid": "#D6D6D6",
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
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.2,
            "figure.dpi": 220,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        FIG_DIR / f"{stem}.png",
        FIG_DIR / f"{stem}.svg",
        FIG_DIR / f"{stem}.pdf",
        FIG_DIR / f"{stem}.tiff",
    ]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * 0.0
    return (values - values.mean()) / std


def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.tick_params(direction="out", length=2.7, width=0.65, colors=COLORS["ink"])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["ink"])
        ax.spines[spine].set_linewidth(0.7)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], lw=0.45, alpha=0.55)
        ax.set_axisbelow(True)


def shade_storm_windows(ax: plt.Axes, frame: pd.DataFrame, alpha: float = 0.21) -> None:
    storm = frame["storm_window_3d"].fillna(0).astype(int).eq(1)
    groups = (storm != storm.shift()).cumsum()
    for _, group in frame.assign(storm_flag=storm).groupby(groups):
        if bool(group["storm_flag"].iloc[0]):
            start = group["date"].min() - pd.Timedelta(hours=12)
            end = group["date"].max() + pd.Timedelta(hours=12)
            ax.axvspan(start, end, color=COLORS["sky"], alpha=alpha, lw=0, zorder=0)


def load_model_frame() -> pd.DataFrame:
    frame = pd.read_csv(OUT / "processed" / "model_dataset_daily.csv", parse_dates=["date"])
    frame = frame[(frame["date"] >= "2023-01-01") & (frame["date"] <= "2023-12-26")].copy()
    frame = frame.sort_values("date").reset_index(drop=True)
    required = ["date", "L_corr", "target_tn_day", "precip", "storm_window_3d"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in model_dataset_daily.csv: {missing}")
    frame["log_L_corr"] = np.log10(frame["L_corr"].clip(lower=0.0) + 1.0)
    frame["z_swat_output"] = zscore(frame["log_L_corr"])
    frame["z_outlet_tn"] = zscore(frame["target_tn_day"])
    return frame


def load_input_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(TABLE_DIR / "table3_swat_prior_evidence_chain.csv")
    rename = {
        "无 SWAT": "No SWAT",
        "未校准 SWAT 特征": "Raw SWAT",
        "直接加入 SWAT 输出": "Raw SWAT",
        "响应校准 SWAT": "Resp.-cal. SWAT",
        "断面响应校准": "Resp.-cal. SWAT",
    }
    metrics["label"] = metrics["证据层级"].map(rename).fillna(metrics["证据层级"])
    order = ["No SWAT", "Raw SWAT", "Resp.-cal. SWAT"]
    metrics["order"] = metrics["label"].map({label: idx for idx, label in enumerate(order)})
    return metrics.sort_values("order").reset_index(drop=True)


def load_stratified_metrics() -> pd.DataFrame:
    strat = pd.read_csv(TABLE_DIR / "table32_gate_connectivity_response_summary.csv")
    order = [
        "Overall",
        "Rainfall window",
        "Non-rainfall",
        "S1 low exchange",
        "S2 normal exchange",
        "S3 high exchange",
    ]
    strat = strat[strat["scenario"].isin(order)].copy()
    strat["order"] = strat["scenario"].map({label: idx for idx, label in enumerate(order)})
    return strat.sort_values("order").reset_index(drop=True)


def make_main_figure() -> list[Path]:
    frame = load_model_frame()
    input_metrics = load_input_metrics()
    strat = load_stratified_metrics()

    fig = plt.figure(figsize=(7.2, 4.8))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.12, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.34,
        hspace=0.45,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    # Panel A: event-like SWAT output and asynchronous outlet TN response.
    shade_storm_windows(ax_a, frame, alpha=0.11)
    ax_a.plot(frame["date"], frame["z_swat_output"], color=COLORS["green"], lw=1.15, label="SWAT output")
    ax_a.plot(frame["date"], frame["z_outlet_tn"], color=COLORS["blue"], lw=1.15, label="Outlet-gate TN")
    ax_a.axhline(0, color=COLORS["gray"], lw=0.55, alpha=0.65)
    ax_a.set_title("a  SWAT output is event-like but not synchronous with outlet TN", loc="left", fontweight="bold")
    ax_a.set_ylabel("Standardized value")
    ax_a.set_xlabel("Date in 2023")
    ax_a.set_ylim(-1.8, 5.2)
    ax_a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_a.legend(
        handles=[
            Line2D([0], [0], color=COLORS["green"], lw=1.4, label="SWAT output"),
            Line2D([0], [0], color=COLORS["blue"], lw=1.4, label="Outlet TN"),
            Line2D([0], [0], color=COLORS["sky"], lw=5.0, alpha=0.35, label="Rainfall window"),
        ],
        frameon=False,
        loc="upper left",
        ncol=1,
    )
    style_axis(ax_a)

    # Panel B: direct SWAT input is not useful; response calibration is.
    bar_colors = [COLORS["gray"], COLORS["orange"], COLORS["vermillion"]]
    x = np.arange(len(input_metrics))
    ax_b.bar(x, input_metrics["RMSE (mg/L)"], color=bar_colors, width=0.68)
    ax_b.set_title("b  Input condition test", loc="left", fontweight="bold")
    ax_b.set_ylabel("RMSE (mg/L)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(input_metrics["label"], rotation=18, ha="right")
    ax_b.set_ylim(0, float(input_metrics["RMSE (mg/L)"].max()) * 1.28)
    for idx, row in input_metrics.iterrows():
        ax_b.text(
            idx,
            row["RMSE (mg/L)"] + 0.006,
            f"{row['RMSE (mg/L)']:.3f}\nNSE={row['NSE']:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    style_axis(ax_b)

    # Panel C: contribution concentrates in rainfall and low/normal exchange windows.
    y = np.arange(len(strat))
    ax_c.barh(y, strat["rmse_reduction_pct"], color=COLORS["vermillion"], height=0.62)
    ax_c.set_title("c  Application windows", loc="left", fontweight="bold")
    ax_c.set_xlabel("RMSE reduction (%)")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([f"{r.scenario} (n={int(r.n)})" for r in strat.itertuples()])
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, max(45, float(strat["rmse_reduction_pct"].max()) * 1.18))
    for idx, row in strat.iterrows():
        ax_c.text(
            row["rmse_reduction_pct"] + 0.8,
            idx,
            f"{row['rmse_reduction_pct']:.1f}%",
            va="center",
            fontsize=6.4,
            color=COLORS["ink"],
        )
    style_axis(ax_c, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        "SWAT outputs are treated as relative process information, not measured inflow loads.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=COLORS["gray"],
    )
    return save_figure(fig, "fig35_swat_tn_contribution_pathway")


def make_supplementary_error_figure() -> list[Path]:
    pred = pd.read_csv(OUT / "processed" / "nps_prior_value_predictions.csv", parse_dates=["date"])
    test = pred[pred["split"].eq("test")].copy()
    wide = test.pivot(index="date", columns="model_name", values="pred_tn").sort_index()
    meta = test.drop_duplicates("date").set_index("date").loc[wide.index]
    required = ["no_swat", "with_swat"]
    missing = [col for col in required if col not in wide.columns]
    if missing:
        raise ValueError(f"Missing prediction columns in nps_prior_value_predictions.csv: {missing}")

    improvement = (
        (wide["no_swat"] - meta["target_tn_day"]).abs()
        - (wide["with_swat"] - meta["target_tn_day"]).abs()
    )
    plot = pd.DataFrame(
        {
            "date": wide.index,
            "improvement": improvement.to_numpy(dtype=float),
            "storm_window_3d": meta["storm_window_3d"].to_numpy(),
            "state_label": meta["state_label"].to_numpy(),
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    shade_storm_windows(ax, plot, alpha=0.14)
    colors = np.where(plot["improvement"].ge(0), COLORS["vermillion"], COLORS["gray"])
    ax.bar(plot["date"], plot["improvement"], width=0.82, color=colors, edgecolor="none")
    ax.axhline(0, color=COLORS["ink"], lw=0.75)
    ax.set_title(
        "Daily absolute-error change after response-calibrated SWAT input",
        loc="left",
        fontweight="bold",
    )
    ax.set_ylabel("|e| No SWAT - |e| calibrated SWAT\n(mg/L)")
    ax.set_xlabel("Test date")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    ax.text(
        0.995,
        0.94,
        f"Better days: {(improvement > 0).sum()}/{len(improvement)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color=COLORS["ink"],
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["vermillion"], lw=5, label="Calibrated SWAT lower error"),
            Line2D([0], [0], color=COLORS["gray"], lw=5, label="No SWAT lower error"),
            Line2D([0], [0], color=COLORS["sky"], lw=5, alpha=0.35, label="Rainfall window"),
        ],
        frameon=False,
        loc="lower left",
        ncol=3,
        bbox_to_anchor=(0.0, -0.55),
    )
    style_axis(ax)
    fig.tight_layout()
    return save_figure(fig, "figS_section35_daily_error_improvement")


def make_contribution_diagnostic_figure() -> list[Path]:
    pred = pd.read_csv(OUT / "processed" / "nps_prior_value_predictions.csv", parse_dates=["date"])
    test = pred[pred["split"].eq("test")].copy()
    wide = test.pivot(index="date", columns="model_name", values="pred_tn").sort_index()
    meta = test.drop_duplicates("date").set_index("date").loc[wide.index]
    required = ["no_swat", "with_swat"]
    missing = [col for col in required if col not in wide.columns]
    if missing:
        raise ValueError(f"Missing prediction columns in nps_prior_value_predictions.csv: {missing}")

    observed = meta["target_tn_day"]
    no_swat_error = wide["no_swat"] - observed
    calibrated_error = wide["with_swat"] - observed
    improvement = no_swat_error.abs() - calibrated_error.abs()
    needed_correction = observed - wide["no_swat"]
    swat_adjustment = wide["with_swat"] - wide["no_swat"]
    sign_agree = np.sign(needed_correction).eq(np.sign(swat_adjustment))

    plot = pd.DataFrame(
        {
            "date": wide.index,
            "improvement": improvement.to_numpy(dtype=float),
            "needed_correction": needed_correction.to_numpy(dtype=float),
            "swat_adjustment": swat_adjustment.to_numpy(dtype=float),
            "abs_adjustment": swat_adjustment.abs().to_numpy(dtype=float),
            "storm_window_3d": meta["storm_window_3d"].to_numpy(),
            "state_label": meta["state_label"].to_numpy(),
            "sign_agree": sign_agree.to_numpy(),
        }
    )
    plot["cum_improvement"] = plot["improvement"].cumsum()
    total_no_swat_error = float(no_swat_error.abs().sum())
    total_calibrated_error = float(calibrated_error.abs().sum())
    total_gain = float(plot["improvement"].sum())
    total_gain_pct = 100.0 * total_gain / total_no_swat_error

    fig, ax_a = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)

    # Panel A: one compact time-axis view carries both paired daily error change and cumulative gain.
    shade_storm_windows(ax_a, plot, alpha=0.21)
    bar_colors = np.where(plot["improvement"].ge(0), COLORS["vermillion"], "#8A8F98")
    ax_a.bar(
        plot["date"],
        plot["improvement"],
        width=0.82,
        color=bar_colors,
        edgecolor="none",
        alpha=0.88,
        zorder=2,
    )
    ax_a.axhline(0, color=COLORS["ink"], lw=0.65)
    left_min = min(float(plot["improvement"].min()) * 1.18, -0.02)
    left_max = max(float(plot["improvement"].max()) * 1.12, 0.02)
    ax_a.set_ylim(left_min, left_max)
    ax_a.set_ylabel("Daily |e| reduction (mg/L)")
    ax_a.set_xlabel("Date")
    ax_a.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_a.tick_params(axis="x", rotation=35)
    ax_a2 = ax_a.twinx()
    ax_a2.plot(plot["date"], plot["cum_improvement"], color=COLORS["blue"], lw=1.55, zorder=4)
    cum_max = max(float(plot["cum_improvement"].max()) * 1.08, 0.05)
    zero_fraction = (0.0 - left_min) / (left_max - left_min)
    right_min = -zero_fraction * cum_max / max(1.0 - zero_fraction, 1e-6)
    ax_a2.set_ylim(right_min, cum_max)
    ax_a2.set_ylabel("Cumulative reduction (mg/L)", color=COLORS["blue"])
    ax_a2.tick_params(direction="out", length=2.7, width=0.65, colors=COLORS["blue"])
    ax_a2.spines["right"].set_visible(True)
    ax_a2.spines["right"].set_color(COLORS["blue"])
    ax_a2.spines["right"].set_linewidth(0.7)
    ax_a2.spines["top"].set_visible(False)
    ax_a.legend(
        handles=[
            Line2D([0], [0], color=COLORS["vermillion"], lw=5, label="Daily gain"),
            Line2D([0], [0], color="#8A8F98", lw=5, label="Daily loss"),
            Line2D([0], [0], color=COLORS["blue"], lw=1.6, label="Cumulative net reduction"),
            Line2D([0], [0], color=COLORS["sky"], lw=5, alpha=0.35, label="Rainfall-affected window"),
        ],
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.8,
    )
    style_axis(ax_a)

    return save_figure(fig, "fig35_swat_tn_daily_contribution_diagnostic")


def main() -> None:
    setup_style()
    outputs = {
        "main": make_contribution_diagnostic_figure(),
    }
    for group, paths in outputs.items():
        print(group)
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
