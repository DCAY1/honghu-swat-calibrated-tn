#!/usr/bin/env python3
"""Generate Section 3.2 gate-connectivity response figures.

The figures use existing model outputs only. Figure text is English so the
Chinese manuscript can provide journal-style captions and interpretation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG_DIR = OUT / "figures"
TABLE_DIR = OUT / "tables"


COLORS = {
    "ink": "#222222",
    "gray": "#666666",
    "light": "#F3F4F6",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "teal": "#009E73",
    "rain_window": "#CFE8F6",
}

STATE_COLORS = {
    "S1": "#0072B2",
    "S2": "#009E73",
    "S3": "#E69F00",
    "S4": "#CC79A7",
}

STATE_LABELS = {
    "S1": "S1 low exchange",
    "S2": "S2 normal exchange",
    "S3": "S3 high exchange",
    "S4": "S4 backflow",
}

MODEL_LABELS = {
    "no_swat": "No SWAT",
    "raw_swat": "Direct SWAT output",
    "with_swat": "Response-calibrated SWAT",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.45,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        FIG_DIR / f"{stem}.png",
        FIG_DIR / f"{stem}.pdf",
        FIG_DIR / f"{stem}.svg",
        FIG_DIR / f"{stem}.tiff",
    ]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def load_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(OUT / "processed" / "nps_prior_value_predictions.csv", parse_dates=["date"])
    test = pred[pred["split"].eq("test")].copy()
    wide = test.pivot(index="date", columns="model_name", values="pred_tn").sort_index()
    meta = test.drop_duplicates("date").set_index("date").loc[wide.index].copy()

    ds = pd.read_csv(OUT / "processed" / "model_dataset_daily.csv", parse_dates=["date"])
    ds = ds.set_index("date")
    cols = [
        "connect_rank_trainfit",
        "exchange_rank_trainfit",
        "connect",
        "exchange_strength",
        "L_corr",
        "L0",
        "L_eff",
        "api3",
        "precip",
        "state_prob_s1",
        "state_prob_s2",
        "state_prob_s3",
        "state_prob_s4",
    ]
    available = [c for c in cols if c in ds.columns]
    meta = meta.join(ds[available], how="left", rsuffix="_ds")
    meta["abs_error_no_swat"] = (wide["no_swat"] - meta["target_tn_day"]).abs()
    meta["abs_error_raw_swat"] = (wide["raw_swat"] - meta["target_tn_day"]).abs()
    meta["abs_error_with_swat"] = (wide["with_swat"] - meta["target_tn_day"]).abs()
    meta["abs_error_reduction"] = meta["abs_error_no_swat"] - meta["abs_error_with_swat"]
    meta["state_label"] = meta["state_label"].astype(str)
    meta["storm_label"] = np.where(meta["storm_window_3d"].astype(int).eq(1), "Rainfall window", "Non-rainfall")
    return test, wide, meta


def make_stratified_table(meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = [
        ("Overall", pd.Series(True, index=meta.index)),
        ("Rainfall window", meta["storm_window_3d"].astype(int).eq(1)),
        ("Non-rainfall", meta["storm_window_3d"].astype(int).eq(0)),
        ("S1 low exchange", meta["state_label"].eq("S1")),
        ("S2 normal exchange", meta["state_label"].eq("S2")),
        ("S3 high exchange", meta["state_label"].eq("S3")),
    ]
    for label, mask in groups:
        sub = meta.loc[mask]
        if sub.empty:
            continue
        rmse_no = float(np.sqrt(np.mean((sub["abs_error_no_swat"]) ** 2)))
        rmse_with = float(np.sqrt(np.mean((sub["abs_error_with_swat"]) ** 2)))
        rows.append(
            {
                "scenario": label,
                "n": len(sub),
                "no_swat_rmse": rmse_no,
                "response_calibrated_swat_rmse": rmse_with,
                "rmse_reduction_pct": 100 * (rmse_no - rmse_with) / rmse_no,
                "mean_abs_error_reduction": float(sub["abs_error_reduction"].mean()),
                "better_days": int((sub["abs_error_reduction"] > 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLE_DIR / "table32_gate_connectivity_response_summary.csv", index=False)
    return out


def shade_states(ax: plt.Axes, meta: pd.DataFrame, alpha: float = 0.09) -> None:
    state = meta["state_label"]
    groups = (state != state.shift()).cumsum()
    for _, g in meta.groupby(groups):
        s = str(g["state_label"].iloc[0])
        ax.axvspan(g.index.min(), g.index.max(), color=STATE_COLORS.get(s, COLORS["gray"]), alpha=alpha, lw=0)


def shade_boolean_windows(ax: plt.Axes, index: pd.Index, mask: pd.Series, color: str, alpha: float = 0.16) -> None:
    flags = pd.Series(mask.to_numpy(dtype=bool), index=index)
    groups = (flags != flags.shift()).cumsum()
    for _, group in flags.groupby(groups):
        if bool(group.iloc[0]):
            ax.axvspan(group.index.min(), group.index.max(), color=color, alpha=alpha, lw=0, zorder=0)


def fig32_gate_connectivity_response_main(wide: pd.DataFrame, meta: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(8.4, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    plot_rows = summary[summary["scenario"].isin(["Rainfall window", "Non-rainfall", "S1 low exchange", "S2 normal exchange", "S3 high exchange"])].copy()
    y = np.arange(len(plot_rows))
    h = 0.34
    ax_a.barh(y + h / 2, plot_rows["no_swat_rmse"], height=h, color="#8A8F98", label="No SWAT")
    ax_a.barh(y - h / 2, plot_rows["response_calibrated_swat_rmse"], height=h, color=COLORS["vermillion"], label="Response-calibrated SWAT")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{r.scenario} (n={int(r.n)})" for r in plot_rows.itertuples()])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("RMSE (mg L$^{-1}$)")
    ax_a.set_title("a  Error reduction by hydrological context", loc="left", fontweight="bold")
    ax_a.set_ylim(len(plot_rows) - 0.45, -0.78)
    ax_a.scatter([0.28, 0.28], [2.55, 2.95], marker="s", s=42, color=["#8A8F98", COLORS["vermillion"]], clip_on=False)
    ax_a.text(0.295, 2.55, "No SWAT", va="center", ha="left", fontsize=6.8)
    ax_a.text(0.295, 2.95, "Response-calibrated SWAT", va="center", ha="left", fontsize=6.8)
    xmax = max(plot_rows["no_swat_rmse"].max(), plot_rows["response_calibrated_swat_rmse"].max())
    ax_a.set_xlim(0, xmax * 1.75)
    for i, r in enumerate(plot_rows.itertuples()):
        ax_a.text(max(r.no_swat_rmse, r.response_calibrated_swat_rmse) + 0.006, i, f"{r.rmse_reduction_pct:.1f}%", va="center", color=COLORS["ink"], fontsize=6.6)

    storm = meta["storm_window_3d"].astype(int).eq(1)
    shade_boolean_windows(ax_b, meta.index, storm, COLORS["rain_window"], alpha=0.78)
    ax_b.plot(meta.index, meta["target_tn_day"], color=COLORS["ink"], lw=1.3, marker="o", ms=2.4, label="Observed")
    ax_b.plot(wide.index, wide["no_swat"], color="#8A8F98", lw=1.1, ls="--", label="No SWAT")
    ax_b.plot(wide.index, wide["with_swat"], color=COLORS["vermillion"], lw=1.4, label="Response-calibrated SWAT")
    ax_b.set_title("b  Test-period response under gate states", loc="left", fontweight="bold")
    ax_b.set_ylabel("Total nitrogen (mg L$^{-1}$)")
    ax_b.set_ylim(1.14, 2.74)
    ax_b.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_b.tick_params(axis="x", rotation=35)
    handles, labels = ax_b.get_legend_handles_labels()
    handles.append(Patch(facecolor=COLORS["rain_window"], alpha=0.78, edgecolor="none"))
    labels.append("Rainfall-affected period")
    ax_b.legend(handles, labels, frameon=False, loc="upper left", ncol=2, handlelength=1.6, columnspacing=0.9)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18, wspace=0.34)
    return save_figure(fig, "fig34_gate_connectivity_tn_response")


def fig33_gate_condition_distributions(meta: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    fig = plt.figure(figsize=(7.8, 5.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.12], width_ratios=[0.92, 1.18], hspace=0.26, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    counts = meta["state_label"].value_counts().reindex(["S1", "S2", "S3"]).fillna(0).astype(int)
    ax_a.bar(np.arange(len(counts)), counts.values, color=[STATE_COLORS.get(s, COLORS["gray"]) for s in counts.index], width=0.62)
    ax_a.set_xticks(np.arange(len(counts)))
    ax_a.set_xticklabels(list(counts.index))
    ax_a.set_ylabel("Test days")
    ax_a.set_title("a  Test-window gate-state composition", loc="left", fontweight="bold")
    for i, v in enumerate(counts.values):
        ax_a.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=7)
    ax_a.set_ylim(0, max(counts.max() * 1.22, 4))

    categories = [
        ("Rain", meta["storm_window_3d"].astype(int).eq(1)),
        ("Dry", meta["storm_window_3d"].astype(int).eq(0)),
        ("S1", meta["state_label"].eq("S1")),
        ("S2", meta["state_label"].eq("S2")),
        ("S3", meta["state_label"].eq("S3")),
    ]
    data = [meta.loc[m, "abs_error_reduction"].dropna().to_numpy(dtype=float) for _, m in categories]
    pos = np.arange(len(data))
    parts = ax_b.violinplot(data, positions=pos, showmeans=False, showextrema=False, widths=0.72)
    for i, body in enumerate(parts["bodies"]):
        label = categories[i][0]
        body.set_facecolor(STATE_COLORS.get(label, COLORS["sky"] if label == "Rain" else "#B8BCC2"))
        body.set_alpha(0.35)
        body.set_edgecolor("none")
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax_b.scatter(np.full(len(vals), i) + jitter, vals, s=12, color=COLORS["ink"], alpha=0.55, linewidth=0)
        ax_b.plot([i - 0.25, i + 0.25], [np.median(vals), np.median(vals)], color=COLORS["vermillion"], lw=1.2)
    ax_b.axhline(0, color="#444444", lw=0.7)
    ax_b.set_xticks(pos)
    ax_b.set_xticklabels([f"{label}\nn={len(vals)}" for (label, _), vals in zip(categories, data)], rotation=0)
    ax_b.set_ylabel("Absolute-error reduction (mg L$^{-1}$)")
    ax_b.set_title("b  Daily improvements are context-dependent", loc="left", fontweight="bold")

    prob_cols = ["state_prob_s1", "state_prob_s2", "state_prob_s3"]
    available = [c for c in prob_cols if c in meta.columns]
    bottom = np.zeros(len(meta))
    for col, s in zip(prob_cols, ["S1", "S2", "S3"]):
        if col not in meta:
            continue
        vals = meta[col].fillna(0).to_numpy(dtype=float)
        ax_c.bar(meta.index, vals, bottom=bottom, width=0.9, color=STATE_COLORS[s], alpha=0.82, label=STATE_LABELS[s])
        bottom += vals
    ax_c.set_ylim(0, 1.02)
    ax_c.set_ylabel("State probability")
    ax_c.set_title("c  Hydraulic-state mixture during the test period", loc="left", fontweight="bold")
    ax_c.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_c.tick_params(axis="x", rotation=35)
    ax_c.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), columnspacing=1.0, handlelength=1.4)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.965, bottom=0.16, hspace=0.26, wspace=0.22)
    return save_figure(fig, "fig35_gate_state_response_distribution")


def fig33_context_specific_swat_gain(meta: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    """Focused Section 3.3 figure: identify where calibrated-SWAT gains concentrate."""
    contexts = ["Rainfall window", "Non-rainfall", "S1 low exchange", "S2 normal exchange", "S3 high exchange"]
    plot = summary[summary["scenario"].isin(contexts)].copy()
    plot["scenario"] = pd.Categorical(plot["scenario"], categories=contexts, ordered=True)
    plot = plot.sort_values("scenario").reset_index(drop=True)

    fig = plt.figure(figsize=(7.2, 3.95), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    y = np.arange(len(plot))
    ax_a.hlines(y, plot["response_calibrated_swat_rmse"], plot["no_swat_rmse"], color="#B8BCC2", lw=2.2, zorder=1)
    ax_a.scatter(plot["no_swat_rmse"], y, s=42, color="#7B818C", label="No SWAT", zorder=3)
    ax_a.scatter(plot["response_calibrated_swat_rmse"], y, s=48, color=COLORS["vermillion"], label="Response-calibrated SWAT", zorder=3)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"{r.scenario} (n={int(r.n)})" for r in plot.itertuples()])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("RMSE (mg L$^{-1}$)")
    ax_a.set_title("a  Context-specific RMSE reduction", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, loc="center right", bbox_to_anchor=(0.98, 0.46), handletextpad=0.4)
    ax_a.set_xlim(0.105, 0.265)
    for i, r in enumerate(plot.itertuples()):
        ax_a.text(
            max(r.no_swat_rmse, r.response_calibrated_swat_rmse) + 0.004,
            i,
            f"{r.rmse_reduction_pct:.1f}%",
            va="center",
            ha="left",
            fontsize=6.7,
            color=COLORS["ink"],
        )

    colors = [COLORS["sky"], "#A9B1BB", STATE_COLORS["S1"], STATE_COLORS["S2"], STATE_COLORS["S3"]]
    ax_b.barh(y, plot["mean_abs_error_reduction"], color=colors, height=0.62, alpha=0.92)
    ax_b.axvline(0, color=COLORS["ink"], lw=0.7)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Mean daily |e| reduction (mg L$^{-1}$)")
    ax_b.set_title("b  Daily gain magnitude and consistency", loc="left", fontweight="bold")
    ax_b.set_xlim(0, max(plot["mean_abs_error_reduction"].max() * 1.34, 0.11))
    for i, r in enumerate(plot.itertuples()):
        ax_b.text(
            r.mean_abs_error_reduction + 0.003,
            i,
            f"{r.better_days}/{int(r.n)} days",
            va="center",
            ha="left",
            fontsize=6.5,
            color=COLORS["ink"],
        )
    ax_b.text(
        0.98,
        0.04,
        "S3 has only 5 test days; descriptive only.",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )

    for ax in (ax_a, ax_b):
        ax.grid(axis="x", alpha=0.20)
        ax.grid(axis="y", alpha=0)
        ax.tick_params(direction="out", length=2.6, width=0.65, colors=COLORS["ink"])
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(0.7)
            ax.spines[spine].set_color(COLORS["ink"])

    return save_figure(fig, "fig33_context_specific_swat_gain")


def main() -> None:
    setup_style()
    _, wide, meta = load_test_data()
    summary = make_stratified_table(meta)
    paths = []
    paths.extend(fig32_gate_connectivity_response_main(wide, meta, summary))
    paths.extend(fig33_gate_condition_distributions(meta, summary))
    paths.extend(fig33_context_specific_swat_gain(meta, summary))
    print("Wrote", TABLE_DIR / "table32_gate_connectivity_response_summary.csv")
    print(summary.to_string(index=False))
    for p in paths:
        print("Wrote", p)


if __name__ == "__main__":
    main()
