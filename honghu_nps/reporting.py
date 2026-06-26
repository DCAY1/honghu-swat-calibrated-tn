from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

STOTEN_COLORS = {
    "observed": "#355C7D",
    "predicted": "#E07A5F",
    "interval": "#F2CC8F",
    "identity": "#7A7A7A",
    "fit": "#4F5D75",
    "grid": "#E6E8EB",
    "spine": "#B8BEC5",
    "text": "#2E3440",
    "bars": ["#6C8EAD", "#A3BE8C", "#D08770", "#B48EAD"],
}


def _nash_sutcliffe_efficiency(actual: pd.Series, pred: pd.Series) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    if len(actual_arr) == 0:
        return float("nan")
    denominator = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    numerator = float(np.sum((actual_arr - pred_arr) ** 2))
    return float(1.0 - numerator / denominator)


def compute_site_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for site_name, site_frame in [("ALL", predictions)] + list(predictions.groupby("site")):
        for target in ("tn", "tp"):
            actual = site_frame[f"target_{target}"]
            pred = site_frame[f"pred_{target}"]
            rows.append(
                {
                    "site": site_name,
                    "target": target.upper(),
                    "rmse": float(mean_squared_error(actual, pred) ** 0.5),
                    "mae": float(mean_absolute_error(actual, pred)),
                    "r2": float(r2_score(actual, pred)),
                    "nse": _nash_sutcliffe_efficiency(actual, pred),
                    "n": int(len(site_frame)),
                }
            )
    return pd.DataFrame(rows)


def summarize_prediction_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    if "event_flag" not in frame.columns:
        frame["event_flag"] = 0
    rows: list[dict[str, float | str]] = []
    for target in ("tn", "tp"):
        width = frame[f"prediction_interval_{target}_high"] - frame[f"prediction_interval_{target}_low"]
        for regime_name, regime_value in [("baseflow", 0), ("event", 1)]:
            regime_mask = frame["event_flag"].eq(regime_value)
            regime_width = width.loc[regime_mask]
            rows.append(
                {
                    "target": target.upper(),
                    "regime": regime_name,
                    "mean_interval_width": float(regime_width.mean()) if not regime_width.empty else 0.0,
                    "n": int(regime_mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def _configure_publication_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Songti SC", "Arial Unicode MS", "DejaVu Sans"],
            "font.serif": ["Times New Roman", "Songti SC", "Arial Unicode MS", "DejaVu Serif"],
            "font.sans-serif": ["Arial Unicode MS", "Songti SC", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": STOTEN_COLORS["spine"],
            "axes.labelcolor": STOTEN_COLORS["text"],
            "axes.titlecolor": STOTEN_COLORS["text"],
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": STOTEN_COLORS["text"],
            "ytick.color": STOTEN_COLORS["text"],
            "legend.fontsize": 8.5,
            "figure.titlesize": 11,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": STOTEN_COLORS["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.grid": False,
        },
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def _apply_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", rotation=0)


def _add_metric_box(ax: plt.Axes, actual: pd.Series, pred: pd.Series, unit_label: str) -> None:
    rmse = mean_squared_error(actual, pred) ** 0.5
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    nse = _nash_sutcliffe_efficiency(actual, pred)
    text = (
        f"N = {len(actual)}\n"
        f"$R^2$ = {r2:.3f}\n"
        f"NSE = {nse:.3f}\n"
        f"RMSE = {rmse:.3f} {unit_label}\n"
        f"MAE = {mae:.3f} {unit_label}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#7F7F7F", "alpha": 0.9},
    )


def _build_site_labels(predictions: pd.DataFrame) -> dict[str, str]:
    sites = list(predictions["site"].drop_duplicates())
    return {site: f"S{idx + 1}" for idx, site in enumerate(sites)}


def _render_site_metrics(metrics: pd.DataFrame, figures_dir: Path, target: str, site_labels: dict[str, str]) -> None:
    target_metrics = metrics[(metrics["target"] == target) & (metrics["site"] != "ALL")].copy()
    target_metrics["site_label"] = target_metrics["site"].map(site_labels)
    melted = target_metrics.melt(
        id_vars=["site", "site_label", "target"],
        value_vars=["rmse", "mae", "r2", "nse"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.barplot(
        data=melted,
        x="site_label",
        y="value",
        hue="metric",
        palette=STOTEN_COLORS["bars"],
        ax=ax,
    )
    ax.set_title(f"{target} Site Metrics")
    ax.set_xlabel("Site")
    ax.set_ylabel("Metric value")
    ax.legend(title=None, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", color=STOTEN_COLORS["grid"], linewidth=0.6)
    _save_figure(fig, figures_dir / f"{target.lower()}_site_metrics")
    plt.close(fig)


def _render_scatter(predictions: pd.DataFrame, figures_dir: Path, target: str, site_labels: dict[str, str]) -> None:
    target_col = f"target_{target.lower()}"
    pred_col = f"pred_{target.lower()}"
    scatter = predictions.loc[:, ["site", target_col, pred_col]].dropna().copy()
    sites = list(scatter["site"].drop_duplicates())
    scatter["site_label"] = scatter["site"].map(site_labels)
    palette = dict(zip(scatter["site_label"].drop_duplicates(), sns.color_palette("colorblind", n_colors=len(sites))))
    unit_label = "mg L$^{-1}$"

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    sns.scatterplot(
        data=scatter,
        x=target_col,
        y=pred_col,
        hue="site_label",
        palette=palette,
        s=40,
        linewidth=0.45,
        edgecolor="#F8F9FA",
        alpha=0.82,
        ax=ax,
    )
    sns.regplot(
        data=scatter,
        x=target_col,
        y=pred_col,
        scatter=False,
        ci=95,
        color=STOTEN_COLORS["fit"],
        line_kws={"linewidth": 1.2},
        ax=ax,
    )
    min_val = min(scatter[target_col].min(), scatter[pred_col].min())
    max_val = max(scatter[target_col].max(), scatter[pred_col].max())
    padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.05
    lower = max(0.0, min_val - padding)
    upper = max_val + padding
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle=(0, (4, 2)),
        color=STOTEN_COLORS["identity"],
        linewidth=1.0,
        label="1:1 line",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{target} Observed vs Predicted")
    ax.set_xlabel(f"Observed {target} ({unit_label})")
    ax.set_ylabel(f"Predicted {target} ({unit_label})")
    _add_metric_box(ax, scatter[target_col], scatter[pred_col], unit_label)
    ax.grid(True, color=STOTEN_COLORS["grid"], linewidth=0.6)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4, borderaxespad=0.3)
    _save_figure(fig, figures_dir / f"{target.lower()}_observed_vs_predicted")
    plt.close(fig)


def _render_time_series_by_site(
    predictions: pd.DataFrame, figures_dir: Path, target: str, site_labels: dict[str, str]
) -> None:
    target_col = f"target_{target.lower()}"
    pred_col = f"pred_{target.lower()}"
    low_col = f"prediction_interval_{target.lower()}_low"
    high_col = f"prediction_interval_{target.lower()}_high"
    sites = list(predictions["site"].drop_duplicates())
    ncols = 2 if len(sites) <= 4 else 3
    nrows = (len(sites) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.1 * ncols, 2.7 * nrows), sharex=False, sharey=False)
    axes_array = np.atleast_1d(axes).ravel()
    observed_color = STOTEN_COLORS["observed"]
    predicted_color = STOTEN_COLORS["predicted"]

    for idx, site in enumerate(sites):
        ax = axes_array[idx]
        site_frame = predictions[predictions["site"] == site].sort_values("date")
        ax.plot(site_frame["date"], site_frame[target_col], color=observed_color, linewidth=1.7, label="Observed")
        ax.plot(
            site_frame["date"],
            site_frame[pred_col],
            color=predicted_color,
            linewidth=1.45,
            linestyle=(0, (4, 2)),
            label="Predicted",
        )
        if low_col in site_frame.columns and high_col in site_frame.columns:
            ax.fill_between(
                site_frame["date"],
                site_frame[low_col],
                site_frame[high_col],
                color=STOTEN_COLORS["interval"],
                alpha=0.22,
                linewidth=0,
                label="95% PI" if idx == 0 else None,
            )
        ax.set_title(site_labels[site])
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{target} (mg L$^{{-1}}$)")
        ax.grid(axis="y", color=STOTEN_COLORS["grid"], linewidth=0.6)
        _apply_date_axis(ax)

    for idx in range(len(sites), len(axes_array)):
        axes_array[idx].axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.suptitle(f"{target} Time Series by Site", y=1.035)
    fig.tight_layout(pad=1.0)
    _save_figure(fig, figures_dir / f"{target.lower()}_time_series_by_site")
    plt.close(fig)


def _render_time_series_all_sites(
    predictions: pd.DataFrame, figures_dir: Path, target: str, site_labels: dict[str, str]
) -> None:
    target_col = f"target_{target.lower()}"
    pred_col = f"pred_{target.lower()}"
    sites = list(predictions["site"].drop_duplicates())
    palette = dict(zip(sites, sns.color_palette(["#6C8EAD", "#D08770", "#8FA6B3", "#A3BE8C"][: len(sites)])))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for site in sites:
        site_frame = predictions[predictions["site"] == site].sort_values("date")
        color = palette[site]
        ax.plot(
            site_frame["date"],
            site_frame[target_col],
            color=color,
            linewidth=1.55,
            alpha=0.9,
            label=f"{site_labels[site]} observed",
        )
        ax.plot(
            site_frame["date"],
            site_frame[pred_col],
            color=color,
            linewidth=1.3,
            linestyle=(0, (4, 2)),
            alpha=0.95,
            label=f"{site_labels[site]} predicted",
        )

    ax.set_title(f"{target} Time Series Across All Sites")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (mg L$^{{-1}}$)")
    ax.grid(axis="y", color=STOTEN_COLORS["grid"], linewidth=0.6)
    _apply_date_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=2.0, columnspacing=1.1)
    _save_figure(fig, figures_dir / f"{target.lower()}_time_series_all_sites")
    plt.close(fig)


def render_evaluation_artifacts(predictions: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_site_metrics(predictions)
    metrics.to_csv(out_dir / "site_target_metrics.csv", index=False)
    site_labels = _build_site_labels(predictions)
    pd.DataFrame({"site": list(site_labels.keys()), "site_code": list(site_labels.values())}).to_csv(
        out_dir / "site_code_mapping.csv", index=False
    )

    _configure_publication_theme()

    for target in ("TN", "TP"):
        _render_site_metrics(metrics, figures_dir, target, site_labels)
        _render_scatter(predictions, figures_dir, target, site_labels)
        _render_time_series_by_site(predictions, figures_dir, target, site_labels)
        _render_time_series_all_sites(predictions, figures_dir, target, site_labels)

    return metrics
