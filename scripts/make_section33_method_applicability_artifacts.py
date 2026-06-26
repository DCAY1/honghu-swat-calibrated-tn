#!/usr/bin/env python3
"""Create Section 3.3 method-applicability table and figure.

The figure is a decision-oriented comparison, not a model leaderboard: it maps
each method by average prediction error and high-TN risk error.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

METHOD_META = {
    "Ridge": {
        "method_class": "统计基线",
        "judgement": "轻量稳健，统计基线中较优",
        "scenario": "快速基线、小样本参照",
        "marker": "o",
    },
    "SVR": {
        "method_class": "非线性机器学习",
        "judgement": "高值误差较大，未形成稳定增益",
        "scenario": "不宜作为当前主模型",
        "marker": "s",
    },
    "XGBoost": {
        "method_class": "树集成模型",
        "judgement": "高值预测失真明显",
        "scenario": "当前样本下不推荐",
        "marker": "^",
    },
    "GRU": {
        "method_class": "序列深度模型",
        "judgement": "时序模型未体现优势",
        "scenario": "样本扩充后再验证",
        "marker": "D",
    },
    "SWAT/DLinear": {
        "method_class": "过程输出序列模型",
        "judgement": "含过程信号，但缺少断面响应校准",
        "scenario": "过程先验参照",
        "marker": "P",
    },
    "响应校准 SWAT 模型": {
        "method_class": "响应校准过程融合模型",
        "judgement": "平均误差与高值误差均较低",
        "scenario": "推荐作为主预测框架",
        "marker": "*",
    },
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
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


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=2.7, width=0.65, colors=COLORS["ink"])
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["ink"])
        ax.spines[spine].set_linewidth(0.7)
    ax.grid(axis="both", color=COLORS["grid"], lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)


def load_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(TABLE_DIR / "table4_model_benchmark.csv")
    required = ["方法", "RMSE (mg/L)", "MAE (mg/L)", "NSE", "Top10% MAE (mg/L)"]
    missing = [col for col in required if col not in metrics.columns]
    if missing:
        raise ValueError(f"Missing columns in table4_model_benchmark.csv: {missing}")
    metrics = metrics[required].copy()
    metrics["方法类别"] = metrics["方法"].map(lambda x: METHOD_META[x]["method_class"])
    metrics["综合判断"] = metrics["方法"].map(lambda x: METHOD_META[x]["judgement"])
    metrics["适用场景"] = metrics["方法"].map(lambda x: METHOD_META[x]["scenario"])
    order = ["Ridge", "SVR", "XGBoost", "GRU", "SWAT/DLinear", "响应校准 SWAT 模型"]
    metrics["order"] = metrics["方法"].map({name: idx for idx, name in enumerate(order)})
    return metrics.sort_values("order").reset_index(drop=True)


def write_applicability_table(metrics: pd.DataFrame) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table = metrics[
        [
            "方法类别",
            "方法",
            "RMSE (mg/L)",
            "MAE (mg/L)",
            "NSE",
            "Top10% MAE (mg/L)",
            "综合判断",
            "适用场景",
        ]
    ].copy()
    for col in ["RMSE (mg/L)", "MAE (mg/L)", "NSE", "Top10% MAE (mg/L)"]:
        table[col] = table[col].map(lambda value: f"{float(value):.4f}")
    csv_path = TABLE_DIR / "table33_method_applicability_matrix.csv"
    md_path = TABLE_DIR / "table33_method_applicability_matrix.md"
    table.to_csv(csv_path, index=False)
    headers = table.columns.tolist()
    rows = table.astype(str).values.tolist()
    with md_path.open("w", encoding="utf-8") as f:
        f.write("**表 3-x. 不同预测方法的误差表现与适用性比较。**\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
    return [csv_path, md_path]


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


def make_tradeoff_figure(metrics: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.85, 3.75), constrained_layout=True)

    x = metrics["MAE (mg/L)"].to_numpy(dtype=float)
    y = metrics["Top10% MAE (mg/L)"].to_numpy(dtype=float)
    nse = metrics["NSE"].to_numpy(dtype=float)

    x_min, x_max = 0.09, 0.24
    y_min, y_max = 0.09, 0.59
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Decision quadrant: low average error and low high-TN error.
    ridge = metrics.loc[metrics["方法"].eq("Ridge")].iloc[0]
    ax.axvspan(x_min, ridge["MAE (mg/L)"], ymin=0, ymax=(ridge["Top10% MAE (mg/L)"] - y_min) / (y_max - y_min),
               color=COLORS["green"], alpha=0.08, lw=0)
    ax.axvline(ridge["MAE (mg/L)"], color=COLORS["gray"], lw=0.65, ls=":", alpha=0.8)
    ax.axhline(ridge["Top10% MAE (mg/L)"], color=COLORS["gray"], lw=0.65, ls=":", alpha=0.8)

    sizes = 42 + 90 * (nse - nse.min()) / max(nse.max() - nse.min(), 1e-9)
    scatter = ax.scatter(
        x,
        y,
        c=nse,
        s=sizes,
        cmap="viridis",
        vmin=0.10,
        vmax=0.80,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

    marker_overrides = {
        "Ridge": (-0.006, 0.028, "right"),
        "SVR": (0.006, 0.006, "left"),
        "XGBoost": (-0.004, 0.018, "right"),
        "GRU": (0.006, -0.010, "left"),
        "SWAT/DLinear": (0.006, -0.020, "left"),
        "响应校准 SWAT 模型": (0.006, 0.006, "left"),
    }
    label_map = {
        "响应校准 SWAT 模型": "Resp.-cal. SWAT",
        "SWAT/DLinear": "SWAT/DLinear",
    }
    for _, row in metrics.iterrows():
        method = row["方法"]
        dx, dy, ha = marker_overrides[method]
        label = label_map.get(method, method)
        ax.text(
            row["MAE (mg/L)"] + dx,
            row["Top10% MAE (mg/L)"] + dy,
            label,
            ha=ha,
            va="center",
            fontsize=6.2,
            color=COLORS["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.8},
            zorder=4,
        )

    ax.annotate(
        "Lower average error\nand lower high-TN error",
        xy=(0.111, 0.145),
        xytext=(0.132, 0.205),
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": COLORS["gray"]},
        fontsize=6.25,
        color=COLORS["gray"],
        ha="left",
        va="center",
    )
    ax.text(
        0.101,
        0.566,
        "Preferred\nregion",
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["green"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.0},
    )
    ax.set_title("Method applicability under average-error and high-TN-risk objectives", loc="left", fontweight="bold")
    ax.set_xlabel("MAE on the test set (mg/L)")
    ax.set_ylabel("Top 10% TN-sample MAE (mg/L)")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("NSE")
    cbar.ax.tick_params(labelsize=6.2, length=2.2, width=0.6)
    style_axis(ax)

    legend = ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["gray"], lw=0.65, ls=":", label="Ridge baseline thresholds"),
        ],
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0,
    )
    legend.set_zorder(5)
    return save_figure(fig, "fig33_method_applicability_tradeoff")


def main() -> None:
    setup_style()
    metrics = load_metrics()
    table_paths = write_applicability_table(metrics)
    figure_paths = make_tradeoff_figure(metrics)
    print("tables")
    for path in table_paths:
        print(path)
    print("figures")
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
