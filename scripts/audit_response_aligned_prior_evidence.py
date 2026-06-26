#!/usr/bin/env python3
"""Audit evidence used in the response-aligned prior manuscript revision.

The script intentionally avoids importing the project model package so it can
run in lightweight environments where optional deep-learning dependencies are
not installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DEFAULT_OUTPUT_DIR = Path("outputs/processed")


def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2))) if len(actual) else np.nan
    mae = float(np.mean(np.abs(pred - actual))) if len(actual) else np.nan
    denom = float(np.sum((actual - np.mean(actual)) ** 2)) if len(actual) else np.nan
    nse = float(1.0 - np.sum((pred - actual) ** 2) / denom) if len(actual) and denom > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "NSE": nse}


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        frame.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )


def prior_proxy(frame: pd.DataFrame) -> np.ndarray:
    candidates = [
        column
        for column in [
            "L_eff",
            "L0",
            "tn_history_feature",
            "precip",
            "api3",
            "api7",
            "connectivity_correction",
        ]
        if column in frame.columns
    ]
    x = numeric_frame(frame, candidates).to_numpy(dtype=float)
    y = pd.to_numeric(frame["target_tn_day"], errors="coerce").to_numpy(dtype=float)
    train_mask = frame["split"].astype(str).eq("train").to_numpy() & np.isfinite(y)
    model = Ridge(alpha=5.0)
    model.fit(x[train_mask], y[train_mask])
    return model.predict(x)


def build_prior_audit(reconstruction: pd.DataFrame, prior_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = prior_predictions.loc[:, ["date", "pred_tn"]].rename(columns={"pred_tn": "explicit_prior"})
    frame = reconstruction.merge(prior, on="date", how="left").sort_values("date").reset_index(drop=True)
    frame["proxy_prior"] = prior_proxy(frame)
    target = pd.to_numeric(frame["target_tn_day"], errors="coerce").to_numpy(dtype=float)
    valid_mask = frame["split"].astype(str).eq("valid").to_numpy() & np.isfinite(target)

    beta_rows: list[dict[str, float]] = []
    explicit = pd.to_numeric(frame["explicit_prior"], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    proxy = pd.to_numeric(frame["proxy_prior"], errors="coerce").ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    for beta in np.round(np.linspace(0.0, 1.0, 11), 1):
        pred = beta * explicit + (1.0 - beta) * proxy
        row = {"beta": float(beta), "split": "valid", "n": int(valid_mask.sum())}
        row.update(compute_metrics(target[valid_mask], pred[valid_mask]))
        beta_rows.append(row)
    beta_grid = pd.DataFrame(beta_rows)

    train = frame[frame["split"].astype(str).eq("train") & frame["target_tn_day"].notna()].copy()
    spearman_rows = []
    for column in ["L_all", "L_key", "L_corr", "L_eff"]:
        spearman_rows.append(
            {
                "metric": "train_spearman_vs_target",
                "prior": "L_align" if column == "L_eff" else column,
                "value": float(train[[column, "target_tn_day"]].corr(method="spearman").iloc[0, 1]),
            }
        )
    state = train.groupby("state_label")["L_eff"].agg(["count", "mean", "median"]).reset_index()
    s1_mean = float(state.loc[state["state_label"].eq("S1"), "mean"].iloc[0])
    s3_mean = float(state.loc[state["state_label"].eq("S3"), "mean"].iloc[0])
    spearman_rows.append({"metric": "train_L_align_S3_to_S1_mean_ratio", "prior": "L_align", "value": s3_mean / s1_mean})
    diagnostics = pd.DataFrame(spearman_rows)
    return beta_grid, diagnostics


def build_decision_case(residual_predictions: pd.DataFrame, reconstruction: pd.DataFrame) -> pd.DataFrame:
    keep = reconstruction.loc[:, ["date", "split", "state_label", "exchange_strength"]].copy()
    frame = residual_predictions.merge(keep, on=["date", "split", "state_label"], how="left")
    window = frame[
        frame["split"].astype(str).eq("test")
        & frame["date"].between(pd.Timestamp("2023-11-10"), pd.Timestamp("2023-11-17"))
        & frame["state_label"].astype(str).eq("S1")
    ].copy()
    return pd.DataFrame(
        [
            {
                "case": "test_S1_low_exchange_2023-11-10_to_2023-11-17",
                "start_date": window["date"].min().date().isoformat(),
                "end_date": window["date"].max().date().isoformat(),
                "n": int(len(window)),
                "mean_exchange_strength": float(window["exchange_strength"].mean()),
                "mean_prior_error_pred_minus_obs": float((window["pred_tn_prior"] - window["target_tn_day"]).mean()),
                "mean_final_error_pred_minus_obs": float((window["pred_tn"] - window["target_tn_day"]).mean()),
                "mean_residual_adjustment": float(window["pred_tn_residual"].mean()),
                "mean_retention_alpha": float(window["alpha_retention"].mean()),
                "mean_retention_adjustment": float(window["pred_residual_retention"].mean()),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    reconstruction = read_csv(args.output_dir / "reconstruction_dataset.csv")
    prior_predictions = read_csv(args.output_dir / "reconstruction_predictions.csv")
    residual_predictions = read_csv(args.output_dir / "state_conditioned_residual_predictions.csv")

    beta_grid, diagnostics = build_prior_audit(reconstruction, prior_predictions)
    decision_case = build_decision_case(residual_predictions, reconstruction)

    beta_path = args.output_dir / "response_aligned_beta_sensitivity.csv"
    diagnostics_path = args.output_dir / "response_aligned_prior_diagnostics.csv"
    decision_path = args.output_dir / "state_diagnostic_decision_case.csv"
    beta_grid.to_csv(beta_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    decision_case.to_csv(decision_path, index=False)

    selected = beta_grid.loc[beta_grid["RMSE"].idxmin()]
    print("Response-aligned prior audit written.")
    print(f"- beta sensitivity: {beta_path}")
    print(f"- prior diagnostics: {diagnostics_path}")
    print(f"- decision case: {decision_path}")
    print(f"- selected beta={selected['beta']:.2f}, valid RMSE={selected['RMSE']:.4f}, NSE={selected['NSE']:.4f}")
    print(f"- S3/S1 L_align mean ratio={diagnostics.loc[diagnostics['metric'].eq('train_L_align_S3_to_S1_mean_ratio'), 'value'].iloc[0]:.2f}")


if __name__ == "__main__":
    main()
