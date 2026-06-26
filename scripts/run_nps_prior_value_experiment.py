#!/usr/bin/env python3
"""Test the decision value of aligned SWAT priors for daily TN response prediction."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from honghu_moe_tn_daily.models import add_temporal_split_column, compute_metrics
except ModuleNotFoundError:
    add_temporal_split_column = None

    def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        actual = np.asarray(actual, dtype=float)
        pred = np.asarray(pred, dtype=float)
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2))) if len(actual) else np.nan
        mae = float(np.mean(np.abs(pred - actual))) if len(actual) else np.nan
        denom = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1.0 - np.sum((pred - actual) ** 2) / denom) if len(actual) and denom > 0 else np.nan
        nse = float(1.0 - np.sum((pred - actual) ** 2) / denom) if len(actual) and denom > 0 else np.nan
        if len(actual):
            threshold = float(np.quantile(actual, 0.9))
            top_mask = actual >= threshold
            top10_mae = float(np.mean(np.abs(pred[top_mask] - actual[top_mask]))) if np.any(top_mask) else np.nan
        else:
            top10_mae = np.nan
        return {"RMSE": rmse, "MAE": mae, "R2": r2, "NSE": nse, "Top10% MAE": top10_mae}


DEFAULT_OUTPUT_DIR = Path("outputs/processed")
DEFAULT_INPUT_NAME = "reconstruction_dataset.csv"
RECONSTRUCTION_PREDICTIONS_NAME = "reconstruction_predictions.csv"
ALPHAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

BASE_FEATURE_COLUMNS = [
    "precip",
    "temp_mean",
    "wind_speed",
    "solar_radiation",
    "api3",
    "api7",
    "api15",
    "Hu",
    "Hd",
    "delta_h",
    "Qg_sgn",
    "Qg_abs",
    "flow_mean",
    "flow_min",
    "flow_max",
    "flow_signed_evidence",
    "level_range",
    "flow_range",
    "connect",
    "log_qg_abs",
    "hydraulic_gradient_abs",
    "mix",
    "res",
    "mem",
    "exchange_strength",
    "connect_rank_trainfit",
    "exchange_rank_trainfit",
    "state_score_s1",
    "state_prob_s1",
    "state_score_s2",
    "state_prob_s2",
    "state_score_s3",
    "state_prob_s3",
    "state_score_s4",
    "state_prob_s4",
    "z_delta_h",
    "z_qg_sgn",
    "z_qg_abs",
    "z_connect",
    "z_mix",
    "z_res",
    "z_mem",
    "archive_xtk_up_level",
    "archive_xtk_flow",
    "archive_xtk_level_diff1",
    "archive_xtk_flow_diff1",
    "archive_xtk_flow_roll3",
    "archive_xtk_flow_roll7",
    "archive_hydro_available",
    "tn_history_feature",
    "warm_retention_proxy",
    "wind_resuspension_proxy",
    "bio_season_proxy",
    "storm_window_3d",
    "mechanistic_window",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]

SWAT_PRIOR_COLUMNS = [
    "L_all",
    "L_key",
    "L_corr",
    "L0",
    "L_aux_1",
    "L_aux_2",
    "L_eff",
    "recon_scale",
    "connectivity_correction",
    "tv_weight",
    "lag_weight_0",
    "lag_weight_1",
    "lag_weight_2",
    "lag_weight_3",
    "lag_weight_4",
    "lag_weight_5",
    "lag_weight_6",
    "lag_weight_7",
]

RESPONSE_ALIGNED_PRIOR_COLUMNS = [
    "pred_tn_prior",
    "pred_tn_reconstruction_raw",
    "pred_tn_dlinear_anchor",
]

STRATIFICATION_COLUMNS = ("state_label", "storm_window_3d", "mechanistic_window", "direction")


@dataclass(frozen=True)
class FittedScenario:
    name: str
    predictions: pd.DataFrame
    selected_alpha: float
    validation_rmse: float
    feature_columns: list[str]


def read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} must contain a date column.")
    if "target_tn_day" not in frame.columns:
        raise ValueError(f"{path} must contain a target_tn_day column.")
    frame["date"] = pd.to_datetime(frame["date"])
    if "split" not in frame.columns:
        if add_temporal_split_column is not None:
            frame = add_temporal_split_column(frame)
        else:
            ordered = frame.sort_values("date").reset_index(drop=True)
            train_end = max(int(len(ordered) * 0.7), 1)
            valid_end = min(max(int(len(ordered) * 0.85), train_end + 1), len(ordered))
            ordered["split"] = "train"
            ordered.loc[train_end:valid_end - 1, "split"] = "valid"
            ordered.loc[valid_end:, "split"] = "test"
            frame = ordered
    if "mask_y" not in frame.columns:
        frame["mask_y"] = 1
    return frame.sort_values("date").reset_index(drop=True)


def load_experiment_frame(output_dir: Path, input_name: str = DEFAULT_INPUT_NAME) -> pd.DataFrame:
    frame = read_dataset(output_dir / input_name)
    predictions_path = output_dir / RECONSTRUCTION_PREDICTIONS_NAME
    if not predictions_path.exists():
        return frame
    prior = pd.read_csv(predictions_path)
    if "date" not in prior.columns or "pred_tn" not in prior.columns:
        return frame
    prior["date"] = pd.to_datetime(prior["date"])
    keep_columns = [
        column
        for column in ["date", "pred_tn", "pred_tn_reconstruction_raw", "pred_tn_dlinear_anchor"]
        if column in prior.columns
    ]
    prior = prior.loc[:, keep_columns].rename(columns={"pred_tn": "pred_tn_prior"})
    merged = frame.merge(prior, on="date", how="left")
    for column in RESPONSE_ALIGNED_PRIOR_COLUMNS:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").ffill().bfill()
    return merged


def usable_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        raise ValueError("No usable feature columns were found.")
    matrix = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def shifted_swat_frame(frame: pd.DataFrame, swat_columns: list[str], lag_days: int) -> pd.DataFrame:
    shifted = frame.copy()
    for column in swat_columns:
        shifted[column] = pd.to_numeric(shifted[column], errors="coerce").shift(lag_days)
    shifted.loc[:, swat_columns] = shifted.loc[:, swat_columns].ffill().bfill()
    return shifted


def supervised_mask(frame: pd.DataFrame, split_name: str) -> np.ndarray:
    split = frame["split"].astype(str).eq(split_name)
    mask_y = pd.to_numeric(frame["mask_y"], errors="coerce").fillna(0).astype(int).eq(1)
    finite_target = pd.to_numeric(frame["target_tn_day"], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
    return (split & mask_y & finite_target).to_numpy()


def select_alpha(
    x: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    if not np.any(train_mask):
        raise ValueError("No supervised training rows are available.")
    scoring_mask = valid_mask if np.any(valid_mask) else train_mask
    best_alpha = float(ALPHAS[0])
    best_rmse = float("inf")
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(x.loc[train_mask], y[train_mask])
        pred = np.clip(np.asarray(model.predict(x.loc[scoring_mask]), dtype=float), a_min=0.0, a_max=None)
        rmse = compute_metrics(y[scoring_mask], pred)["RMSE"]
        if rmse < best_rmse:
            best_alpha = float(alpha)
            best_rmse = float(rmse)
    return best_alpha, best_rmse


def fit_scenario(
    frame: pd.DataFrame,
    scenario_name: str,
    feature_columns: list[str],
) -> FittedScenario:
    selected = usable_columns(frame, feature_columns)
    x = numeric_matrix(frame, selected)
    y = pd.to_numeric(frame["target_tn_day"], errors="coerce").to_numpy(dtype=float)
    train_mask = supervised_mask(frame, "train")
    valid_mask = supervised_mask(frame, "valid")
    fit_mask = train_mask | valid_mask if np.any(valid_mask) else train_mask
    alpha, validation_rmse = select_alpha(x, y, train_mask, valid_mask)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x.loc[fit_mask], y[fit_mask])
    pred = np.clip(np.asarray(model.predict(x), dtype=float), a_min=0.0, a_max=None)
    predictions = frame.loc[
        :,
        [
            column
            for column in [
                "date",
                "site",
                "split",
                "target_tn_day",
                "mask_y",
                "state_label",
                "storm_window_3d",
                "mechanistic_window",
                "direction",
            ]
            if column in frame.columns
        ],
    ].copy()
    predictions.insert(0, "model_name", scenario_name)
    predictions["pred_tn"] = pred
    predictions["residual"] = predictions["pred_tn"] - predictions["target_tn_day"]
    predictions["selected_alpha"] = alpha
    predictions["validation_RMSE"] = validation_rmse
    return FittedScenario(
        name=scenario_name,
        predictions=predictions,
        selected_alpha=alpha,
        validation_rmse=validation_rmse,
        feature_columns=selected,
    )


def metrics_for_subset(
    predictions: pd.DataFrame,
    model_name: str,
    split_name: str,
    subset: pd.DataFrame,
    scope: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object] | None:
    working = subset.copy()
    if "mask_y" in working.columns:
        working = working[pd.to_numeric(working["mask_y"], errors="coerce").fillna(0).astype(int).eq(1)]
    working = working.dropna(subset=["target_tn_day", "pred_tn"])
    if working.empty:
        return None
    row: dict[str, object] = {
        "model_name": model_name,
        "split": split_name,
        "scope": scope,
        "n": int(len(working)),
    }
    if extra:
        row.update(extra)
    row.update(compute_metrics(working["target_tn_day"].to_numpy(dtype=float), working["pred_tn"].to_numpy(dtype=float)))
    return row


def summarize_metrics(scenarios: list[FittedScenario]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for split_name in ("train", "valid", "test"):
            subset = scenario.predictions[scenario.predictions["split"].astype(str).eq(split_name)]
            row = metrics_for_subset(
                scenario.predictions,
                scenario.name,
                split_name,
                subset,
                "overall",
                {
                    "selected_alpha": scenario.selected_alpha,
                    "validation_RMSE": scenario.validation_rmse,
                    "n_features": len(scenario.feature_columns),
                    "feature_columns": "|".join(scenario.feature_columns),
                },
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_stratified_metrics(scenarios: list[FittedScenario]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        predictions = scenario.predictions
        for split_name in ("train", "valid", "test"):
            split_frame = predictions[predictions["split"].astype(str).eq(split_name)]
            for column in STRATIFICATION_COLUMNS:
                if column not in split_frame.columns:
                    continue
                values = split_frame[column]
                if column in {"storm_window_3d", "mechanistic_window"}:
                    labels = np.where(pd.to_numeric(values, errors="coerce").fillna(0.0).gt(0.0), "yes", "no")
                    grouped = split_frame.assign(_stratum=labels).groupby("_stratum", dropna=False)
                else:
                    grouped = split_frame.assign(_stratum=values.fillna("missing").astype(str)).groupby("_stratum", dropna=False)
                for stratum_value, subset in grouped:
                    row = metrics_for_subset(
                        predictions,
                        scenario.name,
                        split_name,
                        subset,
                        f"by_{column}",
                        {
                            "stratum_column": column,
                            "stratum_value": str(stratum_value),
                            "selected_alpha": scenario.selected_alpha,
                            "validation_RMSE": scenario.validation_rmse,
                        },
                    )
                    if row is not None:
                        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(output_dir: Path, input_name: str, misalignment_days: int) -> dict[str, pd.DataFrame]:
    frame = load_experiment_frame(output_dir, input_name)
    base_columns = usable_columns(frame, BASE_FEATURE_COLUMNS)
    raw_swat_columns = usable_columns(frame, SWAT_PRIOR_COLUMNS)
    response_prior_columns = usable_columns(frame, RESPONSE_ALIGNED_PRIOR_COLUMNS)
    swat_columns = raw_swat_columns + response_prior_columns
    if not swat_columns:
        raise ValueError("No SWAT prior columns were found in the reconstruction dataset.")

    scenarios = [
        fit_scenario(frame, "no_swat", base_columns),
        fit_scenario(frame, "raw_swat", base_columns + raw_swat_columns),
        fit_scenario(frame, "with_swat", base_columns + swat_columns),
        fit_scenario(shifted_swat_frame(frame, swat_columns, lag_days=misalignment_days), "misaligned_swat", base_columns + swat_columns),
    ]
    predictions = pd.concat([scenario.predictions for scenario in scenarios], ignore_index=True, sort=False)
    metrics = summarize_metrics(scenarios)
    stratified_metrics = summarize_stratified_metrics(scenarios)
    return {
        "predictions": predictions,
        "metrics": metrics,
        "stratified_metrics": stratified_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    parser.add_argument("--misalignment-days", type=int, default=30)
    args = parser.parse_args()

    outputs = run_experiment(args.output_dir, args.input_name, args.misalignment_days)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs["predictions"].to_csv(args.output_dir / "nps_prior_value_predictions.csv", index=False)
    outputs["metrics"].to_csv(args.output_dir / "nps_prior_value_metrics.csv", index=False)
    outputs["stratified_metrics"].to_csv(args.output_dir / "nps_prior_value_stratified_metrics.csv", index=False)

    test_metrics = outputs["metrics"][
        (outputs["metrics"]["split"].astype(str) == "test") & (outputs["metrics"]["scope"].astype(str) == "overall")
    ].copy()
    print(f"Wrote NPS prior-value experiment outputs to {args.output_dir}")
    print(test_metrics.loc[:, ["model_name", "n", "RMSE", "MAE", "NSE", "Top10% MAE", "selected_alpha"]].to_string(index=False))


if __name__ == "__main__":
    main()
