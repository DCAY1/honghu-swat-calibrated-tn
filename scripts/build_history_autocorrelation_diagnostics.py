from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/processed/reconstruction_dataset.csv")
DEFAULT_OUTPUT = Path("outputs/tables/tableS5_history_autocorrelation_diagnostics.csv")


def _nse(actual: pd.Series, predicted: pd.Series) -> float:
    denominator = float(((actual - actual.mean()) ** 2).sum())
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - ((predicted - actual) ** 2).sum() / denominator)


def build_history_diagnostics(frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    required = {"split", "mask_y", "target_tn_day", "tn_history_feature"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    subset = frame.loc[
        frame["split"].astype(str).eq(split)
        & pd.to_numeric(frame["mask_y"], errors="coerce").fillna(0).eq(1),
        ["target_tn_day", "tn_history_feature"],
    ].copy()
    subset["target_tn_day"] = pd.to_numeric(subset["target_tn_day"], errors="coerce")
    subset["tn_history_feature"] = pd.to_numeric(subset["tn_history_feature"], errors="coerce")
    subset = subset.dropna()

    actual = subset["target_tn_day"].astype(float)
    persistence = subset["tn_history_feature"].astype(float)
    error = persistence - actual
    n = int(len(subset))
    target_mean = float(actual.mean()) if n else float("nan")
    rmse = float(np.sqrt(np.mean(np.square(error)))) if n else float("nan")
    mae = float(np.mean(np.abs(error))) if n else float("nan")
    corr = float(actual.corr(persistence)) if n >= 2 and actual.nunique() > 1 and persistence.nunique() > 1 else float("nan")
    relative_rmse = float(100.0 * rmse / target_mean) if np.isfinite(target_mean) and target_mean != 0.0 else float("nan")

    return pd.DataFrame(
        [
            {
                "split": split,
                "n": n,
                "history_target_pearson_r": corr,
                "target_mean_mg_l": target_mean,
                "persistence_RMSE_mg_l": rmse,
                "persistence_MAE_mg_l": mae,
                "persistence_NSE": _nse(actual, persistence) if n else float("nan"),
                "persistence_relative_RMSE_pct": relative_rmse,
                "baseline_definition": "pred_tn = tn_history_feature",
            }
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TN history autocorrelation diagnostics for the Honghu manuscript.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    frame = pd.read_csv(args.input)
    diagnostics = build_history_diagnostics(frame, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output, index=False)
    print(f"Wrote history diagnostics: {args.output}")


if __name__ == "__main__":
    main()
