from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_history_autocorrelation_diagnostics import build_history_diagnostics, main


class HistoryAutocorrelationDiagnosticsTests(unittest.TestCase):
    def test_build_history_diagnostics_uses_test_persistence_baseline(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=6, freq="D"),
                "split": ["train", "train", "valid", "test", "test", "test"],
                "mask_y": [1, 1, 1, 1, 1, 0],
                "target_tn_day": [1.0, 1.2, 1.3, 2.0, 2.4, 9.0],
                "tn_history_feature": [0.9, 1.0, 1.2, 1.8, 2.2, 8.8],
            }
        )

        diagnostics = build_history_diagnostics(frame)

        self.assertEqual(diagnostics.loc[0, "split"], "test")
        self.assertEqual(int(diagnostics.loc[0, "n"]), 2)
        self.assertAlmostEqual(float(diagnostics.loc[0, "target_mean_mg_l"]), 2.2)
        self.assertAlmostEqual(float(diagnostics.loc[0, "persistence_RMSE_mg_l"]), 0.2)
        self.assertAlmostEqual(float(diagnostics.loc[0, "persistence_MAE_mg_l"]), 0.2)
        self.assertAlmostEqual(float(diagnostics.loc[0, "persistence_relative_RMSE_pct"]), 100.0 * 0.2 / 2.2)

    def test_main_writes_requested_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "reconstruction_dataset.csv"
            output_path = root / "tableS5_history_autocorrelation_diagnostics.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range("2023-11-07", periods=3, freq="D"),
                    "split": "test",
                    "mask_y": 1,
                    "target_tn_day": [2.0, 2.2, 2.5],
                    "tn_history_feature": [1.9, 2.0, 2.3],
                }
            ).to_csv(input_path, index=False)

            main(["--input", str(input_path), "--output", str(output_path)])

            self.assertTrue(output_path.exists())
            written = pd.read_csv(output_path)
            self.assertEqual(int(written.loc[0, "n"]), 3)


if __name__ == "__main__":
    unittest.main()
