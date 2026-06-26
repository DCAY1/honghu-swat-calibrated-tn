import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_nps_prior_value_experiment import load_experiment_frame


class NpsPriorValueExperimentTests(unittest.TestCase):
    def test_load_experiment_frame_merges_response_aligned_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            dates = pd.date_range("2023-01-01", periods=3, freq="D")
            pd.DataFrame(
                {
                    "date": dates,
                    "target_tn_day": [1.0, 1.1, 1.2],
                    "mask_y": [1, 1, 1],
                    "split": ["train", "valid", "test"],
                    "L0": [10.0, 20.0, 30.0],
                }
            ).to_csv(output_dir / "reconstruction_dataset.csv", index=False)
            pd.DataFrame(
                {
                    "date": dates,
                    "pred_tn": [0.9, 1.05, 1.18],
                    "pred_tn_reconstruction_raw": [0.91, 1.06, 1.19],
                    "pred_tn_dlinear_anchor": [0.8, 1.0, 1.1],
                }
            ).to_csv(output_dir / "reconstruction_predictions.csv", index=False)

            frame = load_experiment_frame(output_dir, "reconstruction_dataset.csv")

        self.assertIn("pred_tn_prior", frame.columns)
        self.assertIn("pred_tn_reconstruction_raw", frame.columns)
        self.assertIn("pred_tn_dlinear_anchor", frame.columns)
        self.assertEqual(frame["pred_tn_prior"].tolist(), [0.9, 1.05, 1.18])


if __name__ == "__main__":
    unittest.main()
