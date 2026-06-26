from __future__ import annotations

import argparse
from pathlib import Path

from honghu_nps.data_loading import SourcePaths
from honghu_nps.pipeline import run_full_pipeline

DATA_ROOT = Path(__file__).resolve().parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Honghu non-point source pollution pipeline.")
    parser.add_argument(
        "--meteorology",
        default=str(DATA_ROOT / "meteorology" / "气象数据_2023合并.xlsx"),
    )
    parser.add_argument(
        "--water-quality",
        default=str(DATA_ROOT / "water_quality" / "2021-2025水质日平均值结果.xlsx"),
    )
    parser.add_argument(
        "--swat",
        default=str(DATA_ROOT / "swat" / "SWAT_2023_Final_Data.xlsx"),
    )
    parser.add_argument(
        "--hydrology",
        default=str(DATA_ROOT / "hydrology" / "水文数据.xlsx"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "outputs" / "pipeline"),
    )
    parser.add_argument("--split-date", default="2023-10-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = SourcePaths(
        meteorology=args.meteorology,
        water_quality=args.water_quality,
        swat=args.swat,
        hydrology=args.hydrology,
    )
    outputs = run_full_pipeline(paths, output_dir=args.output_dir, split_date=args.split_date)
    print(f"response_dataset_rows={len(outputs.response_dataset)}")
    print(f"prediction_rows={len(outputs.predictions)}")
    print(f"metrics_rows={len(outputs.metrics)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
