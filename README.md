# Honghu SWAT Calibrated TN Prediction

Data and code for reproducing the paper **融合经响应校准 SWAT 农业面源过程信息的闸控浅水湖泊总氮预测** (response-calibrated SWAT agricultural non-point-source information for daily total nitrogen prediction at the Honghu drainage gate).

## Method summary

- **Target**: daily TN at the Honghu drainage gate (2023-01-01 to 2023-12-26)
- **Model**: standardized ridge regression with three input scenarios:
  1. Conventional monitoring variables only
  2. Conventional + raw SWAT process variables
  3. Conventional + response-calibrated SWAT variables (L_corr, L_eff, gate states)
- **Test performance (calibrated SWAT)**: RMSE 0.13 mg/L, MAE 0.10 mg/L, NSE 0.79

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Table 2 — three-scenario ridge regression
python scripts/run_nps_prior_value_experiment.py

# Figures 4–7
python scripts/make_section31_descriptive_artifacts.py
python scripts/make_section33_method_applicability_artifacts.py
python scripts/make_section32_gate_connectivity_artifacts.py
python scripts/make_section35_swat_contribution_figures.py
```

Figures 1–3 (study area, workflow, 2016 Futian Temple flow calibration) are provided as pre-rendered assets in `outputs/figures/`.

## Repository layout

```
data/              Raw inputs (meteorology, water quality, SWAT, hydrology)
outputs/processed/ Modeling tables (reconstruction_dataset.csv, etc.)
outputs/tables/    Paper table sources
outputs/figures/   Paper figures (PNG/PDF)
scripts/           Reproduction scripts
honghu_nps/        Data loading and hydraulic proxy utilities
```

## Reproducibility levels

1. **Level 1 (guaranteed)**: Regenerate Table 2 and Figures 4–7 from `outputs/processed/` using the scripts above.
2. **Level 2 (optional)**: Rebuild basic NPS features from raw Excel files via `python run_pipeline.py`.
3. **Out of scope**: ArcSWAT rebuild and SUFI-2 recalibration; spatial base layers are cited from public datasets only.

See [PAPER_ARTIFACT_MAP.md](PAPER_ARTIFACT_MAP.md) for figure/table-to-file mapping.

## Publish to GitHub

If https://github.com/qingfenglangyue/honghu-swat-calibrated-tn shows **404**, the remote repository has not been created or `main` has not been pushed yet. From this directory on a machine that can reach GitHub:

```bash
./push_to_github.sh
```

Or manually: create an empty public repo named `honghu-swat-calibrated-tn` at https://github.com/new , then run `git push -u origin main`.

## Citation

If you use this repository, please cite the associated manuscript and the CMFD meteorological forcing dataset (He et al., 2020).

## License

MIT — see [LICENSE](LICENSE).
