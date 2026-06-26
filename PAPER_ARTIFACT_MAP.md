# Paper artifact map

Mapping between the final Chinese manuscript, scripts, and repository files.

| Manuscript | Description | Script | Primary files |
|------------|-------------|--------|---------------|
| 图1 | 洪湖地理位置与流域概况 | Pre-rendered | `outputs/figures/fig1_study_area_workflow.png` |
| 图2 | 研究技术路线 | Pre-rendered | `outputs/figures/fig1_study_area_workflow.png` |
| 图3 | 2016 福田寺流量率定 | Pre-rendered | `outputs/figures/fig2_swat_hydrological_calibration.png`, `flow_monthly_2016.jpg` |
| 图4 | SWAT 面源指数与排水闸 TN (§3.1) | `scripts/make_section31_descriptive_artifacts.py` | `outputs/processed/model_dataset_daily.csv` |
| 图5 | 分情境预测增益 (§3.3) | `scripts/make_section33_method_applicability_artifacts.py` | `outputs/tables/table5_stratified_gain.csv` |
| 图6 | 测试期闸控状态组成 (§3.3) | `scripts/make_section32_gate_connectivity_artifacts.py` | `outputs/processed/gate_state_daily.csv` |
| 图7 | 逐日误差修正诊断 (§3.4) | `scripts/make_section35_swat_contribution_figures.py` | `outputs/processed/nps_prior_value_predictions.csv` |
| 表1 | SWAT 水文参数率定 | Static CSV | `outputs/tables/table1_swat_calibration.csv` |
| 表2 | 三类情景测试集指标 | `scripts/run_nps_prior_value_experiment.py` | `outputs/processed/reconstruction_dataset.csv` → `outputs/tables/table3_swat_prior_evidence_chain.csv` |

Repository: https://github.com/DCAY1/honghu-swat-calibrated-tn

## Expected Table 2 values (test set)

| Scenario | RMSE | MAE | NSE |
|----------|------|-----|-----|
| 常规监测信息 (no_swat) | 0.206 | 0.164 | 0.518 |
| 原始 SWAT (raw_swat) | 0.207 | 0.165 | 0.512 |
| 响应校准 SWAT (with_swat) | 0.136 | 0.109 | 0.791 |

Values are rounded to two decimals in the manuscript.
