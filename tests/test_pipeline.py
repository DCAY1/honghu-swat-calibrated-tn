import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from honghu_nps.data_loading import (
    normalize_hydrology_frame,
    normalize_meteorology_frame,
    normalize_water_quality_frame,
)
from honghu_nps.hydraulics import build_daily_gate_state, build_hydraulic_proxies
from honghu_nps.modeling import (
    BASE_FEATURES,
    _build_lstm_sequences,
    _fit_linear_model,
    _project_to_physical_domain,
    build_model_upgrade_benchmarks,
    fit_predictive_system,
    select_model_spec,
)
from honghu_nps.pipeline import build_dynamic_site_features
from honghu_nps.prior import build_boundary_prior, build_boundary_prior_identifiability
from honghu_nps.reporting import compute_site_metrics, render_evaluation_artifacts


class DataNormalizationTests(unittest.TestCase):
    def test_normalize_water_quality_frame_keeps_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "断面名称": ["排水闸", "湖心A(洪湖湖心A)"],
                "日期": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "总氮(mg/L)": [1.2, 1.6],
                "总磷(mg/L)": [0.08, 0.11],
            }
        )

        normalized = normalize_water_quality_frame(frame)

        self.assertEqual(
            list(normalized.columns),
            ["date", "site", "target_tn", "target_tp"],
        )
        self.assertEqual(normalized["site"].tolist(), ["排水闸", "湖心A(洪湖湖心A)"])

    def test_normalize_hydrology_frame_extracts_level_and_trend(self) -> None:
        frame = pd.DataFrame(
            {
                "时间": pd.to_datetime(["2023-01-01", "2023-01-01"]),
                "站名": ["新滩口闸上", "新滩口闸下"],
                "闸上水位": ["19.28 ↑", None],
                "闸下水位": [None, "17.42 ↓"],
                "总过闸流量": [89.5, None],
            }
        )

        normalized = normalize_hydrology_frame(frame)

        self.assertIn("level_value", normalized.columns)
        self.assertIn("level_trend", normalized.columns)
        self.assertEqual(normalized["level_value"].round(2).tolist(), [19.28, 17.42])
        self.assertEqual(normalized["level_trend"].tolist(), ["up", "down"])


class PriorAndHydraulicsTests(unittest.TestCase):
    def test_build_daily_gate_state_classifies_gate_mode(self) -> None:
        hydrology = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]),
                "station_name": ["新滩口闸上", "新滩口闸下", "新滩口闸上", "新滩口闸下"],
                "level_value": [19.2, 17.2, 17.1, 18.5],
                "level_trend": ["up", "down", "down", "up"],
                "flow": [90.0, None, 10.0, None],
            }
        )

        gate_state = build_daily_gate_state(hydrology)

        self.assertEqual(
            gate_state["gate_mode"].tolist(),
            ["high_exchange_flush", "backflow_recharge"],
        )
        self.assertEqual(gate_state["direction"].tolist(), ["outflow", "inflow"])

    def test_build_boundary_prior_creates_scenario_bands(self) -> None:
        swat_sub = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "sub_id": [1, 1],
                "sub_area_km2": [100.0, 100.0],
                "precip_mm": [1.0, 25.0],
                "wyld_mm": [0.5, 6.0],
                "org_n_kgha": [0.2, 1.5],
                "org_p_kgha": [0.05, 0.2],
                "no3_surq": [0.1, 1.0],
                "sol_p": [0.02, 0.1],
                "sedp_kgha": [0.01, 0.08],
            }
        )
        swat_reach = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "rch_id": [1, 1],
                "flow_out": [20.0, 40.0],
                "total_n_kg": [50.0, 100.0],
                "total_p_kg": [3.0, 5.0],
            }
        )
        gate_state = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "gate": ["新滩口闸", "新滩口闸"],
                "flow": [50.0, 120.0],
                "head_diff": [2.0, 3.0],
                "direction": ["outflow", "outflow"],
                "gate_mode": ["low_exchange_stagnation", "high_exchange_flush"],
                "connectivity_factor": [0.5, 1.0],
            }
        )
        meteo = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "precipitation": [1.0, 30.0],
                "temp_mean": [8.0, 11.0],
            }
        )

        prior = build_boundary_prior(swat_sub, swat_reach, gate_state, meteo)

        self.assertTrue((prior["prior_tn_conservative"] <= prior["prior_tn_central"]).all())
        self.assertTrue((prior["prior_tn_central"] <= prior["prior_tn_responsive"]).all())
        self.assertIn("effective_input_tn", prior.columns)
        self.assertIn("mass_balance_tn_ratio", prior.columns)
        self.assertTrue((prior["effective_input_tn"] <= prior["storm_amplified_tn"] + 1e-9).all())
        self.assertTrue((prior["mass_balance_tn_ratio"] <= 1.0 + 1e-9).all())
        self.assertGreater(
            prior.loc[prior["date"] == pd.Timestamp("2023-01-02"), "storm_amplified_tn"].iat[0],
            prior.loc[prior["date"] == pd.Timestamp("2023-01-01"), "storm_amplified_tn"].iat[0],
        )
        self.assertIn("effective_boundary_tn_central", prior.columns)
        self.assertIn("effective_boundary_tp_central", prior.columns)
        self.assertNotIn("prior_tn_low", prior.columns)
        self.assertNotIn("prior_tn_mid", prior.columns)
        self.assertNotIn("prior_tn_high", prior.columns)

    def test_build_boundary_prior_identifiability_responds_to_source_connectivity_and_event_switches(self) -> None:
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        swat_sub = pd.DataFrame(
            {
                "date": dates,
                "sub_id": [1, 1, 1],
                "sub_area_km2": [100.0, 100.0, 100.0],
                "precip_mm": [2.0, 30.0, 2.0],
                "wyld_mm": [0.5, 6.0, 0.5],
                "org_n_kgha": [0.5, 1.5, 0.5],
                "org_p_kgha": [0.1, 0.3, 0.1],
                "no3_surq": [0.2, 1.0, 0.2],
                "sol_p": [0.03, 0.1, 0.03],
                "sedp_kgha": [0.02, 0.08, 0.02],
            }
        )
        swat_reach = pd.DataFrame(
            {
                "date": dates,
                "rch_id": [1, 1, 1],
                "flow_out": [20.0, 35.0, 20.0],
                "total_n_kg": [40.0, 90.0, 40.0],
                "total_p_kg": [3.0, 7.0, 3.0],
            }
        )
        gate_state = pd.DataFrame(
            {
                "date": dates,
                "gate": ["新滩口闸"] * 3,
                "flow": [25.0, 100.0, 25.0],
                "head_diff": [0.4, 2.5, -0.3],
                "direction": ["outflow", "outflow", "inflow"],
                "gate_mode": ["low_exchange_stagnation", "high_exchange_flush", "backflow_recharge"],
                "connectivity_factor": [0.25, 0.95, 0.25],
            }
        )
        meteo = pd.DataFrame(
            {
                "date": dates,
                "precipitation": [2.0, 32.0, 2.0],
                "temp_mean": [8.0, 10.0, 9.0],
            }
        )

        baseline = build_boundary_prior(swat_sub, swat_reach, gate_state, meteo)
        low_source = build_boundary_prior(
            swat_sub.assign(org_n_kgha=swat_sub["org_n_kgha"] * 0.5),
            swat_reach.assign(total_n_kg=swat_reach["total_n_kg"] * 0.5),
            gate_state,
            meteo,
        )
        high_connectivity = build_boundary_prior(
            swat_sub,
            swat_reach,
            gate_state.assign(connectivity_factor=1.0, direction="outflow", gate_mode="high_exchange_flush"),
            meteo,
        )
        no_event = build_boundary_prior(
            swat_sub.assign(precip_mm=0.0, wyld_mm=0.0),
            swat_reach,
            gate_state,
            meteo.assign(precipitation=0.0),
        )

        self.assertTrue(
            (
                low_source["effective_boundary_tn_central"]
                <= baseline["effective_boundary_tn_central"] + 1e-9
            ).all()
        )
        self.assertGreater(
            high_connectivity.loc[high_connectivity["date"] == pd.Timestamp("2023-01-02"), "effective_boundary_tn_central"].iat[0],
            baseline.loc[baseline["date"] == pd.Timestamp("2023-01-02"), "effective_boundary_tn_central"].iat[0],
        )
        self.assertGreater(
            baseline.loc[baseline["date"] == pd.Timestamp("2023-01-02"), "effective_boundary_tn_central"].iat[0],
            no_event.loc[no_event["date"] == pd.Timestamp("2023-01-02"), "effective_boundary_tn_central"].iat[0],
        )

    def test_build_boundary_prior_identifiability_returns_named_scenarios(self) -> None:
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        swat_sub = pd.DataFrame(
            {
                "date": dates,
                "sub_id": [1, 1, 1],
                "sub_area_km2": [100.0, 100.0, 100.0],
                "precip_mm": [2.0, 30.0, 2.0],
                "wyld_mm": [0.5, 6.0, 0.5],
                "org_n_kgha": [0.5, 1.5, 0.5],
                "org_p_kgha": [0.1, 0.3, 0.1],
                "no3_surq": [0.2, 1.0, 0.2],
                "sol_p": [0.03, 0.1, 0.03],
                "sedp_kgha": [0.02, 0.08, 0.02],
            }
        )
        swat_reach = pd.DataFrame(
            {
                "date": dates,
                "rch_id": [1, 1, 1],
                "flow_out": [20.0, 35.0, 20.0],
                "total_n_kg": [40.0, 90.0, 40.0],
                "total_p_kg": [3.0, 7.0, 3.0],
            }
        )
        gate_state = pd.DataFrame(
            {
                "date": dates,
                "gate": ["新滩口闸"] * 3,
                "flow": [25.0, 100.0, 25.0],
                "head_diff": [0.4, 2.5, -0.3],
                "direction": ["outflow", "outflow", "inflow"],
                "gate_mode": ["low_exchange_stagnation", "high_exchange_flush", "backflow_recharge"],
                "connectivity_factor": [0.25, 0.95, 0.25],
            }
        )
        meteo = pd.DataFrame(
            {
                "date": dates,
                "precipitation": [2.0, 32.0, 2.0],
                "temp_mean": [8.0, 10.0, 9.0],
            }
        )

        diagnostic = build_boundary_prior_identifiability(swat_sub, swat_reach, gate_state, meteo)

        self.assertEqual(
            sorted(diagnostic["scenario"].unique().tolist()),
            ["baseline", "connectivity_switch", "event_off", "swat_weakened"],
        )
        self.assertIn("effective_boundary_tn_central", diagnostic.columns)

    def test_build_hydraulic_proxies_produces_memory_terms(self) -> None:
        gate_state = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "gate": ["新滩口闸", "新滩口闸"],
                "flow": [20.0, 120.0],
                "head_diff": [0.5, 3.0],
                "direction": ["outflow", "outflow"],
                "gate_mode": ["low_exchange_stagnation", "high_exchange_flush"],
                "connectivity_factor": [0.2, 0.9],
            }
        )

        proxies = build_hydraulic_proxies(gate_state)

        self.assertIn("residence_time_proxy", proxies.columns)
        self.assertIn("lake_level_memory", proxies.columns)
        self.assertIn("mixing_intensity", proxies.columns)
        self.assertIn("hydraulic_memory", proxies.columns)
        self.assertGreater(
            proxies.loc[proxies["date"] == pd.Timestamp("2023-01-01"), "residence_time_proxy"].iat[0],
            proxies.loc[proxies["date"] == pd.Timestamp("2023-01-02"), "residence_time_proxy"].iat[0],
        )

    def test_build_dynamic_site_features_adds_event_and_lag_features(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=8, freq="D"),
                "site": ["排水闸"] * 4 + ["湖心A(洪湖湖心A)"] * 4,
                "travel_lag_days": [1, 1, 1, 1, 3, 3, 3, 3],
                "prior_tn_central": [1, 2, 3, 4, 2, 3, 4, 5],
                "prior_tp_central": [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5],
                "storm_amplified_tn": [1, 2, 3, 5, 2, 3, 4, 6],
                "storm_amplified_tp": [0.1, 0.2, 0.3, 0.5, 0.2, 0.3, 0.4, 0.6],
                "flow": [10, 12, 20, 25, 8, 9, 10, 12],
                "connectivity_factor": [0.1, 0.2, 0.5, 0.7, 0.1, 0.15, 0.2, 0.25],
                "mixing_intensity": [0.1, 0.2, 0.4, 0.6, 0.1, 0.15, 0.2, 0.25],
                "hydraulic_memory": [0.1, 0.15, 0.3, 0.45, 0.1, 0.13, 0.18, 0.2],
                "gate_mode": ["low_exchange_stagnation", "high_exchange_flush", "high_exchange_flush", "backflow_recharge"] * 2,
                "event_flag": [0, 1, 1, 0, 0, 0, 1, 1],
                "precip": [0, 10, 8, 1, 0, 0, 12, 15],
                "target_tn": [1.0] * 8,
                "target_tp": [0.1] * 8,
                "site_lagged_tn_1d": [0.9] * 8,
                "site_lagged_tp_1d": [0.09] * 8,
                "site_lagged_tn_3d": [0.8] * 8,
                "site_lagged_tp_3d": [0.08] * 8,
            }
        )

        featured = build_dynamic_site_features(frame)

        self.assertIn("prior_tn_central_lag_1", featured.columns)
        self.assertIn("storm_amplified_tn_roll3", featured.columns)
        self.assertIn("event_count_7d", featured.columns)
        self.assertNotIn("gate_mode_switch_3d", featured.columns)
        self.assertIn("site_travel_lag_adjusted_prior_tn_central", featured.columns)
        self.assertNotIn("hydraulic_state_index", featured.columns)
        self.assertNotIn("backwater_index", featured.columns)
        self.assertNotIn("stagnation_index", featured.columns)
        row = featured.loc[(featured["site"] == "湖心A(洪湖湖心A)") & (featured["date"] == pd.Timestamp("2023-01-08"))].iloc[0]
        self.assertEqual(row["event_count_7d"], 2)
        self.assertGreater(row["storm_amplified_tn_roll3"], 0)
        self.assertGreater(row["hydraulic_memory"], 0)


class ModelingTests(unittest.TestCase):
    def test_fit_linear_model_handles_large_feature_scale_with_standardization(self) -> None:
        dates = pd.date_range("2023-01-01", periods=80, freq="D")
        signal = np.linspace(1.0, 5.0, len(dates))
        dataset = pd.DataFrame(
            {
                "date": dates,
                "site": ["排水闸"] * len(dates),
                "target_tn": signal,
                "target_tp": np.linspace(0.08, 0.12, len(dates)),
                "prior_tn_conservative": signal - 0.1,
                "prior_tn_central": signal,
                "prior_tn_responsive": signal + 0.1,
                "prior_tp_conservative": np.linspace(0.07, 0.11, len(dates)),
                "prior_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "prior_tp_responsive": np.linspace(0.09, 0.13, len(dates)),
                "effective_boundary_tn_central": signal,
                "effective_boundary_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "prior_tn_central_lag_1": signal,
                "prior_tn_central_lag_3": signal,
                "prior_tn_central_lag_7": signal,
                "prior_tp_central_lag_1": np.linspace(0.08, 0.12, len(dates)),
                "prior_tp_central_lag_3": np.linspace(0.08, 0.12, len(dates)),
                "prior_tp_central_lag_7": np.linspace(0.08, 0.12, len(dates)),
                "site_travel_lag_adjusted_prior_tn_central": signal,
                "site_travel_lag_adjusted_prior_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "storm_amplified_tn_roll3": signal,
                "storm_amplified_tn_roll7": signal,
                "storm_amplified_tp_roll3": np.linspace(0.08, 0.12, len(dates)),
                "storm_amplified_tp_roll7": np.linspace(0.08, 0.12, len(dates)),
                "connectivity_factor": np.linspace(0.2, 0.6, len(dates)),
                "residence_time_proxy": np.linspace(8.0, 4.0, len(dates)),
                "mixing_intensity": np.linspace(0.15, 0.75, len(dates)),
                "hydraulic_memory": np.linspace(0.2, 0.7, len(dates)),
                "precip": np.zeros(len(dates)),
                "event_count_7d": np.zeros(len(dates)),
                "event_precip_7d": np.zeros(len(dates)),
                "temp_mean": np.linspace(5.0, 15.0, len(dates)),
                "site_lagged_tn_1d": signal,
                "site_lagged_tp_1d": np.linspace(0.08, 0.12, len(dates)),
                "gate_mode": ["high_exchange_flush"] * len(dates),
            }
        )

        model, design = _fit_linear_model(dataset, "tn", "elastic_net", alpha=0.05, l1_ratio=0.5)
        coef = pd.Series(model.coef_, index=design.columns)

        self.assertGreater(abs(coef["prior_tn_central"]), 0.0)
        self.assertGreater(abs(coef["effective_boundary_tn_central"]), 0.0)
        self.assertLess(np.sqrt(np.mean((model.predict(design) - signal) ** 2)), 0.1)

    def test_select_model_spec_prefers_simpler_model_for_linear_signal(self) -> None:
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        dataset = pd.DataFrame(
            {
                "date": dates,
                "site": ["排水闸"] * len(dates),
                "target_tn": np.linspace(1.0, 2.0, len(dates)),
                "target_tp": np.linspace(0.08, 0.12, len(dates)),
                "prior_tn_conservative": np.linspace(0.9, 1.9, len(dates)),
                "prior_tn_central": np.linspace(1.0, 2.0, len(dates)),
                "prior_tn_responsive": np.linspace(1.1, 2.1, len(dates)),
                "prior_tp_conservative": np.linspace(0.07, 0.11, len(dates)),
                "prior_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "prior_tp_responsive": np.linspace(0.09, 0.13, len(dates)),
                "effective_boundary_tn_central": np.linspace(1.0, 2.0, len(dates)),
                "effective_boundary_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "connectivity_factor": np.linspace(0.2, 0.6, len(dates)),
                "mixing_intensity": np.linspace(0.2, 0.6, len(dates)),
                "hydraulic_memory": np.linspace(0.2, 0.5, len(dates)),
                "residence_time_proxy": np.linspace(8, 4, len(dates)),
                "precip": np.zeros(len(dates)),
                "temp_mean": np.linspace(5, 15, len(dates)),
                "site_lagged_tn_1d": np.linspace(1.0, 2.0, len(dates)),
                "site_lagged_tp_1d": np.linspace(0.08, 0.12, len(dates)),
                "gate_mode": ["high_exchange_flush"] * len(dates),
            }
        )

        spec = select_model_spec(dataset, target="tn")

        self.assertEqual(spec.residual_kind, "none")

    def test_select_model_spec_can_choose_upper_tail_weighting_for_spikes(self) -> None:
        dates = pd.date_range("2023-01-01", periods=80, freq="D")
        prior = np.linspace(1.0, 7.0, len(dates))
        target = prior + np.clip(prior - 5.0, 0.0, None) * 8.0
        dataset = pd.DataFrame(
            {
                "date": dates,
                "site": ["湖心B(洪湖湖心B)"] * len(dates),
                "target_tn": target,
                "target_tp": np.linspace(0.08, 0.12, len(dates)),
                "prior_tn_conservative": prior - 0.2,
                "prior_tn_central": prior,
                "prior_tn_responsive": prior + 0.2,
                "prior_tp_conservative": np.linspace(0.07, 0.11, len(dates)),
                "prior_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "prior_tp_responsive": np.linspace(0.09, 0.13, len(dates)),
                "effective_boundary_tn_central": prior,
                "effective_boundary_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "prior_tn_central_lag_1": pd.Series(prior).shift(1).bfill(),
                "prior_tn_central_lag_3": pd.Series(prior).shift(3).bfill(),
                "prior_tn_central_lag_7": pd.Series(prior).shift(7).bfill(),
                "prior_tp_central_lag_1": np.linspace(0.08, 0.12, len(dates)),
                "prior_tp_central_lag_3": np.linspace(0.08, 0.12, len(dates)),
                "prior_tp_central_lag_7": np.linspace(0.08, 0.12, len(dates)),
                "site_travel_lag_adjusted_prior_tn_central": prior,
                "site_travel_lag_adjusted_prior_tp_central": np.linspace(0.08, 0.12, len(dates)),
                "storm_amplified_tn_roll3": prior,
                "storm_amplified_tn_roll7": prior,
                "storm_amplified_tp_roll3": np.linspace(0.08, 0.12, len(dates)),
                "storm_amplified_tp_roll7": np.linspace(0.08, 0.12, len(dates)),
                "connectivity_factor": np.linspace(0.2, 0.9, len(dates)),
                "residence_time_proxy": np.linspace(8, 2, len(dates)),
                "mixing_intensity": np.linspace(0.2, 0.9, len(dates)),
                "hydraulic_memory": np.linspace(0.15, 0.85, len(dates)),
                "precip": np.linspace(0, 20, len(dates)),
                "event_count_7d": np.linspace(0, 7, len(dates)),
                "event_precip_7d": np.linspace(0, 30, len(dates)),
                "temp_mean": np.linspace(5, 20, len(dates)),
                "site_lagged_tn_1d": pd.Series(target).shift(1).bfill(),
                "site_lagged_tp_1d": np.linspace(0.08, 0.12, len(dates)),
                "gate_mode": ["high_exchange_flush"] * len(dates),
            }
        )

        spec = select_model_spec(dataset, target="tn")

        self.assertEqual(spec.weight_scheme, "upper_tail_q85")

    def test_project_to_physical_domain_enforces_non_negative_and_reasonable_cap(self) -> None:
        frame = pd.DataFrame(
            {
                "site_lagged_tn_1d": [1.2, 2.0],
                "prior_tn_central": [1.0, 1.0],
                "effective_boundary_tn_central": [0.8, 1.1],
                "mass_balance_tn_ratio": [0.5, 0.9],
                "mixing_intensity": [0.2, 0.8],
                "residence_time_proxy": [3.0, 1.5],
            }
        )

        projected, adjustment = _project_to_physical_domain(np.array([-2.0, 10.0]), frame, "tn")

        self.assertGreaterEqual(projected[0], 0.0)
        self.assertLess(projected[1], 10.0)
        self.assertEqual(len(adjustment), 2)

    def test_build_lstm_sequences_uses_fixed_lookback(self) -> None:
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        frame = pd.DataFrame(
            {
                "date": dates,
                "site": ["排水闸"] * len(dates),
                "target_tn": np.linspace(1.0, 2.0, len(dates)),
                "swat_tn_raw": np.linspace(10.0, 20.0, len(dates)),
                "precip": np.linspace(0.0, 5.0, len(dates)),
                "event_flag": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                "season_sin": np.sin(2 * np.pi * dates.month / 12.0),
                "season_cos": np.cos(2 * np.pi * dates.month / 12.0),
            }
        )

        seq_x, seq_y = _build_lstm_sequences(frame, "tn", ["swat_tn_raw", "precip", "event_flag", "season_sin", "season_cos"], lookback=7)

        self.assertEqual(seq_x.shape, (3, 7, 5))
        self.assertEqual(seq_y.shape, (3,))

    def test_fit_predictive_system_returns_predictions_and_contributions(self) -> None:
        dataset = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10, freq="D").tolist() * 2,
                "site": ["排水闸"] * 10 + ["湖心A(洪湖湖心A)"] * 10,
                "target_tn": [1.0, 1.1, 1.3, 1.4, 1.5, 1.45, 1.6, 1.7, 1.8, 1.75] * 2,
                "target_tp": [0.08, 0.09, 0.11, 0.1, 0.12, 0.115, 0.13, 0.14, 0.15, 0.145] * 2,
                "prior_tn_conservative": [0.8] * 20,
                "prior_tn_central": [1.0] * 20,
                "prior_tn_responsive": [1.2] * 20,
                "prior_tp_conservative": [0.06] * 20,
                "prior_tp_central": [0.08] * 20,
                "prior_tp_responsive": [0.1] * 20,
                "effective_boundary_tn_central": [1.0] * 20,
                "effective_boundary_tp_central": [0.08] * 20,
                "effective_input_tn": [0.9] * 20,
                "effective_input_tp": [0.07] * 20,
                "mass_balance_tn_ratio": [0.8] * 20,
                "mass_balance_tp_ratio": [0.8] * 20,
                "connectivity_factor": [0.2, 0.25, 0.3, 0.4, 0.7, 0.6, 0.8, 0.82, 0.9, 0.5] * 2,
                "mixing_intensity": [0.15, 0.2, 0.25, 0.35, 0.55, 0.5, 0.65, 0.68, 0.72, 0.4] * 2,
                "hydraulic_memory": [0.18, 0.2, 0.24, 0.32, 0.48, 0.52, 0.6, 0.66, 0.72, 0.62] * 2,
                "residence_time_proxy": [8, 7, 6, 5, 4, 4, 3, 3, 2, 4] * 2,
                "precip": [0, 0, 5, 8, 25, 10, 0, 20, 15, 0] * 2,
                "temp_mean": [6, 6, 8, 10, 12, 14, 15, 16, 17, 18] * 2,
                "site_lagged_tn_1d": [1.0, 1.0, 1.1, 1.3, 1.4, 1.5, 1.45, 1.6, 1.7, 1.8] * 2,
                "site_lagged_tp_1d": [0.08, 0.08, 0.09, 0.11, 0.1, 0.12, 0.115, 0.13, 0.14, 0.15] * 2,
                "gate_mode": ["low_exchange_stagnation", "low_exchange_stagnation", "low_exchange_stagnation", "high_exchange_flush", "high_exchange_flush", "high_exchange_flush", "high_exchange_flush", "high_exchange_flush", "high_exchange_flush", "backflow_recharge"] * 2,
                "event_flag": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0] * 2,
            }
        )

        predictions, metrics = fit_predictive_system(dataset, split_date="2023-01-08")

        self.assertFalse(predictions.empty)
        self.assertIn("pred_tn", predictions.columns)
        self.assertIn("pred_tn_central", predictions.columns)
        self.assertIn("pred_tn_conservative", predictions.columns)
        self.assertIn("pred_tn_responsive", predictions.columns)
        self.assertIn("contribution_boundary_generator_tn", predictions.columns)
        self.assertIn("prediction_interval_tn_low", predictions.columns)
        self.assertIn("selected_linear_tn", predictions.columns)
        self.assertIn("selected_regime_tn", predictions.columns)
        self.assertIn("contribution_hydraulic_state_tn", predictions.columns)
        self.assertIn("contribution_physical_projection_tn", predictions.columns)
        self.assertIn("split", metrics.columns)
        self.assertIn("rmse_tn", metrics.columns)
        self.assertIn("nse_tn", metrics.columns)
        self.assertIn("nse_tp", metrics.columns)
        self.assertIn("storm_amplified_tn_roll3", BASE_FEATURES)
        self.assertTrue((predictions["pred_tn"] >= 0.0).all())
        self.assertTrue((predictions["pred_tp"] >= 0.0).all())
        self.assertIn("selected_site_strategy_tn", predictions.columns)
        self.assertIn("selected_site_strategy_tp", predictions.columns)
        self.assertGreater(
            predictions.loc[predictions["event_flag"] == 1, "prediction_interval_tn_high"].sub(
                predictions.loc[predictions["event_flag"] == 1, "prediction_interval_tn_low"]
            ).mean(),
            predictions.loc[predictions["event_flag"] == 0, "prediction_interval_tn_high"].sub(
                predictions.loc[predictions["event_flag"] == 0, "prediction_interval_tn_low"]
            ).mean(),
        )

    def test_fit_predictive_system_backfills_missing_event_flag(self) -> None:
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        dataset = pd.DataFrame(
            {
                "date": dates.tolist() * 2,
                "site": ["排水闸"] * 10 + ["湖心A(洪湖湖心A)"] * 10,
                "target_tn": np.linspace(1.0, 1.9, 20),
                "target_tp": np.linspace(0.08, 0.18, 20),
                "prior_tn_conservative": [0.8] * 20,
                "prior_tn_central": [1.0] * 20,
                "prior_tn_responsive": [1.2] * 20,
                "prior_tp_conservative": [0.06] * 20,
                "prior_tp_central": [0.08] * 20,
                "prior_tp_responsive": [0.1] * 20,
                "effective_boundary_tn_conservative": [0.8] * 20,
                "effective_boundary_tn_central": [1.0] * 20,
                "effective_boundary_tn_responsive": [1.2] * 20,
                "effective_boundary_tp_conservative": [0.06] * 20,
                "effective_boundary_tp_central": [0.08] * 20,
                "effective_boundary_tp_responsive": [0.1] * 20,
                "effective_input_tn": [0.9] * 20,
                "effective_input_tp": [0.07] * 20,
                "mass_balance_tn_ratio": [0.8] * 20,
                "mass_balance_tp_ratio": [0.8] * 20,
                "connectivity_factor": [0.4] * 20,
                "mixing_intensity": [0.3] * 20,
                "hydraulic_memory": [0.2] * 20,
                "residence_time_proxy": [5.0] * 20,
                "directionality_index": [1.0] * 20,
                "precip": [0.0] * 20,
                "temp_mean": [10.0] * 20,
                "event_count_7d": [0.0] * 20,
                "event_precip_7d": [0.0] * 20,
                "site_lagged_tn_1d": [1.0] * 20,
                "site_lagged_tp_1d": [0.08] * 20,
                "prior_tn_central_lag_1": [1.0] * 20,
                "prior_tn_central_lag_3": [1.0] * 20,
                "prior_tn_central_lag_7": [1.0] * 20,
                "prior_tp_central_lag_1": [0.08] * 20,
                "prior_tp_central_lag_3": [0.08] * 20,
                "prior_tp_central_lag_7": [0.08] * 20,
                "site_travel_lag_adjusted_prior_tn_central": [1.0] * 20,
                "site_travel_lag_adjusted_prior_tp_central": [0.08] * 20,
                "storm_amplified_tn_roll3": [1.0] * 20,
                "storm_amplified_tn_roll7": [1.0] * 20,
                "storm_amplified_tp_roll3": [0.08] * 20,
                "storm_amplified_tp_roll7": [0.08] * 20,
            }
        )

        predictions, _ = fit_predictive_system(dataset, split_date="2023-01-08")

        self.assertIn("event_flag", predictions.columns)
        self.assertTrue((predictions["event_flag"] == 0).all())

    @unittest.skipUnless(HAS_TORCH, "PyTorch is optional for paper reproduction")
    def test_build_model_upgrade_benchmarks_returns_all_comparisons(self) -> None:
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        dataset = pd.DataFrame(
            {
                "date": dates.tolist() * 2,
                "site": ["排水闸"] * 20 + ["湖心A(洪湖湖心A)"] * 20,
                "target_tn": np.linspace(1.0, 3.0, 40),
                "target_tp": np.linspace(0.08, 0.2, 40),
                "swat_tn_raw": np.linspace(10.0, 20.0, 40),
                "swat_tp_raw": np.linspace(1.0, 2.0, 40),
                "prior_tn_central": np.linspace(1.0, 2.0, 40),
                "prior_tp_central": np.linspace(0.08, 0.16, 40),
                "effective_boundary_tn_central": np.linspace(1.0, 2.1, 40),
                "effective_boundary_tp_central": np.linspace(0.08, 0.17, 40),
                "effective_input_tn": np.linspace(0.9, 1.9, 40),
                "effective_input_tp": np.linspace(0.07, 0.15, 40),
                "mass_balance_tn_ratio": [0.8] * 40,
                "mass_balance_tp_ratio": [0.8] * 40,
                "connectivity_factor": np.linspace(0.2, 0.8, 40),
                "mixing_intensity": np.linspace(0.2, 0.7, 40),
                "hydraulic_memory": np.linspace(0.1, 0.6, 40),
                "residence_time_proxy": np.linspace(8.0, 3.0, 40),
                "directionality_index": [1.0] * 40,
                "precip": np.linspace(0.0, 10.0, 40),
                "temp_mean": np.linspace(5.0, 15.0, 40),
                "event_flag": ([0, 1] * 20),
            }
        )

        benchmark = build_model_upgrade_benchmarks(dataset, split_date="2023-01-15")

        self.assertEqual(
            sorted(benchmark["benchmark"].unique().tolist()),
            [
                "Effective prior + Ridge",
                "Effective prior + Ridge + hydraulics",
                "Raw SWAT + LSTM",
                "Raw SWAT + Ridge",
            ],
        )
        self.assertIn("rmse_tn", benchmark.columns)
        self.assertIn("nse_tn", benchmark.columns)
        self.assertIn("nse_tp", benchmark.columns)


class ReportingTests(unittest.TestCase):
    def test_compute_site_metrics_builds_site_target_summary(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]),
                "site": ["排水闸", "排水闸", "杨柴湖", "杨柴湖"],
                "target_tn": [1.0, 2.0, 1.0, 2.0],
                "pred_tn": [1.1, 1.8, 0.9, 2.1],
                "target_tp": [0.1, 0.2, 0.1, 0.2],
                "pred_tp": [0.11, 0.18, 0.09, 0.19],
            }
        )

        metrics = compute_site_metrics(predictions)

        self.assertIn("site", metrics.columns)
        self.assertIn("target", metrics.columns)
        self.assertIn("rmse", metrics.columns)
        self.assertIn("nse", metrics.columns)
        self.assertEqual(sorted(metrics["site"].unique().tolist()), ["ALL", "排水闸", "杨柴湖"])
        self.assertEqual(sorted(metrics["target"].unique().tolist()), ["TN", "TP"])

    def test_render_evaluation_artifacts_writes_sci_figure_set(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                    ]
                ),
                "site": ["排水闸", "排水闸", "排水闸", "杨柴湖", "杨柴湖", "杨柴湖"],
                "target_tn": [1.0, 1.2, 1.1, 0.8, 0.85, 0.9],
                "pred_tn": [0.95, 1.18, 1.08, 0.82, 0.83, 0.88],
                "target_tp": [0.08, 0.09, 0.10, 0.05, 0.055, 0.06],
                "pred_tp": [0.081, 0.088, 0.098, 0.051, 0.054, 0.059],
            }
        )

        with TemporaryDirectory() as tmp_dir:
            metrics = render_evaluation_artifacts(predictions, tmp_dir)
            figures_dir = Path(tmp_dir) / "figures"

            expected_files = {
                "tn_time_series_by_site.png",
                "tn_time_series_by_site.pdf",
                "tp_time_series_by_site.png",
                "tp_time_series_by_site.pdf",
                "tn_observed_vs_predicted.png",
                "tn_observed_vs_predicted.pdf",
                "tp_observed_vs_predicted.png",
                "tp_observed_vs_predicted.pdf",
                "tn_time_series_all_sites.png",
                "tn_time_series_all_sites.pdf",
                "tp_time_series_all_sites.png",
                "tp_time_series_all_sites.pdf",
                "tn_site_metrics.png",
                "tn_site_metrics.pdf",
                "tp_site_metrics.png",
                "tp_site_metrics.pdf",
            }

            self.assertEqual(expected_files, {path.name for path in figures_dir.iterdir()})
            self.assertEqual(set(metrics["site"]), {"ALL", "排水闸", "杨柴湖"})


if __name__ == "__main__":
    unittest.main()
