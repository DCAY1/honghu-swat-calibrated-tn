from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is expected in the local venv but keep import safe.
    torch = None
    nn = None


HYDRAULIC_CORE_FEATURES = [
    "directionality_index",
    "connectivity_factor",
    "mixing_intensity",
    "hydraulic_memory",
    "residence_time_proxy",
]

HISTORY_FEATURES = [
    "precip",
    "temp_mean",
    "event_count_7d",
    "event_precip_7d",
    "storm_amplified_tn_roll3",
    "storm_amplified_tn_roll7",
    "storm_amplified_tp_roll3",
    "storm_amplified_tp_roll7",
    "prior_tn_central_lag_1",
    "prior_tn_central_lag_3",
    "prior_tn_central_lag_7",
    "prior_tp_central_lag_1",
    "prior_tp_central_lag_3",
    "prior_tp_central_lag_7",
    "site_lagged_tn_1d",
    "site_lagged_tp_1d",
]

BOUNDARY_FEATURES = [
    "prior_tn_central",
    "prior_tp_central",
    "effective_boundary_tn_central",
    "effective_boundary_tp_central",
    "effective_input_tn",
    "effective_input_tp",
    "mass_balance_tn_ratio",
    "mass_balance_tp_ratio",
    "site_travel_lag_adjusted_prior_tn_central",
    "site_travel_lag_adjusted_prior_tp_central",
]

BASE_FEATURES = BOUNDARY_FEATURES + HYDRAULIC_CORE_FEATURES + HISTORY_FEATURES


@dataclass(frozen=True)
class ModelSpec:
    residual_kind: str
    linear_kind: str = "ridge"
    linear_alpha: float = 0.5
    linear_l1_ratio: float = 0.5
    weight_scheme: str = "none"


@dataclass
class TargetModels:
    design_columns: list[str]
    linear: object
    linear_kind: str
    weight_scheme: str
    residual: object | None
    residual_kind: str
    residual_sigma: float
    site_strategy: str = "baseline"


def _boundary_feature_columns(target: str) -> list[str]:
    return [
        f"prior_{target}_central",
        f"effective_boundary_{target}_central",
        f"effective_input_{target}",
        f"mass_balance_{target}_ratio",
        f"site_travel_lag_adjusted_prior_{target}_central",
    ]


def _history_feature_columns(target: str) -> list[str]:
    return [
        f"site_lagged_{target}_1d",
        f"prior_{target}_central_lag_1",
        f"prior_{target}_central_lag_3",
        f"prior_{target}_central_lag_7",
        f"storm_amplified_{target}_roll3",
        f"storm_amplified_{target}_roll7",
        "precip",
        "temp_mean",
        "event_count_7d",
        "event_precip_7d",
    ]


def _group_columns(target: str) -> dict[str, list[str]]:
    return {
        "boundary_generator": _boundary_feature_columns(target),
        "hydraulic_state": HYDRAULIC_CORE_FEATURES,
        "history": _history_feature_columns(target),
    }


def _prepare_design_matrix(
    frame: pd.DataFrame, target: str, include_site_dummies: bool = False
) -> tuple[pd.DataFrame, list[str]]:
    dataset = frame.copy()
    dataset["month"] = dataset["date"].dt.month
    dataset["season_sin"] = np.sin(2 * np.pi * dataset["month"] / 12.0)
    dataset["season_cos"] = np.cos(2 * np.pi * dataset["month"] / 12.0)
    if "directionality_index" not in dataset.columns:
        dataset["directionality_index"] = dataset.get("direction", pd.Series(index=dataset.index, dtype=object)).map(
            {"inflow": -1.0, "stagnant": 0.0, "outflow": 1.0}
        )

    required_cols = sorted(
        set(_boundary_feature_columns(target) + HYDRAULIC_CORE_FEATURES + _history_feature_columns(target) + ["season_sin", "season_cos"])
    )
    for column in required_cols:
        if column not in dataset.columns:
            dataset[column] = 0.0
    design = dataset[required_cols].fillna(0.0).astype(float)
    if include_site_dummies:
        site_dummies = pd.get_dummies(dataset["site"], prefix="site", dtype=float, drop_first=True)
        design = pd.concat([design, site_dummies], axis=1)
    return design, list(design.columns)


def _align_design_matrix(
    frame: pd.DataFrame, target: str, columns: list[str], include_site_dummies: bool = False
) -> pd.DataFrame:
    design, _ = _prepare_design_matrix(frame, target, include_site_dummies=include_site_dummies)
    return design.reindex(columns=columns, fill_value=0.0).astype(float)


def _make_residual_model(kind: str) -> object | None:
    if kind == "none":
        return None
    if kind == "rf_small":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=1,
        )
    if kind == "rf_mid":
        return RandomForestRegressor(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=1,
        )
    if kind == "hgb_small":
        return HistGradientBoostingRegressor(
            max_depth=2,
            max_iter=120,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
    raise ValueError(f"Unknown residual model kind: {kind}")


def _fit_linear_model(
    frame: pd.DataFrame,
    target: str,
    kind: str,
    alpha: float,
    l1_ratio: float,
    weight_scheme: str = "none",
    include_site_dummies: bool = False,
) -> tuple[object, pd.DataFrame]:
    design, columns = _prepare_design_matrix(frame, target, include_site_dummies=include_site_dummies)
    scaler = StandardScaler()
    scaled_design = pd.DataFrame(scaler.fit_transform(design), columns=columns, index=design.index)
    if kind == "ridge":
        model = Ridge(alpha=alpha)
    elif kind == "elastic_net":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=42)
    else:
        raise ValueError(f"Unknown linear model kind: {kind}")
    y = frame[f"target_{target}"].to_numpy()
    sample_weight = _build_sample_weights(y, weight_scheme)
    model.fit(scaled_design, y, sample_weight=sample_weight)

    scale = pd.Series(scaler.scale_, index=columns).replace(0.0, 1.0)
    mean = pd.Series(scaler.mean_, index=columns)
    raw_coef = pd.Series(model.coef_, index=columns) / scale
    raw_intercept = float(model.intercept_ - (mean * raw_coef).sum())

    model.coef_ = raw_coef.to_numpy()
    model.intercept_ = raw_intercept
    model.design_columns_ = columns  # type: ignore[attr-defined]
    return model, design


def _build_sample_weights(target_values: np.ndarray, weight_scheme: str) -> np.ndarray | None:
    if weight_scheme == "none":
        return None
    if weight_scheme == "upper_tail_q85":
        threshold = float(np.quantile(target_values, 0.85))
        return np.where(target_values >= threshold, 5.0, 1.0)
    if weight_scheme == "upper_tail_q90":
        threshold = float(np.quantile(target_values, 0.90))
        return np.where(target_values >= threshold, 5.0, 1.0)
    raise ValueError(f"Unknown weight scheme: {weight_scheme}")


def _temporal_validation_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_dates = frame["date"].sort_values().reset_index(drop=True)
    split_idx = max(int(len(ordered_dates) * 0.8), 1)
    cutoff = ordered_dates.iloc[min(split_idx, len(ordered_dates) - 1)]
    subtrain = frame[frame["date"] < cutoff].copy()
    valid = frame[frame["date"] >= cutoff].copy()
    if subtrain.empty or valid.empty:
        midpoint = max(len(frame) // 2, 1)
        subtrain = frame.iloc[:midpoint].copy()
        valid = frame.iloc[midpoint:].copy()
    return subtrain, valid


def _supports_upper_tail_weighting(train_frame: pd.DataFrame, target: str) -> bool:
    if target != "tn":
        return False
    values = train_frame[f"target_{target}"].to_numpy()
    if len(values) < 60:
        return False
    median = float(np.quantile(values, 0.50))
    q95 = float(np.quantile(values, 0.95))
    if median <= 0.0:
        return False
    return (q95 / median) >= 2.5


def _select_linear_spec(train_frame: pd.DataFrame, target: str, include_site_dummies: bool = False) -> ModelSpec:
    subtrain, valid = _temporal_validation_split(train_frame.sort_values("date").reset_index(drop=True))
    candidates = [ModelSpec(linear_kind="ridge", linear_alpha=0.5, residual_kind="none")]
    allow_upper_tail_weighting = _supports_upper_tail_weighting(train_frame, target)
    if allow_upper_tail_weighting:
        candidates.extend(
            [
                ModelSpec(linear_kind="ridge", linear_alpha=0.5, residual_kind="none", weight_scheme="upper_tail_q85"),
                ModelSpec(linear_kind="ridge", linear_alpha=0.5, residual_kind="none", weight_scheme="upper_tail_q90"),
            ]
        )
    for alpha in (0.001, 0.01, 0.05, 0.1, 0.5):
        for l1_ratio in (0.2, 0.5, 0.8):
            candidates.append(
                ModelSpec(
                    linear_kind="elastic_net",
                    linear_alpha=alpha,
                    linear_l1_ratio=l1_ratio,
                    residual_kind="none",
                )
            )
            if allow_upper_tail_weighting:
                candidates.append(
                    ModelSpec(
                        linear_kind="elastic_net",
                        linear_alpha=alpha,
                        linear_l1_ratio=l1_ratio,
                        residual_kind="none",
                        weight_scheme="upper_tail_q85",
                    )
                )

    y_valid = valid[f"target_{target}"].to_numpy()
    best_spec = candidates[0]
    best_rmse = float("inf")
    improvement_gate = 0.02

    for spec in candidates:
        linear, _ = _fit_linear_model(
            subtrain,
            target,
            spec.linear_kind,
            spec.linear_alpha,
            spec.linear_l1_ratio,
            spec.weight_scheme,
            include_site_dummies=include_site_dummies,
        )
        columns = list(linear.design_columns_)  # type: ignore[attr-defined]
        pred_valid = linear.predict(
            _align_design_matrix(valid, target, columns, include_site_dummies=include_site_dummies)
        )
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred_valid)))
        if best_rmse == float("inf"):
            best_rmse = rmse
            best_spec = spec
            continue
        if spec.linear_kind == "ridge":
            if rmse <= best_rmse:
                best_rmse = rmse
                best_spec = spec
        elif rmse < best_rmse * (1.0 - improvement_gate):
            best_rmse = rmse
            best_spec = spec
    return best_spec


def select_model_spec(train_frame: pd.DataFrame, target: str, include_site_dummies: bool = False) -> ModelSpec:
    return _select_linear_spec(train_frame, target, include_site_dummies=include_site_dummies)


def _fit_target_models(train_frame: pd.DataFrame, target: str, include_site_dummies: bool = False) -> TargetModels:
    spec = select_model_spec(train_frame, target, include_site_dummies=include_site_dummies)
    linear, design = _fit_linear_model(
        train_frame,
        target,
        spec.linear_kind,
        spec.linear_alpha,
        spec.linear_l1_ratio,
        spec.weight_scheme,
        include_site_dummies=include_site_dummies,
    )
    y = train_frame[f"target_{target}"].to_numpy()
    baseline = linear.predict(design)
    residual_model = _make_residual_model(spec.residual_kind)

    if residual_model is not None:
        residual_target = y - baseline
        q_low, q_high = np.quantile(residual_target, [0.01, 0.99])
        residual_target = np.clip(residual_target, q_low, q_high)
        residual_model.fit(design, residual_target)
        residual_pred = residual_model.predict(design)
    else:
        residual_pred = np.zeros_like(y)

    residual_sigma = float(np.std(y - (baseline + residual_pred)))
    return TargetModels(
        design_columns=list(linear.design_columns_),  # type: ignore[attr-defined]
        linear=linear,
        linear_kind=spec.linear_kind,
        weight_scheme=spec.weight_scheme,
        residual=residual_model,
        residual_kind=spec.residual_kind,
        residual_sigma=residual_sigma,
    )


def _scenario_frame(frame: pd.DataFrame, target: str, scenario: str) -> pd.DataFrame:
    scenario_frame = frame.copy()
    scenario_frame[f"prior_{target}_central"] = scenario_frame[f"prior_{target}_{scenario}"]
    boundary_col = f"effective_boundary_{target}_{scenario}"
    if boundary_col in scenario_frame.columns:
        scenario_frame[f"effective_boundary_{target}_central"] = scenario_frame[boundary_col]
    return scenario_frame


def _project_to_physical_domain(raw_prediction: np.ndarray, frame: pd.DataFrame, target: str) -> tuple[np.ndarray, pd.Series]:
    pred = pd.Series(raw_prediction, index=frame.index, dtype=float)
    lagged = pd.Series(frame.get(f"site_lagged_{target}_1d", 0.0), index=frame.index, dtype=float).fillna(0.0)
    prior_central = pd.Series(frame.get(f"prior_{target}_central", 0.0), index=frame.index, dtype=float).fillna(0.0)
    effective_boundary = pd.Series(
        frame.get(f"effective_boundary_{target}_central", frame.get(f"effective_input_{target}", prior_central)),
        index=frame.index,
        dtype=float,
    ).fillna(0.0)
    mass_ratio = pd.Series(frame.get(f"mass_balance_{target}_ratio", 0.0), index=frame.index, dtype=float).fillna(0.0)
    connectivity = pd.Series(frame.get("connectivity_factor", 0.0), index=frame.index, dtype=float).fillna(0.0).clip(0.0, 1.0)
    mixing = pd.Series(frame.get("mixing_intensity", 0.0), index=frame.index, dtype=float).fillna(0.0).clip(0.0, 1.0)
    residence = pd.Series(frame.get("residence_time_proxy", 0.0), index=frame.index, dtype=float).fillna(0.0)

    input_signal = np.log1p(np.maximum(effective_boundary, 0.0))
    memory_floor = np.maximum(0.0, lagged * (0.18 + 0.22 * np.clip(residence, 0.0, 3.0)) * (1.0 - 0.35 * mixing))
    physical_cap = np.maximum(
        memory_floor,
        np.maximum(0.0, lagged) + (0.55 + 0.65 * mass_ratio) * input_signal / (1.0 + 0.35 * mixing),
    )
    physical_cap = physical_cap * (0.85 + 0.30 * connectivity)
    projected = pred.clip(lower=0.0)
    projected = np.minimum(projected, physical_cap + 0.30 * np.maximum(prior_central, 0.0))
    adjustment = projected - pred
    return projected.to_numpy(), adjustment


def _prepare_sequence_frame(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    dataset["month"] = dataset["date"].dt.month
    dataset["season_sin"] = np.sin(2 * np.pi * dataset["month"] / 12.0)
    dataset["season_cos"] = np.cos(2 * np.pi * dataset["month"] / 12.0)
    if "event_flag" not in dataset.columns:
        dataset["event_flag"] = 0
    return dataset


def _build_lstm_sequences(
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    lookback: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    prepared = _prepare_sequence_frame(frame).sort_values(["site", "date"]).reset_index(drop=True)
    for _, site_frame in prepared.groupby("site"):
        site_frame = site_frame.sort_values("date").reset_index(drop=True)
        values = site_frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
        labels = site_frame[f"target_{target}"].to_numpy(dtype=np.float32)
        for idx in range(lookback, len(site_frame)):
            sequences.append(values[idx - lookback : idx])
            targets.append(float(labels[idx]))
    if not sequences:
        return np.empty((0, lookback, len(feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(sequences).astype(np.float32), np.asarray(targets, dtype=np.float32)


def _evaluate_predictions(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "r2": float(r2_score(actual, pred)),
        "nse": _nash_sutcliffe_efficiency(actual, pred),
    }


def _nash_sutcliffe_efficiency(actual: np.ndarray | pd.Series, pred: np.ndarray | pd.Series) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    if len(actual_arr) == 0:
        return float("nan")
    denominator = float(np.sum((actual_arr - actual_arr.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    numerator = float(np.sum((actual_arr - pred_arr) ** 2))
    return float(1.0 - numerator / denominator)


def _ridge_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_columns: list[str],
) -> dict[str, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_columns].fillna(0.0))
    x_test = scaler.transform(test[feature_columns].fillna(0.0))
    model = Ridge(alpha=0.5)
    model.fit(x_train, train[f"target_{target}"].to_numpy())
    pred = model.predict(x_test)
    return _evaluate_predictions(test[f"target_{target}"].to_numpy(), pred)


def _lstm_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    lookback: int = 7,
) -> dict[str, float]:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for the Raw SWAT + LSTM benchmark.")
    train_frame = _prepare_sequence_frame(train)
    test_frame = _prepare_sequence_frame(test)
    x_train, y_train = _build_lstm_sequences(train_frame, target, feature_columns, lookback=lookback)
    x_test, y_test = _build_lstm_sequences(test_frame, target, feature_columns, lookback=lookback)
    if len(x_train) == 0 or len(x_test) == 0:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "nse": 0.0}

    torch.manual_seed(42)

    class SimpleLSTM(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=16, num_layers=1, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(inputs)
            return self.head(output[:, -1, :]).squeeze(-1)

    model = SimpleLSTM(input_size=len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    best_state = None
    best_loss = float("inf")
    patience = 5
    patience_left = patience

    for _ in range(40):
        model.train()
        optimizer.zero_grad()
        pred_train = model(x_train_tensor)
        loss = loss_fn(pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        if loss_value + 1e-6 < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_test, dtype=torch.float32)).cpu().numpy()
    return _evaluate_predictions(y_test, pred)


def build_model_upgrade_benchmarks(dataset: pd.DataFrame, split_date: str) -> pd.DataFrame:
    frame = dataset.copy().sort_values(["site", "date"]).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "event_flag" not in frame.columns:
        frame["event_flag"] = 0
    split_ts = pd.Timestamp(split_date)
    train = frame[frame["date"] < split_ts].copy()
    test = frame[frame["date"] >= split_ts].copy()
    rows: list[dict[str, float | str]] = []
    for target in ("tn", "tp"):
        raw_swat_features = [
            f"swat_{target}_raw",
            "precip",
            "event_flag",
            "temp_mean",
            "season_sin",
            "season_cos",
        ]
        effective_prior_features = [
            f"prior_{target}_central",
            f"effective_boundary_{target}_central",
            f"effective_input_{target}",
            f"mass_balance_{target}_ratio",
            "precip",
            "event_flag",
            "temp_mean",
            "season_sin",
            "season_cos",
        ]
        hydraulics_features = effective_prior_features + HYDRAULIC_CORE_FEATURES
        prepared_train = _prepare_sequence_frame(train)
        prepared_test = _prepare_sequence_frame(test)
        comparisons = {
            "Raw SWAT + Ridge": _ridge_baseline(prepared_train, prepared_test, target, raw_swat_features),
            "Raw SWAT + LSTM": _lstm_baseline(prepared_train, prepared_test, target, raw_swat_features),
            "Effective prior + Ridge": _ridge_baseline(prepared_train, prepared_test, target, effective_prior_features),
            "Effective prior + Ridge + hydraulics": _ridge_baseline(
                prepared_train, prepared_test, target, hydraulics_features
            ),
        }
        for benchmark_name, metrics in comparisons.items():
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "target": target.upper(),
                    f"rmse_{target}": metrics["rmse"],
                    f"mae_{target}": metrics["mae"],
                    f"r2_{target}": metrics["r2"],
                    f"nse_{target}": metrics["nse"],
                }
            )
    return pd.DataFrame(rows)


def _predict_target(
    models: TargetModels, frame: pd.DataFrame, target: str, include_site_dummies: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    design = _align_design_matrix(frame, target, models.design_columns, include_site_dummies=include_site_dummies)
    baseline = models.linear.predict(design)
    residual = models.residual.predict(design) if models.residual is not None else np.zeros(len(frame))
    unconstrained = baseline + residual

    contributions = pd.DataFrame(index=frame.index)
    coef = pd.Series(models.linear.coef_, index=models.design_columns)
    for group_name, columns in _group_columns(target).items():
        active = [col for col in models.design_columns if col in columns]
        if active:
            group_values = design[active].mul(coef[active], axis=1).sum(axis=1)
        else:
            group_values = pd.Series(0.0, index=frame.index)
        contributions[f"contribution_{group_name}_{target}"] = group_values
    total, projection_adjustment = _project_to_physical_domain(unconstrained, frame, target)
    contributions[f"contribution_physical_projection_{target}"] = projection_adjustment
    contributions[f"contribution_intercept_{target}"] = models.linear.intercept_
    return total, contributions


def _predict_scenario(
    models: TargetModels, frame: pd.DataFrame, target: str, scenario: str, include_site_dummies: bool = False
) -> np.ndarray:
    scenario_frame = _scenario_frame(frame, target, scenario)
    pred, _ = _predict_target(models, scenario_frame, target, include_site_dummies=include_site_dummies)
    return pred


def _event_scaled_intervals(
    frame: pd.DataFrame,
    target: str,
    central: pd.Series,
    conservative: pd.Series,
    responsive: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    event_flag = pd.Series(frame.get("event_flag", 0), index=frame.index, dtype=float).fillna(0.0)
    scenario_width = (responsive - conservative).abs()
    minimum_width = 0.06 * central.abs() + 0.04 * event_flag
    width = np.maximum(scenario_width, minimum_width) * (1.0 + 0.35 * event_flag)
    lower = (central - 0.5 * width).clip(lower=0.0)
    upper = central + 0.5 * width
    return lower, upper


def _summarize_split_metrics(frame: pd.DataFrame, split_name: str) -> list[dict[str, float | str]]:
    metric_rows: list[dict[str, float | str]] = []
    for site_name, site_frame in [("ALL", frame)] + list(frame.groupby("site")):
        row: dict[str, float | str] = {"split": split_name, "site": site_name}
        for target in ("tn", "tp"):
            actual = site_frame[f"target_{target}"]
            pred = site_frame[f"pred_{target}"]
            row[f"rmse_{target}"] = float(np.sqrt(mean_squared_error(actual, pred)))
            row[f"mae_{target}"] = float(mean_absolute_error(actual, pred))
            row[f"r2_{target}"] = float(r2_score(actual, pred))
            row[f"nse_{target}"] = _nash_sutcliffe_efficiency(actual, pred)
        metric_rows.append(row)
    return metric_rows


def _fit_site_target_models(train: pd.DataFrame, target: str) -> dict[str, TargetModels]:
    models: dict[str, TargetModels] = {}
    for site, site_frame in train.groupby("site"):
        base_model = _fit_target_models(site_frame.copy(), target, include_site_dummies=False)
        if site == "排水闸":
            design = _align_design_matrix(site_frame, target, base_model.design_columns, include_site_dummies=False)
            y = site_frame[f"target_{target}"].to_numpy()
            base_pred = base_model.linear.predict(design)
            residual_target = y - base_pred
            if target == "tn":
                site_residual = RandomForestRegressor(
                    n_estimators=120,
                    max_depth=4,
                    min_samples_leaf=6,
                    random_state=42,
                    n_jobs=1,
                )
                strategy = "drainage_gate_rf_residual"
            else:
                site_residual = HistGradientBoostingRegressor(
                    max_depth=2,
                    max_iter=120,
                    learning_rate=0.05,
                    min_samples_leaf=15,
                    l2_regularization=0.8,
                    random_state=42,
                )
                strategy = "drainage_gate_hgb_residual"
            site_residual.fit(design, residual_target)
            base_model.residual = site_residual
            base_model.residual_kind = strategy
            base_model.site_strategy = strategy
            projected_pred, _ = _project_to_physical_domain(base_pred + site_residual.predict(design), site_frame, target)
            base_model.residual_sigma = float(np.std(y - projected_pred))
        models[site] = base_model
    return models


def _predict_site_target(
    models: dict[str, TargetModels], frame: pd.DataFrame, target: str
) -> tuple[np.ndarray, pd.DataFrame]:
    preds = pd.Series(index=frame.index, dtype=float)
    conservative = pd.Series(index=frame.index, dtype=float)
    responsive = pd.Series(index=frame.index, dtype=float)
    contrib_frames = []
    linear_kind_series = pd.Series(index=frame.index, dtype=object)
    weighting_series = pd.Series(index=frame.index, dtype=object)
    residual_kind_series = pd.Series(index=frame.index, dtype=object)
    site_strategy_series = pd.Series(index=frame.index, dtype=object)
    for site, site_frame in frame.groupby("site"):
        model = models[site]
        site_pred, contrib = _predict_target(model, site_frame, target, include_site_dummies=False)
        preds.loc[site_frame.index] = site_pred
        conservative.loc[site_frame.index] = _predict_scenario(model, site_frame, target, "conservative", include_site_dummies=False)
        responsive.loc[site_frame.index] = _predict_scenario(model, site_frame, target, "responsive", include_site_dummies=False)
        linear_kind_series.loc[site_frame.index] = model.linear_kind
        weighting_series.loc[site_frame.index] = model.weight_scheme
        residual_kind_series.loc[site_frame.index] = model.residual_kind
        site_strategy_series.loc[site_frame.index] = model.site_strategy
        contrib_frames.append(contrib)
    contributions = pd.concat(contrib_frames).sort_index()
    interval_low, interval_high = _event_scaled_intervals(frame, target, preds, conservative, responsive)
    meta = pd.DataFrame(
        {
            f"pred_{target}_conservative": conservative,
            f"pred_{target}_central": preds,
            f"pred_{target}_responsive": responsive,
            f"prediction_interval_{target}_low": interval_low,
            f"prediction_interval_{target}_high": interval_high,
            f"selected_linear_{target}": linear_kind_series,
            f"selected_weighting_{target}": weighting_series,
            f"selected_residual_{target}": residual_kind_series,
            f"selected_site_strategy_{target}": site_strategy_series,
            f"selected_regime_{target}": "all",
        }
    ).sort_index()
    return preds.sort_index().to_numpy(), pd.concat([meta, contributions], axis=1)


def _predict_global_target(model: TargetModels, frame: pd.DataFrame, target: str) -> tuple[np.ndarray, pd.DataFrame]:
    pred, contrib = _predict_target(model, frame, target, include_site_dummies=True)
    conservative = _predict_scenario(model, frame, target, "conservative", include_site_dummies=True)
    responsive = _predict_scenario(model, frame, target, "responsive", include_site_dummies=True)
    interval_low, interval_high = _event_scaled_intervals(
        frame,
        target,
        pd.Series(pred, index=frame.index),
        pd.Series(conservative, index=frame.index),
        pd.Series(responsive, index=frame.index),
    )
    meta = pd.DataFrame(
        {
            f"pred_{target}_conservative": conservative,
            f"pred_{target}_central": pred,
            f"pred_{target}_responsive": responsive,
            f"prediction_interval_{target}_low": interval_low,
            f"prediction_interval_{target}_high": interval_high,
            f"selected_linear_{target}": model.linear_kind,
            f"selected_weighting_{target}": model.weight_scheme,
            f"selected_residual_{target}": model.residual_kind,
            f"selected_site_strategy_{target}": model.site_strategy,
            f"selected_regime_{target}": "global",
        },
        index=frame.index,
    )
    return pred, pd.concat([meta, contrib], axis=1)


def fit_predictive_system(dataset: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = dataset.copy().sort_values(["site", "date"]).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "event_flag" not in frame.columns:
        frame["event_flag"] = 0
    else:
        frame["event_flag"] = frame["event_flag"].fillna(0).astype(int)
    split_ts = pd.Timestamp(split_date)
    train = frame[frame["date"] < split_ts].copy()
    test = frame[frame["date"] >= split_ts].copy()
    if train.empty or test.empty:
        raise ValueError("Training and test splits must both be non-empty.")

    tn_models = _fit_site_target_models(train, "tn")
    tp_model = _fit_target_models(train, "tp", include_site_dummies=True)

    base_columns = ["date", "site", "target_tn", "target_tp", "event_flag"]
    outputs = test.loc[:, [col for col in base_columns if col in test.columns]].copy()
    train_outputs = train.loc[:, [col for col in base_columns if col in train.columns]].copy()

    train_pred_tn, train_meta_tn = _predict_site_target(tn_models, train, "tn")
    test_pred_tn, test_meta_tn = _predict_site_target(tn_models, test, "tn")
    train_outputs["pred_tn"] = train_pred_tn
    outputs["pred_tn"] = test_pred_tn
    for column in train_meta_tn.columns:
        train_outputs[column] = train_meta_tn[column]
    for column in test_meta_tn.columns:
        outputs[column] = test_meta_tn[column]

    train_pred_tp, train_meta_tp = _predict_global_target(tp_model, train, "tp")
    test_pred_tp, test_meta_tp = _predict_global_target(tp_model, test, "tp")
    train_outputs["pred_tp"] = train_pred_tp
    outputs["pred_tp"] = test_pred_tp
    for column in train_meta_tp.columns:
        train_outputs[column] = train_meta_tp[column]
    for column in test_meta_tp.columns:
        outputs[column] = test_meta_tp[column]

    metric_rows = _summarize_split_metrics(train_outputs, "train")
    metric_rows.extend(_summarize_split_metrics(outputs, "test"))
    return outputs, pd.DataFrame(metric_rows)
