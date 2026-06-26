from __future__ import annotations

import numpy as np


def assert_close(name: str, values: np.ndarray, expected: float = 1.0, tol: float = 1e-9) -> None:
    if not np.allclose(values, expected, atol=tol, rtol=0.0):
        raise AssertionError(f"{name} failed: expected {expected}, got {values}")


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    rng = np.random.default_rng(42)
    n_days = 12
    n_sources = 5
    n_states = 4
    n_lags = 8
    n_branches = 3

    area = rng.uniform(10.0, 200.0, size=n_sources)
    organic_n = rng.uniform(0.0, 5.0, size=(n_days, n_sources))
    nitrate_n = rng.uniform(0.0, 3.0, size=(n_days, n_sources))
    source_input = area.reshape(1, -1) * (organic_n + nitrate_n)
    reach_input = rng.uniform(0.0, 500.0, size=(n_days, 3))
    l_all = source_input.sum(axis=1) + reach_input.sum(axis=1)
    l_key = source_input[:, :3].sum(axis=1) + reach_input[:, :2].sum(axis=1)

    corr_scores = rng.uniform(0.0, 1.0, size=n_sources + 3)
    corr_weights = corr_scores / corr_scores.sum()
    source_matrix = np.column_stack([source_input, reach_input])
    l_corr = source_matrix @ corr_weights
    assert_close("SWAT correlation weights", corr_weights.sum())
    if np.any(l_all < 0) or np.any(l_key < 0) or np.any(l_corr < 0):
        raise AssertionError("SWAT prior candidates must be non-negative load-like quantities.")

    state_logits = rng.normal(size=(n_days, n_states))
    state_prob = softmax(state_logits)
    assert_close("state probabilities", state_prob.sum(axis=1))

    kernels = softmax(rng.normal(size=(n_states, n_lags)))
    assert_close("state lag kernels", kernels.sum(axis=1))

    gamma = np.array([0.85, 1.0, 1.25, 0.75])
    connectivity = rng.uniform(0.7, 1.3, size=n_days)
    l0 = l_corr
    prior_history = np.zeros((n_days, n_lags))
    for t in range(n_days):
        for lag in range(n_lags):
            prior_history[t, lag] = l0[max(t - lag, 0)]
    state_lagged = prior_history @ kernels.T
    l_align = np.sum(state_prob * (gamma.reshape(1, -1) * connectivity.reshape(-1, 1) * state_lagged), axis=1)
    if np.any(l_align < 0) or l_align.shape != (n_days,):
        raise AssertionError("L_align must be a non-negative daily vector.")

    y_raw = rng.uniform(0.5, 2.5, size=n_days)
    y_proxy = rng.uniform(0.5, 2.5, size=n_days)
    beta = 0.35
    y_prior = beta * y_raw + (1.0 - beta) * y_proxy
    lower = np.minimum(y_raw, y_proxy)
    upper = np.maximum(y_raw, y_proxy)
    if np.any(y_prior < lower - 1e-12) or np.any(y_prior > upper + 1e-12):
        raise AssertionError("Calibrated prior must be a convex combination of raw and proxy predictions.")

    branch_residuals = rng.normal(loc=0.0, scale=0.08, size=(n_days, n_branches))
    alpha = softmax(rng.normal(size=(n_days, n_branches)))
    assert_close("residual branch weights", alpha.sum(axis=1))
    residual = np.sum(alpha * branch_residuals, axis=1)
    y_hat = y_prior + residual
    if residual.shape != y_prior.shape or y_hat.shape != y_prior.shape:
        raise AssertionError("Residual fusion and final prediction must preserve daily prediction shape.")

    pred_loss = float(np.mean((y_hat - rng.uniform(0.5, 2.5, size=n_days)) ** 2))
    residual_mag = float(np.mean(residual**2))
    event_relax = float(np.mean(np.maximum(np.abs(residual) - 0.2, 0.0) ** 2))
    consistency = float(np.mean((alpha[:, 1] - alpha[:, 2]) ** 2))
    lambdas = np.array([0.1, 0.05, 0.05])
    total_loss = pred_loss + lambdas[0] * residual_mag + lambdas[1] * event_relax + lambdas[2] * consistency
    if total_loss < 0:
        raise AssertionError("Total loss must be non-negative when all components and lambdas are non-negative.")

    print("Formula sanity check passed.")
    print(f"- SWAT candidates: L_all/L_key/L_corr non-negative, weight sum = {corr_weights.sum():.6f}")
    print(f"- L_align: shape={l_align.shape}, min={l_align.min():.6f}, max={l_align.max():.6f}")
    print(f"- Prior calibration: beta={beta:.2f}, convex combination verified")
    print(f"- Residual fusion: branch weights sum to {alpha.sum(axis=1).mean():.6f} on average")
    print(f"- Loss: non-negative total objective = {total_loss:.6f}")


if __name__ == "__main__":
    main()
