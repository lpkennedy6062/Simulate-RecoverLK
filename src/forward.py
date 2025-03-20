import numpy as np

def forward_ez(v, a, t):
    """Compute predicted summary statistics from EZ diffusion parameters."""
    y = np.exp(-a * v)
    R_pred = 1 / (1 + y)
    M_pred = t + (a ** 2 / v) * ((1 - y) / (1 + y))
    V_pred = (a ** 2 / v ** 3) * ((1 - 2 * a * v * y - y ** 2) / (1 + y) ** 2)
    V_pred = np.clip(V_pred, 1e-6, 1)
    return R_pred, M_pred, V_pred

def simulate_observed_stats(R_pred, M_pred, V_pred, N):
    """Generate noisy observed summary statistics."""
    T_obs = np.random.binomial(N, R_pred)
    R_obs = T_obs / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    shape = (N - 1) / 2
    scale = (2 * V_pred) / (N - 1)
    V_obs = np.random.gamma(shape, scale)
    return R_obs, M_obs, V_obs
