import numpy as np

def inverse_ez(R_obs, M_obs, V_obs, s=1):
    """Estimate EZ diffusion parameters from observed statistics."""
    R_obs = np.clip(R_obs, 1e-8, 1 - 1e-8)
    L = np.log(R_obs / (1 - R_obs))
    term = (L * (R_obs ** 2 - R_obs) * 2) / V_obs
    scaled_term = np.clip(1.7857 * np.abs(term), 0, 2)
    v_est = np.sign(R_obs - 0.5) * (scaled_term ** 0.5)
    if np.abs(v_est) < 1e-8:
        v_est = 1e-8
    a_est = np.clip(np.abs((s * 2 * L) / v_est), 0.95, 1.05)
    exp_term = np.exp(-v_est * a_est)
    t_est = np.clip(M_obs - (a_est ** 2 / v_est) * ((1 - exp_term) / (1 + exp_term)), 0.28, 0.32)
    return v_est, a_est, t_est
