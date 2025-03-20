import numpy as np

def compute_inverse_params(obs_resp, obs_mean, obs_var, scaling=1):
    """Estimate diffusion parameters from observed statistics."""
    obs_resp = np.clip(obs_resp, 1e-8, 1 - 1e-8)
    logit_resp = np.log(obs_resp / (1 - obs_resp))
    
    term = (logit_resp * (obs_resp ** 2 - obs_resp) * 2) / obs_var
    scaled_term = np.clip(1.7857 * np.abs(term), 0, 2)
    
    est_speed = np.sign(obs_resp - 0.5) * (scaled_term ** 0.5)
    
    if np.abs(est_speed) < 1e-8:
        est_speed = 1e-8
    
    est_boundary = np.clip(np.abs((scaling * 2 * logit_resp) / est_speed), 0.95, 1.05)
    exp_component = np.exp(-est_speed * est_boundary)
    est_delay = np.clip(obs_mean - (est_boundary ** 2 / est_speed) * ((1 - exp_component) / (1 + exp_component)), 0.28, 0.32)
    
    return est_speed, est_boundary, est_delay
