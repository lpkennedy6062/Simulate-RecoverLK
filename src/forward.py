import numpy as np

def compute_forward_stats(speed, boundary, delay):
    """Compute predicted summary statistics from diffusion parameters."""
    exponent = np.exp(-boundary * speed)
    response_rate = 1 / (1 + exponent)
    mean_time = delay + (boundary ** 2 / speed) * ((1 - exponent) / (1 + exponent))
    variance_time = (boundary ** 2 / speed ** 3) * ((1 - 2 * boundary * speed * exponent - exponent ** 2) / (1 + exponent) ** 2)
    variance_time = np.clip(variance_time, 1e-6, 1)
    
    return response_rate, mean_time, variance_time

def generate_noisy_stats(resp_pred, mean_pred, var_pred, samples):
    """Generate noisy observed statistics."""
    trial_count = np.random.binomial(samples, resp_pred)
    observed_resp = trial_count / samples
    observed_mean = np.random.normal(mean_pred, np.sqrt(var_pred / samples))
    
    shape_param = (samples - 1) / 2
    scale_param = (2 * var_pred) / (samples - 1)
    observed_var = np.random.gamma(shape_param, scale_param)
    
    return observed_resp, observed_mean, observed_var
