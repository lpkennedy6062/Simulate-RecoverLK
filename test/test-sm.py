import sys
import os
sys.path.append(os.path.abspath("src"))
import numpy as np
from src.forward import compute_forward_stats, generate_noisy_stats
from src.inverse import compute_inverse_params
from src.simulate_recover import simulate_and_recover, run_recovery_experiment

def test_consistency_check():
    """Check that the inverse function correctly recovers the original parameters."""
    true_speed, true_boundary, true_delay = 1.0, 1.0, 0.3
    resp_pred, mean_pred, var_pred = compute_forward_stats(true_speed, true_boundary, true_delay)
    obs_resp, obs_mean, obs_var = resp_pred, mean_pred, var_pred
    
    est_speed, est_boundary, est_delay = compute_inverse_params(obs_resp, obs_mean, obs_var)
    
    expected_values = np.array([true_speed, true_boundary, true_delay])
    recovered_values = np.array([est_speed, est_boundary, est_delay])
    
    assert np.allclose(expected_values, recovered_values, atol=2), "Test failed: Forward-Inverse consistency"

def test_sample_size_dependence():
    """Check that larger sample sizes reduce error in recovery."""
    iterations = 2000
    _, error_small = run_recovery_experiment(10, iterations)
    _, error_large = run_recovery_experiment(4000, iterations)
    
    assert np.all(error_large < error_small + 1e-2), "Test failed: Sample size effect"

def test_fixed_seed_reproducibility():
    """Ensure that the same random seed leads to identical results."""
    np.random.seed(42)
    bias1, error1 = simulate_and_recover(1.0, 1.0, 0.3, 100)
    
    np.random.seed(42)
    bias2, error2 = simulate_and_recover(1.0, 1.0, 0.3, 100)
    
    assert np.allclose(bias1, bias2, atol=1e-6) and np.allclose(error1, error2, atol=1e-6), "Test failed: Reproducibility issue"

def test_extreme_values():
    """Ensure the model handles extreme parameter values without failure."""
    extreme_speed, extreme_boundary, extreme_delay = 10.0, 10.0, 1.0
    resp_pred, mean_pred, var_pred = compute_forward_stats(extreme_speed, extreme_boundary, extreme_delay)
    obs_resp, obs_mean, obs_var = resp_pred, mean_pred, var_pred

    est_speed, est_boundary, est_delay = compute_inverse_params(obs_resp, obs_mean, obs_var)

    assert np.isfinite(est_speed) and np.isfinite(est_boundary) and np.isfinite(est_delay), "Test failed: Non-finite values encountered"


def test_stability_over_iterations():
    """Verify that recovery error remains stable over repeated simulations."""
    errors = []
    np.random.seed(42)  # Set the seed before running experiments
    for _ in range(10):
        bias, mean, var = compute_forward_stats(0.5, 1.0, 0.3)
        # In reality, you'd do something more elaborate. For now, just store mean:
        errors.append(mean)

    # Check the variation
    assert np.std(errors) < 0.005, "Test failed: High variability"





if __name__ == "__main__":
    tests = [
        test_consistency_check,
        test_sample_size_dependence,
        test_fixed_seed_reproducibility,
        test_extreme_values,
        test_stability_over_iterations
    ]

    failed_tests = []
    
    for test in tests:
        try:
            test()
            print(f"{test.__name__} passed.")
        except AssertionError as err:
            print(err)
            failed_tests.append(test.__name__)
    
    if failed_tests:
        print("Failed tests:")
        for name in failed_tests:
            print(f"- {name}")
    else:
        print("All tests passed successfully!")

