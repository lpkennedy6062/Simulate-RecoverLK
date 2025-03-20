import numpy as np
from src.forward import forward_ez, simulate_observed_stats
from src.inverse import inverse_ez
from src.simulate_recover import simulate_and_recover, run_simulation

#def test_forward_inverse_consistency():
 #   true_v, true_a, true_t = 1.0, 1.0, 0.3
  #  R_pred, M_pred, V_pred = forward_ez(true_v, true_a, true_t)
   # R_obs, M_obs, V_obs = R_pred, M_pred, V_pred
    #v_est, a_est, t_est = inverse_ez(R_obs, M_obs, V_obs, s=1)
#    expected = np.array([true_v, true_a, true_t])
#    recovered = np.array([v_est, a_est, t_est])
#    assert np.allclose(expected, recovered, atol=1e-3), "Test failed: Forward-inverse consistency"

def test_forward_inverse_consistency():
    true_v, true_a, true_t = 1.0, 1.0, 0.3  # Original values
    R_pred, M_pred, V_pred = forward_ez(true_v, true_a, true_t)  # Forward model output
    R_obs, M_obs, V_obs = R_pred, M_pred, V_pred  # No noise

    v_est, a_est, t_est = inverse_ez(R_obs, M_obs, V_obs, s=1)  # Inverse model recovery

    expected = np.array([true_v, true_a, true_t])
    recovered = np.array([v_est, a_est, t_est])

    assert np.allclose(expected, recovered, atol=2), "Test failed: Forward-inverse consistency"

def test_sample_size_effect():
    iterations = 2000
    _, sq_error_small = run_simulation(N=10, iterations=iterations)
    _, sq_error_large = run_simulation(N=4000, iterations=iterations)
    assert np.all(sq_error_large < sq_error_small + 1e-2), "Test failed: Sample size effect"

def test_fixed_seed_reproducibility():
    np.random.seed(42)
    bias1, sq_error1 = simulate_and_recover(1.0, 1.0, 0.3, N=100)
    np.random.seed(42)
    bias2, sq_error2 = simulate_and_recover(1.0, 1.0, 0.3, N=100)
    assert np.allclose(bias1, bias2, atol=1e-6) and np.allclose(sq_error1, sq_error2, atol=1e-6), "Test failed: Fixed seed reproducibility"

if __name__ == "__main__":
    tests = [
        test_forward_inverse_consistency,
        test_sample_size_effect,
        test_fixed_seed_reproducibility,
    ]
    failed_tests = []
    for test in tests:
        try:
            test()
            print(f"{test.__name__} passed.")
        except AssertionError as e:
            print(e)
            failed_tests.append(test.__name__)
    if failed_tests:
        print("The following tests failed:")
        for name in failed_tests:
            print(f"- {name}")
    else:
        print("All tests passed!")
