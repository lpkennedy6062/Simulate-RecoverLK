import numpy as np
from forward import compute_forward_stats, generate_noisy_stats
from inverse import compute_inverse_params

def simulate_and_recover(true_speed, true_boundary, true_delay, sample_count):
    """Simulate data and recover parameters."""
    resp_pred, mean_pred, var_pred = compute_forward_stats(true_speed, true_boundary, true_delay)
    obs_resp, obs_mean, obs_var = generate_noisy_stats(resp_pred, mean_pred, var_pred, sample_count)
    
    est_speed, est_boundary, est_delay = compute_inverse_params(obs_resp, obs_mean, obs_var)
    
    bias = np.array([true_speed, true_boundary, true_delay]) - np.array([est_speed, est_boundary, est_delay])
    sq_error = bias ** 2
    return bias, sq_error

def run_recovery_experiment(sample_sizes, iterations=1000):
    """Run the full simulate-and-recover process for different sample sizes."""
    biases = []
    sq_errors = []
    
    for _ in range(iterations):
        rand_boundary = np.random.uniform(0.5, 2)
        rand_speed = np.random.uniform(0.5, 2)
        rand_delay = np.random.uniform(0.1, 0.5)
        
        bias, sq_error = simulate_and_recover(rand_speed, rand_boundary, rand_delay, sample_sizes)
        biases.append(bias)
        sq_errors.append(sq_error)
    
    mean_bias = np.mean(biases, axis=0)
    mean_sq_error = np.mean(sq_errors, axis=0)
    return mean_bias, mean_sq_error

def main():
    trials = [10, 40, 4000]
    iterations = 1000
    
    for trial in trials:
        mean_bias, mean_sq_error = run_recovery_experiment(trial, iterations)
        print(f"Trial Count: {trial}")
        print("Mean Bias [speed, boundary, delay]:", mean_bias)
        print("Mean Squared Error [speed, boundary, delay]:", mean_sq_error)
        print("----------------------------")

if __name__ == "__main__":
    print("Executing simulation and recovery process...")
    main()
    print("Process completed!")
