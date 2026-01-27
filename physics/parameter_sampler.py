import numpy as np

def sample_parameters(
    k_range=(0.01, 5),
    H_range=(0.1, 5),
    alpha_range=(0, 2),
    eta_range=(0, 15),
    n_samples=5000
):
    k = np.random.uniform(*k_range, n_samples)
    H = np.random.uniform(*H_range, n_samples)
    alpha = np.random.uniform(*alpha_range, n_samples)
    eta = np.random.uniform(*eta_range, n_samples)

    return np.column_stack([k, H, alpha, eta])
