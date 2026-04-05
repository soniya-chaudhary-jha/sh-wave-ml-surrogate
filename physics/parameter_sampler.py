import os

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


# Main function
def main():
    samples = sample_parameters(n_samples=1000)

    # Save file
    save_path = "data/parameter_samples.csv"
    os.makedirs("data", exist_ok=True)
    np.savetxt(save_path, samples, delimiter=",", header="k,H,alpha,eta", comments="")

    print(f"Data saved at: {save_path}")


# Run the program
if __name__ == "__main__":
    main()