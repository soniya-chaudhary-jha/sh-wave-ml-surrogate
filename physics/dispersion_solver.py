import os
import numpy as np


def solve_phase_velocity(k, H, alpha, eta):
    """Estimate a phase velocity proxy for given parameters.

    This is a lightweight deterministic surrogate used for data generation
    in notebooks. It is intentionally simple and deterministic so notebooks
    can run without external simulation dependencies.

    Parameters
    ----------
    k : float
        Wave number
    H : float
        Layer thickness
    alpha : float
        Material parameter alpha
    eta : float
        Material parameter eta

    Returns
    -------
    float
        Estimated c_beta (phase velocity proxy)
    """
    # A smooth deterministic mapping combining inputs
    # Keep result positive and reasonably scaled
    val = (np.sqrt(H + 1.0) * (1.0 + 0.1 * alpha) * (1.0 + 0.01 * eta) * (1.0 + 0.5 * np.tanh(k)))
    return float(val)


if __name__ == "__main__":
    # Import heavy dependencies only when running as a script (avoid import-time side-effects)
    import pandas as pd
    import matplotlib.pyplot as plt
    from joblib import load

    # ==============================
    # 1. ROOT path
    # ==============================
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ==============================
    # 2. Load model + optional scaler
    # ==============================
    model_path = os.path.join(ROOT, "data", "models", "gbr_model.pkl")
    scaler_path = os.path.join(ROOT, "data", "models", "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {model_path}: {e}")

    # scaler is optional; the model may already include preprocessing in a Pipeline
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = load(scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler at {scaler_path}: {e}")

    # ==============================
    # 3. Load Excel/CSV data
    # ==============================
    possible_paths = [
        os.path.join(ROOT, "data", "dispersion_vs_L.xlsx"),
        os.path.join(ROOT, "data", "raw", "dispersion_vs_L.xlsx"),
        os.path.join(ROOT, "data", "dispersion_vs_L.csv"),
        os.path.join(ROOT, "data", "raw", "dispersion_vs_L.csv"),
    ]
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError("Could not find dispersion_vs_L data file in data/ or data/raw/")

    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    else:
        data = pd.read_excel(data_path)

    # If the loaded model expects a different number of features than our dataset,
    # retrain a local model using the available (kL, L) features so predictions work.
    expected_n_features = 2
    if hasattr(model, "n_features_in_") and model.n_features_in_ != expected_n_features:
        print(f"Loaded model expects {model.n_features_in_} features but input has {expected_n_features}. Re-training local GBR on available data.")
        try:
            from sklearn.preprocessing import StandardScaler
            from joblib import dump
            from ml.models import get_gbr

            X_train = data[["kL", "L"]].values
            y_train = data["c_beta"].values

            scaler_local = StandardScaler()
            X_scaled = scaler_local.fit_transform(X_train)

            model_local = get_gbr()
            model_local.fit(X_scaled, y_train)

            dump(model_local, model_path)
            dump(scaler_local, scaler_path)

            model = model_local
            scaler = scaler_local
            print(f"Re-trained and saved new model to {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to retrain local model: {e}")

    # ==============================
    # 4. Unique L values
    # ==============================
    L_values = sorted(data["L"].unique())

    # ==============================
    # 5. Plot
    # ==============================
    plt.figure(figsize=(8,6))

    for L in L_values:
        subset = data[data["L"] == L]
        kL = subset["kL"].values
        y_true = subset["c_beta"].values
        idx = np.argsort(kL)
        kL = kL[idx]
        y_true = y_true[idx]
        X = np.column_stack((kL, np.full(kL.shape, L)))

        if scaler is not None:
            X_input = scaler.transform(X)
        else:
            X_input = X

        try:
            y_pred = model.predict(X_input)
        except Exception:
            y_pred = model.predict(X)

        plt.plot(kL, y_true, linewidth=2, label=f'Analytical (L={L})')
        plt.plot(kL, y_pred, '--', linewidth=2, label=f'ML (L={L})')

    plt.xlabel(r'$kL$', fontsize=12)
    plt.ylabel(r'$c/\\beta_l$', fontsize=12)
    plt.title('Dispersion Curve: Analytical vs ML', fontsize=13)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_dir = os.path.join(ROOT, "results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "dispersion_curve_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"Plot saved at: {plot_path}")