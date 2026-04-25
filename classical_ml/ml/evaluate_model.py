import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

    print("=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"MAE  : {mae:.6e}")
    print(f"RMSE : {rmse:.6e}")
    print(f"R²   : {r2:.6f}")
    print(f"MAPE : {mape:.4f} %")
    print("=" * 50)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


if __name__ == "__main__":

    # Load trained model
    model_path = os.path.join("data", "models", "gbr_model.pkl")
    scaler_path = os.path.join("data", "models", "scaler.pkl")
    if not os.path.exists(model_path):
        print("Model file not found. Run training first.")
        exit(1)

    model = load(model_path)

    # Find data file
    possible = [
        os.path.join("data", "dispersion_full_dataset.xlsx"),
        os.path.join("data", "raw", "dispersion_full_dataset.xlsx"),
        os.path.join("data", "dispersion_full_dataset.csv"),
        os.path.join("data", "raw", "dispersion_full_dataset.csv"),
    ]
    data_path = next((p for p in possible if os.path.exists(p)), None)
    if data_path is None:
        print("dispersion_full_dataset data file not found in data/ or data/raw/")
        exit(1)

    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    else:
        data = pd.read_excel(data_path)

    # Inputs: kL and L
    X = data[["kL", "L", "alpha1", "s", "P1", "P2"]].values

    # Output: c/beta_l
    y = data["c_beta"].values

    # Load scaler if available, otherwise fit a new one
    if os.path.exists(scaler_path):
        scaler = load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Evaluate
    metrics = evaluate(model, X_test, y_test)

    print("Evaluation completed successfully")