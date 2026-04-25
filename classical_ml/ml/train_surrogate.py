import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml.models import get_gbr
from joblib import dump
import os


def find_data_path(root):
    candidates = [
        os.path.join(root, "data", "dispersion_full_dataset.xlsx"),
        os.path.join(root, "data", "raw", "dispersion_full_dataset.xlsx"),
        os.path.join(root, "data", "dispersion_full_dataset.csv"),
        os.path.join(root, "data", "raw", "dispersion_full_dataset.csv"),
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = find_data_path(ROOT)
if data_path is None:
    raise FileNotFoundError("Could not find dispersion_full_dataset data file in data/ or data/raw/")

# Load
if data_path.endswith('.csv'):
    data = pd.read_csv(data_path)
else:
    data = pd.read_excel(data_path)

# Inputs: (kL, L)
X = data[["kL", "L", "alpha1", "s", "P1", "P2"]].values

# Output: c / beta_l
y = data["c_beta"].values

# -----------------------------
# STEP 1: Train-test split FIRST
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 2: Scaling AFTER split
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train model
model = get_gbr()
model.fit(X_train, y_train)

# Save model + scaler
model_dir = os.path.join(ROOT, "data", "models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "gbr_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

dump(model, model_path)
dump(scaler, scaler_path)

print("Training completed successfully.")