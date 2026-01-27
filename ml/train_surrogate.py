import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from physics.parameter_sampler import sample_parameters
from physics.dispersion_solver import solve_phase_velocity
from ml.models import get_gbr
import pickle

# Generate data
X = sample_parameters(n_samples=6000)

y = np.array([
    solve_phase_velocity(k, H, a, e) 
    for k, H, a, e in X
])

mask = ~np.isnan(y)
X, y = X[mask], y[mask]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train
model = get_gbr()
model.fit(X_train, y_train)

# Save
pd.DataFrame(X).to_csv("data/raw/analytical_results.csv", index=False)
with open("data/raw/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training completed.")
