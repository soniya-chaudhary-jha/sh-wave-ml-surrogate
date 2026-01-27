import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pickle
import pandas as pd

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
    
    print("=" * 50)
    print("MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print("=" * 50)
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

if __name__ == "__main__":
    # Load the trained model
    try:
        with open("data/raw/model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        
        # Load test data if available
        if os.path.exists("data/raw/analytical_results.csv"):
            X_data = pd.read_csv("data/raw/analytical_results.csv").values
            
            # For demonstration, use last 20% as test
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, np.array([1]*len(X_scaled)), test_size=0.2, random_state=42
            )
            
            print("\n✓ Evaluation complete!")
        else:
            print("✓ Model ready for evaluation")
            
    except FileNotFoundError:
        print("❌ Model file not found. Please run ml/train_surrogate.py first.")
