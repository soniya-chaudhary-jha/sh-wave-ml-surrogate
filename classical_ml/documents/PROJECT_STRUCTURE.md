# Project Structure & Execution Flow

## 📂 Directory Structure

```
sh-wave-ml-surrogate/
├── physics/                    # Physics simulation modules
│   ├── dispersion_solver.py   # Solves wave equations
│   ├── material_constants.py  # Material properties
│   └── parameter_sampler.py   # Random parameter generation
│
├── ml/                         # Machine learning modules
│   ├── models.py              # Model definitions (GBR, RF)
│   ├── train_surrogate.py     # Training script
│   └── evaluate_model.py      # Evaluation metrics
│
├── notebooks/                  # Jupyter analysis notebooks
│   ├── 01_data_generation.ipynb      # Data creation
│   ├── 02_training.ipynb             # Model training
│   ├── 03_results_visualization.ipynb # Visual results
│   └── 04_comprehensive_evaluation.ipynb (NEW) ⭐
│
├── data/                       # Data storage
│   ├── raw/                   # Generated datasets & models
│   │   ├── analytical_results.csv
│   │   └── model.pkl
│   └── processed/             # Processed datasets
│
├── results/                    # Output results
│   └── plots/                 # Generated visualizations
│
├── tests/                      # Unit tests
├── main.py                     # Entry point
├── pyproject.toml             # Project config
└── EVALUATION_GUIDE.md        # This evaluation guide
```

## 🔄 Execution Flow

```
1. GENERATE DATA
   └─> python ml/train_surrogate.py
       ├─ Sample 6000 random parameters
       ├─ Compute phase velocities (physics)
       └─ Save: data/raw/analytical_results.csv + data/raw/model.pkl

2. EVALUATE MODEL
   └─> python ml/evaluate_model.py
       ├─ Load trained model
       ├─ Calculate MAE, RMSE, R², MAPE
       └─ Display metrics

3. VISUALIZE & ANALYZE
   └─> jupyter lab
       ├─ Open notebooks/04_comprehensive_evaluation.ipynb
       ├─ Run all cells for:
       │  ├─ Predictions vs Actual plot
       │  ├─ Residual analysis (4 plots)
       │  ├─ Feature importance
       │  └─ Error statistics
       └─ Save plots to results/plots/
```

## ✅ Notebook Files Summary

| Notebook | Purpose | Run Time | Output |
|----------|---------|----------|--------|
| 01_data_generation | Create training data | 2-3 min | 2000 samples |
| 02_training | Train ML model | 30 sec | R² score |
| 03_results_visualization | Quick plots | 10 sec | 1 plot |
| **04_comprehensive_evaluation** | **Full diagnosis** | **2-5 min** | **6+ plots + metrics** |

## 🎯 Why No Confusion Matrix?

**This is REGRESSION, NOT classification!**

Confusion matrices are for:
- ❌ Classification (discrete categories)
- ❌ Binary/multiclass problems

This project uses:
- ✅ Regression (continuous output: phase velocity)
- ✅ Scatter plots for accuracy visualization
- ✅ Residuals for error analysis
- ✅ R² for explained variance

