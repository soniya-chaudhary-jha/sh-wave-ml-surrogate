# SH Wave ML Surrogate - Quick Reference Summary

## Problem Statement
**Develop an ML surrogate model to replace expensive analytical dispersion solving for SH wave propagation prediction in orthotropic elastic layers bonded to viscoelastic half-spaces.**

---

## Core Information at a Glance

| Category | Details |
|----------|---------|
| **Project Name** | SH Wave ML Surrogate |
| **Domain** | Physics-informed Machine Learning (Wave Propagation) |
| **Physics Domain** | SH-wave propagation in composite materials |
| **Main Objective** | Fast phase velocity prediction (1000x speedup vs. physics solver) |
| **Model Type** | Gradient Boosting Regressor (primary), Random Forest (baseline) |
| **Framework** | scikit-learn |
| **Language** | Python 3.12+ |

---

## Dataset Summary

| Attribute | Value |
|-----------|-------|
| **Source** | Analytically generated (physics-based) |
| **Total Samples** | 6,000 parameter sets |
| **Valid Samples** | ~5,800 (after NaN filtering) |
| **Training Samples** | ~4,560 (80%) |
| **Test Samples** | ~1,140 (20%) |
| **Data Format** | CSV (parameters) + Pickle (model) |
| **CSV Location** | `data/raw/analytical_results.csv` |
| **Model Location** | `data/raw/model.pkl` |

---

## Input Features (Complete List)

| # | Feature | Symbol | Range | Units | Physical Meaning |
|---|---------|--------|-------|-------|-----------------|
| 1 | Wave Number | k | 0.01 - 5.0 | dimensionless | Spatial frequency of wave |
| 2 | Layer Thickness | H | 0.1 - 5.0 | meters | Elastic layer thickness |
| 3 | Heterogeneity | α | 0.0 - 2.0 | dimensionless | Material variation parameter |
| 4 | Damping | η | 0.0 - 15.0 | dimensionless | Viscoelastic dissipation |

**Output Feature:**
| Feature | Symbol | Physical Meaning |
|---------|--------|-----------------|
| Phase Velocity | c | SH-wave propagation speed |

---

## Feature Engineering & Preprocessing

| Step | Method | Location |
|------|--------|----------|
| 1. Sampling | Uniform random within ranges | `physics/parameter_sampler.py` |
| 2. Physics | Dispersion solver (fsolve) | `physics/dispersion_solver.py` |
| 3. Cleaning | Remove NaN solutions | `ml/train_surrogate.py` |
| 4. Scaling | StandardScaler (zero mean, unit variance) | `ml/train_surrogate.py` |
| 5. Splitting | Train/Test 80/20 split (random_state=42) | `ml/train_surrogate.py` |

---

## Dispersion Equation

```
tan(k·H·√((c/β₁)² - 1)) = (C44_ve/C44)·(1 + damping + hetero)
```

**Material Constants:**
- C66 = 3.99e10 Pa (orthotropic elastic constant)
- C44 = 5.82e10 Pa (elastic shear modulus)
- C44_ve = 6.34e10 Pa (viscoelastic shear modulus)
- ρ₁ = 4500 kg/m³ (elastic layer density)
- ρ₂ = 3364 kg/m³ (viscoelastic half-space density)

---

## Model Architecture

### Primary Model: Gradient Boosting Regressor

```python
GradientBoostingRegressor(
    n_estimators=300,        # 300 trees
    learning_rate=0.05,      # 5% shrinkage
    max_depth=4,             # Shallow trees
    loss='squared_error'
)
```

### Baseline Model: Random Forest Regressor
```python
RandomForestRegressor(n_estimators=300, n_jobs=-1)
```

**Model Benefits:**
- Non-linear relationship capture
- Feature importance scores
- Fast inference (~1 ms/sample)
- Robust to outliers
- ~1000x speedup vs. physics solver

---

## Training Summary

| Phase | Details |
|-------|---------|
| **Location** | `ml/train_surrogate.py` |
| **Command** | `uv run python ml/train_surrogate.py` |
| **Duration** | ~30 seconds |
| **Input Data** | ~5,800 scaled samples |
| **Outputs** | `model.pkl` + `analytical_results.csv` |
| **Optimization** | Gradient boosting with MSE loss |
| **Regularization** | Learning rate shrinkage + tree depth constraint |

---

## Evaluation Metrics

### Primary Metrics (4-metric evaluation)

| Metric | Formula | Ideal Range | Interpretation |
|--------|---------|-------------|-----------------|
| **R² Score** | 1 - SS_res/SS_tot | 0.85 - 1.0 | Variance explained (%) |
| **MAE** | Mean(\|actual - pred\|) | Low | Avg absolute error |
| **RMSE** | √(Mean((actual-pred)²)) | Low | Emphasizes large errors |
| **MAPE** | Mean(\|error/actual\|) × 100% | < 5% | Percentage error |

### R² Score Interpretation
```
R² > 0.95  →  EXCELLENT ✓  (Production ready)
R² 0.85-0.95  →  VERY GOOD ✓ (Most applications)
R² 0.70-0.85  →  GOOD ✓      (Use with caution)
R² < 0.70  →  FAIR ⚠        (Needs improvement)
```

---

## Evaluation Notebooks

| # | Notebook | Purpose | Output |
|---|----------|---------|--------|
| 1 | `01_data_generation.ipynb` | Data creation demo | Raw data + targets |
| 2 | `02_training.ipynb` | Training walkthrough | R² score |
| 3 | `03_results_visualization.ipynb` | Quick plot | Predictions vs actual |
| 4 | `04_comprehensive_evaluation.ipynb` | Full diagnosis | 6+ plots + metrics |

---

## Comprehensive Evaluation (Notebook 4) Outputs

### Visualizations Generated
1. **Predictions vs Actual** - Scatter plot with perfect fit line
2. **Residual Analysis** - 4 subplots:
   - Residuals vs predicted values
   - Residuals histogram (distribution)
   - Q-Q plot (normality check)
   - Errors vs wave number
3. **Feature Importance** - Bar chart of feature rankings
4. **Error Statistics** - Metrics stratified by parameter ranges

### Plot Locations
- `results/plots/predictions_vs_actual.png`
- `results/plots/residual_analysis.png`
- `results/plots/feature_importance.png`

---

## Quick Execution Guide

```bash
# Step 1: Generate data & train model
uv run python ml/train_surrogate.py
# Output: data/raw/model.pkl, data/raw/analytical_results.csv

# Step 2: Quick CLI evaluation
uv run python ml/evaluate_model.py
# Output: Prints 4 metrics (MAE, RMSE, R², MAPE)

# Step 3: Comprehensive Jupyter analysis
uv run jupyter lab
# Open: notebooks/04_comprehensive_evaluation.ipynb
# Output: 6+ plots in results/plots/
```

---

## Key Physics Constants & Definitions

| Constant | Symbol | Value | Unit | Meaning |
|----------|--------|-------|------|---------|
| Elastic Constant | C66 | 3.99e10 | Pa | Orthotropic material property |
| Shear Modulus | C44 | 5.82e10 | Pa | Elastic layer |
| Viscoelastic Modulus | C44_ve | 6.34e10 | Pa | Viscoelastic half-space |
| Density 1 | ρ₁ | 4500 | kg/m³ | Elastic layer |
| Density 2 | ρ₂ | 3364 | kg/m³ | Viscoelastic half-space |

---

## Feature Importance Expected Order (Physics-based)

1. **k (Wave Number)** - Controls dispersion behavior
2. **H (Layer Thickness)** - Governs resonance/guiding
3. **η (Damping)** - Affects viscoelastic response
4. **α (Heterogeneity)** - Material variation effect

*(Actual rankings determined by trained GBR model)*

---

## Preprocessing Transformations

### StandardScaler Applied
```
For each feature:
    z = (x - μ) / σ
    
Where:
    μ = mean of feature in training set
    σ = standard deviation in training set
```

### Train/Test Split
```
Total valid samples: ~5,800
├── Training set: ~4,560 (80%)
└── Test set: ~1,140 (20%)
Random state: 42 (reproducible)
```

---

## File Structure

```
sh-wave-ml-surrogate/
├── physics/
│   ├── dispersion_solver.py    # Wave equation solver
│   ├── material_constants.py   # C66, C44, densities
│   └── parameter_sampler.py    # Random sampling
├── ml/
│   ├── models.py              # GBR & RF definitions
│   ├── train_surrogate.py     # Training pipeline
│   ├── evaluate_model.py      # Metric calculations
│   └── inverse_design.py      # Placeholder
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_training.ipynb
│   ├── 03_results_visualization.ipynb
│   └── 04_comprehensive_evaluation.ipynb
├── data/
│   ├── raw/
│   │   ├── analytical_results.csv
│   │   └── model.pkl
│   └── processed/
├── results/
│   └── plots/
├── pyproject.toml
├── requirements.txt
└── tests/
    └── test_dispersion_solver.py
```

---

## Dependencies

**Python Version:** ≥ 3.12

**Core Libraries:**
- numpy (≥2.4.1) - Numerical computing
- scipy (≥1.17.0) - Scientific computing (fsolve)
- scikit-learn (≥1.8.0) - ML models
- pandas (≥3.0.0) - Data handling
- matplotlib (≥3.10.8) - Visualization
- jupyter (≥1.1.1) - Notebook environment
- tensorflow (optional) - Future neural network models

---

## Performance Expectations

| Metric | Expected Value | Status |
|--------|----------------|--------|
| R² Score | 0.85 - 0.95 | Very Good ✓ |
| MAPE | 2-5% | Good ✓ |
| Training Time | ~30 seconds | Fast ✓ |
| Inference Time | ~1 ms/sample | 1000x faster ✓ |
| Model Size | ~1-5 MB | Compact ✓ |

---

## Known Limitations

- ✗ Inverse design: Placeholder only (not implemented)
- ✗ Test coverage: Minimal
- ✗ Sample efficiency: 6,000 samples baseline (could expand)
- ✗ Hyperparameter tuning: Baseline settings (not optimized)
- ✗ Cross-validation: Only train-test split (no k-fold CV)
- ✗ Neural networks: TensorFlow in requirements but unused

---

## Residual Diagnostics Checklist

Use this checklist to verify model quality:

```
✓ Mean of residuals ≈ 0              (No systematic bias)
✓ Constant variance (homoscedasticity) (No funnel pattern)
✓ Random scatter around zero          (Independence)
✓ Q-Q plot points near diagonal       (Normality)
✓ No correlation with input values    (Properly fitted)
```

---

## Contact & Documentation

- **README.md** - Project overview & quick start
- **ARCHITECTURE.md** - System architecture diagram
- **PROJECT_STRUCTURE.md** - File structure & execution flow
- **EVALUATION_GUIDE.md** - Detailed evaluation methodology
- **COMPREHENSIVE_ANALYSIS.md** - This detailed technical summary

