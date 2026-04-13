# Notebook Files & Evaluation Guide

## 📁 Notebooks in Project

### 1. **01_data_generation.ipynb**
**Purpose:** Generate training data from physics simulations
- Samples 2000 random parameters (k, H, α, η)
- Solves dispersion relation for each sample
- Output: Raw training data with features and targets

**Usage:** Run to understand data generation pipeline

---

### 2. **02_training.ipynb**
**Purpose:** Train the ML surrogate model
- Loads generated data
- Scales features using StandardScaler
- Splits into train/test (80/20)
- Trains Gradient Boosting Regressor
- Shows R² score on test set

**Usage:** Run to understand training process and model selection

---

### 3. **03_results_visualization.ipynb**
**Purpose:** Basic visualization of model performance
- Predictions vs Actual scatter plot
- Shows accuracy visually
- Identifies outliers and problem regions

**Usage:** Quick visual check of model quality

---

### 4. **04_comprehensive_evaluation.ipynb** (NEW)
**Purpose:** Detailed evaluation with multiple metrics and diagnostics
- **Evaluation Metrics:**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
  - MAPE (Mean Absolute Percentage Error)
  
- **Visualizations:**
  - Predictions vs Actual scatter plot
  - Residual analysis (4 subplots)
  - Feature importance ranking
  - Error distribution by parameter ranges
  
- **Key Outputs:**
  - Residuals distribution (check for normality)
  - Q-Q plot (validate normal distribution assumption)
  - Error patterns by input parameters
  - Feature contribution to predictions

**Usage:** Run for complete model diagnosis and validation

---

## 📊 How to Check if Evaluation is Correct

### Metrics to Monitor:

| Metric | Good Range | Interpretation |
|--------|-----------|-----------------|
| **R² Score** | 0.85 - 1.0 | Higher = better fit; 0 = no fit; negative = worse than mean |
| **MAPE** | < 5% | Percentage error; intuitive for interpretation |
| **RMSE** | Low | Emphasizes large errors; same units as target |
| **MAE** | Low | Average absolute error; robust to outliers |

### Visual Checks:

1. **Predictions vs Actual Scatter Plot:**
   - Points should cluster near the diagonal line
   - No obvious patterns or systematic bias
   - Uniform spread (homoscedasticity)

2. **Residuals Plot:**
   - ✓ Mean residual near 0
   - ✓ No funnel shape (constant variance)
   - ✓ No patterns or trends
   - ✗ Curved pattern = nonlinear relationship missed

3. **Q-Q Plot:**
   - ✓ Points follow the diagonal line closely
   - ✗ Deviation at tails = outliers or non-normal errors

4. **Feature Importance:**
   - Identify which parameters matter most
   - Verify importance aligns with physics knowledge

---

## ⚠️ Note on Confusion Matrix

**This is a REGRESSION problem, NOT classification!**
- Confusion matrices are for classification (discrete classes)
- For regression, use:
  - **Scatter plots** (predictions vs actual)
  - **Residual plots** (errors analysis)
  - **Regression metrics** (MAE, RMSE, R², MAPE)

---

## 🚀 Running the Evaluation

### Option 1: Quick Evaluation (CLI)
```bash
uv run python ml/evaluate_model.py
```

### Option 2: Comprehensive Analysis (Jupyter)
```bash
uv run jupyter lab
# Open: notebooks/04_comprehensive_evaluation.ipynb
# Run all cells for full analysis
```

---

## 📈 Interpretation Guide

### R² Score Results:
- **R² > 0.95:** Excellent ✓ - Ready for production
- **R² 0.85-0.95:** Very Good ✓ - Suitable for most applications
- **R² 0.70-0.85:** Good ✓ - Can be used with caution
- **R² < 0.70:** Fair ⚠ - Consider retraining

### Next Steps:
1. Check residual plots for patterns
2. Review feature importance
3. If poor: Add features, more data, or better hyperparameters
4. Deploy model for inverse design optimization

