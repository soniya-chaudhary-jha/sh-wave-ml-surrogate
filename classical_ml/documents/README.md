# SH Wave Propagation – ML Surrogate Model

This project develops a physics-driven machine learning surrogate to model shear (SH) wave propagation in orthotropic elastic layers bonded to heterogeneous viscoelastic half-spaces.  
The ML model replaces repeated analytical dispersion solving with fast and accurate predictions.

---

## 🚀 Project Pipeline


---

## ✨ Key Features

- Analytical dispersion relation solver for SH-wave propagation  
- Physics-generated training dataset  
- Machine learning surrogate for rapid prediction  
- Optional inverse material design framework  

---

## 📥 Input Parameters (Physics Features)

| Symbol | Description |
|-------|------------|
| k | Wave number |
| H | Orthotropic layer thickness |
| α | Heterogeneity parameter |
| η | Viscoelastic damping parameter |

---

## 📤 Model Outputs

- Phase velocity of SH wave (c)  
- *(Optional)* Attenuation/damping from complex wave speed  

---

## 🧠 Machine Learning Models

- Gradient Boosting Regressor (primary surrogate)  
- Random Forest Regressor (baseline)  
- Neural Networks (for large datasets)  

---

## ▶️ How to Run the Pipeline

Generate dataset and train the ML surrogate:

```bash
uv run python ml/train_surrogate.py
uv run python ml/evaluate_model.py
uv run jupyter lab
```

### Quick Start (recommended)

1. Create and activate a virtual environment, then install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the surrogate (saves model + scaler to `data/models`):

```bash
python scripts/train.py
```

3. Evaluate the trained model:

```bash
python scripts/evaluate.py
```

Notes:
- Place the dataset at `data/dispersion_vs_L.xlsx` or `data/raw/dispersion_vs_L.xlsx`. CSV works too.
- Executed notebooks are saved under `executed_notebooks/` when run with `nbconvert`.