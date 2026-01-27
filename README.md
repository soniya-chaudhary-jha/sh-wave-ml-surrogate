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