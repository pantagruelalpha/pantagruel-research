---
title: "Credit Risk Modelling with Gradient Boosting: From PD to IFRS 9"
description: "A step-by-step guide to building a Probability of Default model using LightGBM, calibrating it for IFRS 9 expected credit loss computation, and validating it with proper time-based cross-validation."
pubDate: "Mar 18 2026"
heroImage: "/post_img.webp"
badge: "RISK"
tags: ["credit-risk", "machine-learning", "IFRS9"]
---

## Why Credit Risk Modelling is a Perfect ML Problem

Credit scoring has clear binary labels (default / no default), large datasets, and a direct economic objective. Gradient boosting methods — particularly **LightGBM** and **XGBoost** — consistently outperform logistic regression on consumer and corporate credit tasks.

But deploying a model in a regulated context (Basel III, IFRS 9) introduces constraints: the model must be **interpretable**, **well-calibrated**, and validated on out-of-sample data with a proper time split.

---

## Data Preparation

The key challenge is avoiding **label leakage**: features must only use information available at origination time, not at observation time.

```python
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# Features at origination date
features = [
    "ltv_ratio", "dti_ratio", "credit_score", "loan_term",
    "employment_years", "annual_income_log", "interest_rate",
    "property_type_enc", "macro_gdp_growth", "macro_unemployment"
]

X = df[features]
y = df["default_12m"]  # 1 if defaulted within 12 months
```

## Training with Time-Based Cross-Validation

Never use random k-fold for credit data — a loan issued in 2023 cannot be used to predict a loan from 2022.

```python
tscv = TimeSeriesSplit(n_splits=5)
aucs = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    aucs.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")

print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
```

## Calibration for IFRS 9

IFRS 9 requires a **point-in-time Probability of Default (PD)** that reflects current economic conditions. Raw LightGBM scores are well-ranked but not well-calibrated probabilities. We use isotonic regression to fix this:

```python
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)

pd_estimates = calibrated.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, pd_estimates)
print(f"Brier Score: {brier:.4f}")  # lower is better
```

## Expected Credit Loss Computation

Under IFRS 9, ECL is computed as:

```
ECL = PD × LGD × EAD
```

Where:
- **PD**: Probability of Default (our model output)
- **LGD**: Loss Given Default (typically 40–60% for unsecured retail)
- **EAD**: Exposure at Default (outstanding balance at default time)

```python
df["pd"] = pd_estimates
df["lgd"] = 0.45  # simplified flat LGD
df["ead"] = df["outstanding_balance"]
df["ecl"] = df["pd"] * df["lgd"] * df["ead"]

print(f"Total ECL provision: €{df['ecl'].sum():,.0f}")
```

## Model Interpretability with SHAP

Regulators require understanding of model decisions. SHAP values provide additive feature attributions consistent with model predictions:

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")
```

The most important features in our experiment were `credit_score`, `dti_ratio`, and `macro_unemployment` — consistent with economic intuition and making the model defensible to auditors.

## Key Takeaways

1. **Time-based splits are mandatory** — random splits inflate AUC by 3–5 points.
2. **Calibrate your probabilities** — uncalibrated scores lead to under/over-provisioning.
3. **SHAP for every model** — regulators and risk committees expect to understand driver variables.
4. **Monitor for drift** — retrain quarterly and track PSI (Population Stability Index) on input distributions.
