---
title: "Transformer Models for Financial Time Series Forecasting"
description: "A practical guide to applying Temporal Fusion Transformers and Informer architectures to equity return prediction, covering data preparation, purged cross-validation and avoiding look-ahead bias."
pubDate: "Mar 10 2026"
heroImage: "/post_img.webp"
tags: ["machine-learning", "time-series", "transformers"]
badge: "NEW"
---

## Introduction

The transformer architecture, originally proposed for natural language processing in *Attention Is All You Need* (Vaswani et al., 2017), has fundamentally changed sequence modelling. Its application to financial time series is promising but requires careful adaptation: financial data is non-stationary, low signal-to-noise, and subject to regime changes that can make a model trained on past data almost useless going forward.

In this article we focus on two architectures — **Temporal Fusion Transformers (TFT)** and **Informer** — and walk through a production-grade pipeline for cross-sectional equity return prediction.

---

## Why Standard Transformers Struggle with Finance

Vanilla transformers have quadratic complexity in sequence length and assume stationarity implicitly through positional encodings. Financial time series violate these assumptions in several ways:

- **Non-stationarity**: Price levels are integrated of order 1 (I(1)); returns are stationary but exhibit conditional heteroscedasticity (GARCH effects).
- **Low SNR**: Academic estimates put the Sharpe ratio of most alpha signals below 0.5 annualised.
- **Irregular sampling**: Corporate actions, earnings dates, and macro releases create irregular event-driven dynamics.

## Temporal Fusion Transformers

The TFT (Lim et al., 2021) addresses several of these issues with a multi-horizon architecture that explicitly separates:

1. **Static covariates** (e.g. sector, market cap regime)
2. **Known future inputs** (e.g. earnings date dummies, macro release calendar)
3. **Observed past inputs** (e.g. returns, volume, analyst revisions)

```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

dataset = TimeSeriesDataSet(
    data=panel_df,
    time_idx="time_idx",
    target="fwd_ret_1m",
    group_ids=["ticker"],
    static_categoricals=["sector"],
    time_varying_known_reals=["earnings_dummy", "macro_release"],
    time_varying_unknown_reals=["ret_1d", "volume_z", "analyst_revision"],
    max_encoder_length=60,
    max_prediction_length=21,
)

model = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    loss=QuantileLoss(),
)
```

## Purged Cross-Validation

Standard k-fold cross-validation causes **information leakage** in financial series due to autocorrelation. The solution is **purged + embargo cross-validation** (de Prado, 2018):

```python
from mlfinlab.cross_validation import PurgedKFold

cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

for train_idx, test_idx in cv.split(X, pred_times=t, eval_times=e):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
```

The purging step removes training samples whose labels overlap with the test window, while the embargo discards samples immediately after the test set to prevent leakage via autocorrelated features.

## Results and Practical Considerations

In our experiments on S&P 500 constituents (2010–2024), TFT with purged CV achieved an **information coefficient (IC) of 0.048** on 1-month forward returns — modest but statistically significant (t-stat > 3.0). Key takeaways:

- **Feature engineering matters more than architecture**: log-volume z-score, short-interest ratio, and momentum reversal features added 30% IC improvement over price-only inputs.
- **Regularise aggressively**: dropout > 0.2 and early stopping are essential.
- **Do not overfit on the validation set**: use a true out-of-sample holdout for final evaluation.

In the next article we will use these return predictions as inputs to a portfolio optimisation layer.
