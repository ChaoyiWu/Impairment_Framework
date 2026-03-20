# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A showcase Jupyter notebook demonstrating impairment **Probability of Default (PD)** model development in three steps:
1. Simulate 10,000 loan-level observations with engineered spline features
2. Score observations using pre-estimated logistic regression coefficients to simulate a binary default flag (5% default rate)
3. Fit a logistic regression model and evaluate performance

## Running the Notebook

```bash
jupyter notebook pd_model_showcase.ipynb
# or
jupyter lab pd_model_showcase.ipynb
```

Required packages: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `scikit-learn`, `scipy`

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn scipy jupyter
```

## Architecture

**Single notebook:** `pd_model_showcase.ipynb`

### Key design decisions

- **Feature engineering (Step 1):** `time_since_obs` is transformed into four piecewise-linear spline basis functions (`perf_rel_t_T1`–`T4`) at knots 2, 3, and 18. These are the forms used by the pre-estimated model, not the raw variable.
- **Default simulation (Step 2):** Gaussian noise is added to the logit score before ranking, so the top-5% default selection is probabilistic rather than purely deterministic. Exactly 500 of 10,000 rows are labelled as defaults.
- **Model fitting (Step 3):** Uses all 11 columns from Step 1 as predictors (7 raw + 4 spline terms). `statsmodels.GLM` is used for the coefficient/p-value summary; custormized functions are used for VIF, discriminative power and accuracy.

### Pre-estimated coefficients (from Image #3)

| Feature | Description | Distribution |
|---------|-------------|--------------|
| `time_since_obs` | Month since observation month | Exponential(scale=25), clipped [1, 84] |
| `ct_open_cards` | count of open cards after transformation | Normal(1.07, 0.60), clipped [0, 2.08] |
| `ct_inquiries` | count of inquiries after transformation | Exponential(scale=2), clipped [0, 15] |
| `fico` | FICO credit score after transformation| Normal(2825, 1495), clipped [442, 6132] |
| `utilization` | utilisation ratio | Normal(1.74, 0.65), clipped [1.06, 2.87] |
| `ct_latefee` | count of late fees in last 6 months after transformation | Poisson(λ=2), clipped [0, 5] |
| `UR_M_3MGR_PIT_L6_NT` | Unemployment rate 3 month average change with 6 months lag  | Normal(-0.02, 0.038), clipped [-0.156, 0.204] |
