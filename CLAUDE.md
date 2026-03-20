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

- **Feature engineering (Step 1):** `perf_rel_t` is transformed into four piecewise-linear spline basis functions (`perf_rel_t_T1`–`T4`) at knots 2, 3, and 18. These are the forms used by the pre-estimated model, not the raw variable.
- **Default simulation (Step 2):** Gaussian noise is added to the logit score before ranking, so the top-5% default selection is probabilistic rather than purely deterministic. Exactly 500 of 10,000 rows are labelled as defaults.
- **Model fitting (Step 3):** Uses all 11 columns from Step 1 as predictors (7 raw + 4 spline terms). `statsmodels.Logit` is used for the coefficient/p-value summary; `sklearn.LogisticRegression` is used for AUC-ROC and the ROC curve plot.

### Pre-estimated coefficients (from Image #3)

| Feature | Coefficient |
|---|---|
| const | -7.1430 |
| perf_rel_t_T1 | 1.9057 |
| perf_rel_t_T2 | 0.0157 |
| perf_rel_t_T3 | -0.0303 |
| perf_rel_t_T4 | -0.0054 |
| ecm_0106_tot_num_T | 0.2352 |
| ecm_0601_tot_num_T | 0.0329 |
| fico_snp_T | -0.0004 |
| util_snp_T | 0.4751 |
| num_la_snp_6mo_T | 0.1048 |
| UR_M_3MGR_PIT_L6_NT | 3.3694 |
