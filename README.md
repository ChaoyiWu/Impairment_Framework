# Impairment Probability of Default (PD) Model — Showcase

A self-contained Jupyter notebook demonstrating an end-to-end workflow for developing an impairment **Probability of Default (PD)** model, from synthetic data generation through model fitting and evaluation.

---

## Overview

The showcase is structured in three steps:

| Step | Description |
|------|-------------|
| 1 | **Data Simulation** — Generate 10,000 synthetic loan-level observations with realistic feature distributions and engineer piecewise-linear spline terms |
| 2 | **Default Event Simulation** — Score each observation using pre-estimated logistic regression coefficients; simulate a binary `default_event` flag targeting a 5% default rate |
| 3 | **Model Fitting** — Fit a logistic regression model and evaluate performance via AUC-ROC, coefficient tables, and visualisations |

---

## Features Simulated

| Feature | Description | Distribution |
|---------|-------------|--------------|
| `perf_rel_t` | Performance relationship tenure (months) | Exponential(scale=25), clipped [1, 84] |
| `ecm_0106_tot_num_T` | 1–6 month ECM trade count | Normal(1.07, 0.60), clipped [0, 2.08] |
| `ecm_0601_tot_num_T` | 6–1 month ECM trade count | Exponential(scale=2), clipped [0, 15] |
| `fico_snp_T` | FICO/S&P credit score | Normal(2825, 1495), clipped [442, 6132] |
| `util_snp_T` | Utilisation ratio | Normal(1.74, 0.65), clipped [1.06, 2.87] |
| `num_la_snp_6mo_T` | Number of loan applications (6 mo) | Poisson(λ=2), clipped [0, 5] |
| `UR_M_3MGR_PIT_L6_NT` | Unemployment rate change (3-month, PIT) | Normal(-0.02, 0.038), clipped [-0.156, 0.204] |

Four spline basis functions are derived from `perf_rel_t` at knots **2, 3, and 18**:

| Derived Feature | Formula |
|----------------|---------|
| `perf_rel_t_T1` | min(x, 2) |
| `perf_rel_t_T2` | min(max(0, x − 2), 1) |
| `perf_rel_t_T3` | min(max(0, x − 3), 15) |
| `perf_rel_t_T4` | max(0, x − 18) |

---

## Pre-Estimated Model Coefficients

The logit score used in Step 2 is computed from:

| Feature | Coefficient |
|---------|-------------|
| Intercept | −7.1430 |
| perf_rel_t_T1 | 1.9057 |
| perf_rel_t_T2 | 0.0157 |
| perf_rel_t_T3 | −0.0303 |
| perf_rel_t_T4 | −0.0054 |
| ecm_0106_tot_num_T | 0.2352 |
| ecm_0601_tot_num_T | 0.0329 |
| fico_snp_T | −0.0004 |
| util_snp_T | 0.4751 |
| num_la_snp_6mo_T | 0.1048 |
| UR_M_3MGR_PIT_L6_NT | 3.3694 |

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn scipy jupyter
```

### Run the Notebook

```bash
jupyter notebook pd_model_showcase.ipynb
```

---

## Repository Structure

```
Impairment_Framework/
├── pd_model_showcase.ipynb   # Main showcase notebook
├── README.md
└── CLAUDE.md                 # Claude Code guidance
```
