# Customer Churn Prediction

Predicting which customers of a bike retailer will stop buying, using data from the gold layer of my SQL Server data warehouse (Medallion architecture, star schema).

## Problem

The business is non-contractual: customers don't cancel, they simply stop ordering. Churn is defined with a single cutoff:

- **History:** all orders before **2013-07-01** → used to build features
- **Window:** 2013-07-01 to 2014-01-01 (6 months)
- **Churned = 1** if a customer bought before the cutoff but placed **no order in the window**

Features are built only from history, so no information from the future leaks into the model.

11,414 customers · churn rate 63.4% (= the baseline to beat).

## Data

| File | Description |
|---|---|
| `data/fact_sales.csv`, `dim_customer.csv`, `dim_products.csv` | Exports of the warehouse gold views |
| `data/customer_features.csv` | One row per customer (built in notebook 01) |
| `data/churn_predictions.csv` | Test-set predictions with probability and risk level |

## Features

| Feature | Meaning |
|---|---|
| recency | days since last order (at cutoff) |
| frequency | number of distinct orders |
| monetary | total spend |
| tenure | days since first order |
| n_bikes, n_clothing, n_accessories | items bought per category |
| country, gender, marital_status | one-hot encoded |
| age | age at cutoff (missing → train median) |

## Models

Stratified 80/20 train/test split. Results on the held-out test set (2,283 customers):

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Baseline (always "churn") | 0.634 | 0.634 | 1.000 | 0.500 |
| Logistic regression (scaled) | 0.800 | 0.805 | 0.904 | 0.852 |
| Random forest | 0.908 | 0.907 | 0.954 | 0.961 |
| **XGBoost** | **0.908** | **0.918** | 0.938 | **0.968** |

XGBoost was chosen: same accuracy as the random forest, but the best ROC-AUC and precision.

**Overfitting check:** both tree models reach ~0.99 train accuracy. Regularising the random forest with `min_samples_leaf` (5/10/15) closed the gap but lowered test AUC, so the default settings were kept.

## Key findings

- **Recency is by far the strongest driver.** Permutation importance (drop in test AUC when shuffled): recency 0.293, monetary 0.085, country_America 0.046.
- XGBoost's built-in (gain) importance ranked rare categories such as `gender_Unknown` (10 customers) near the top; permutation importance showed they contribute ~0.
- **Churn rate by country:** America 77% · France 70% · Germany 65% · UK 57% · Canada 55% · **Australia 44%**. Combined with the dashboard finding that Australia has ~2× the revenue per customer, Australia is the strongest market.
- Risk levels on the test set: High 1,408 · Medium 103 · Low 772. The Medium group is the most actionable target for retention offers.

## Project structure

```
data/        gold-layer exports, features, predictions
notebooks/
  01_customer_segmentation.ipynb   feature engineering + churn label
  02_modeling.ipynb                training, evaluation, importance, export
```

## Run it

```bash
pip install -r requirements.txt
jupyter notebook
```

Run notebook 01, then notebook 02.

## Next steps

- Power BI page for churn risk (predictions linked to `dim_customer` on `customer_key`)
- Tune with cross-validation instead of the test set
- Exchange-rate API integration (Frankfurter/ECB) into the warehouse
