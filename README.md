# Heavy Equipment Price Prediction

A regression pipeline that predicts the price of heavy equipment (construction/industrial machinery) from auction transaction records, with an emphasis on domain-aware feature engineering and leakage-safe validation.

## Problem Statement

Accurately pricing used heavy equipment is a high-stakes regression problem — mispricing directly costs buyers or sellers money at auction. This project predicts sale price from transaction metadata, equipment specifications, and usage history across **138,701 training transactions** and **15,000 held-out test transactions**, evaluated using **RMSLE** (Root Mean Squared Log Error), which penalizes relative rather than absolute error — appropriate given the wide price range in the data ($7,500–$142,000).

## Dataset

- 50 raw columns spanning transaction metadata, equipment specifications (many anonymized, e.g. `col1`–`col30`), usage meters, and categorical descriptors
- Substantial missingness: several columns >90% missing; `OperationalHoursMeter` missing in 41% of rows
- Target (`TargetValue`) is right-skewed (skewness = 0.97), motivating a log-transform to align training with the RMSLE evaluation metric

## Approach

### 1. Exploratory Data Analysis
- Diagnosed a data quality anomaly: `ManufactureYear` contained placeholder values near year 1000, corrected by treating pre-1900 values as missing
- Identified 31,583 rows with `OperationalHoursMeter = 0`, almost certainly a placeholder for "unknown" rather than genuine zero-usage, and treated these as missing rather than literal zeros
- Confirmed via correlation analysis that `AssetAge` (derived) was a far stronger predictor than either raw `ManufactureYear` or `OperationalHoursMeter` (near-zero raw correlations of -0.008 and 0.002 respectively), and that the age–price relationship is non-linear (steep early depreciation, flattening with age)

### 2. Feature Engineering
- `AssetAge`: transaction year minus corrected manufacture year (negative/invalid ages set to null)
- `Hours_Per_Year`: usage rate normalized by asset age, capturing utilization intensity rather than raw hours
- `HasHours`: binary indicator preserving the signal in *whether* usage data was reported
- `Blank_Column_Count`: row-level count of missing fields, treating sparsity pattern as a feature
- `DescriptorLength`: length of the free-text equipment description field
- Normalized inconsistent null representations (e.g., "none or unspecified", "na", empty strings) into a single missing-value convention across 40+ categorical columns

### 3. Target Encoding & Leakage Prevention
- Applied **smoothed, K-fold-safe target encoding** for 8 high-cardinality categorical variables: an internal 5-fold split computes each fold's encoding from the *other* folds only, so no row's encoded value is ever derived from its own target — preventing target leakage into the encoding step itself
- Concatenated train/test for consistent categorical encoding, with the target column dropped *before* concatenation to eliminate any leakage vector
- Used quantile-stratified (5-bin) train/validation splitting to preserve price distribution across the split

### 4. Modeling
Benchmarked three model families — linear (Ridge), bagging (Random Forest), and gradient boosting (LightGBM) — with `RandomizedSearchCV` hyperparameter tuning, validated with 5-fold cross-validation:

| Model | RMSLE |
|---|---|
| Ridge Regression | 0.320 |
| Random Forest (tuned) | — |
| **LightGBM** (5-fold CV, out-of-fold) | **0.199** |

LightGBM was selected for its native handling of missing values and categorical splits, reducing RMSLE by **38%** relative to the Ridge baseline (0.320 → 0.199). Final performance was measured via out-of-fold (OOF) predictions across all 5 folds, and fold-level scores were compared against the aggregate OOF score to confirm stable generalization rather than a lucky split — Random Forest, itself a bagging ensemble of decision trees, was included in this comparison as the bagging benchmark rather than as a separately engineered ensembling step.

### 5. Why LightGBM Outperforms Ridge Here
The gap reflects a preprocessing-assumption mismatch rather than a simple "boosting beats linear models" story:
- Sentinel-value imputation for missing data is compatible with tree-based split logic but violates the linearity assumptions Ridge depends on
- The confirmed non-linear `AssetAge`–price relationship disadvantages a linear model without explicit polynomial or binned terms
- Tree-based models can exploit target-encoded categoricals through arbitrary-threshold splits in ways a linear model cannot

## Key Results

- **0.199 RMSLE** (LightGBM, 5-fold out-of-fold), a **38% reduction** from the Ridge baseline (0.320)
- Diagnosed and corrected two distinct data quality anomalies (invalid manufacture years, placeholder zero-hour readings) before they could bias the model
- Built a leakage-safe target encoding and validation pipeline suitable for a real auction-pricing scenario

## Tech Stack

Python · pandas · scikit-learn · LightGBM · K-fold target encoding · matplotlib/seaborn

