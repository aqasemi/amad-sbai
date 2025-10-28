## Technical Implementation Overview

This document describes the end-to-end technical design and implementation of the Datathon project, covering data sources, filtering, cleaning, feature engineering, modeling, inference, EDA, and tooling decisions. All paths referenced are absolute to simplify reproducibility on the target environment.

### Repository structure (relevant parts)

- `/home/ubuntu/datathon/streamlit_app.py`: UI, orchestration of linAge + risk models, single-subject inference, explanations
- `/home/ubuntu/datathon/pca-clinicalage/linAge/linAge.py`: linAge and PhenoAge computation, derived features, BAI
- `/home/ubuntu/datathon/pca-clinicalage/linAge/linAge_Paras.csv`: linAge parameters and metadata (per sex)
- `/home/ubuntu/datathon/risk-assessment/train_diabetes.py`: diabetes risk training (LogisticRegression)
- `/home/ubuntu/datathon/risk-assessment/train_hypertension.py`: hypertension risk training (LogisticRegression)
- `/home/ubuntu/datathon/risk-assessment/infer.py`: model loading and explanation utilities
- `/home/ubuntu/datathon/risk-assessment/models/*.joblib`: persisted trained models (pipelines + metadata)
- `/home/ubuntu/datathon/eda/*`: comprehensive EDA over large Datathon datasets (Pandas/Polars versions)

## Data sources and filtering

### Core linAge inputs
- Biomarkers and clinical parameters (`DATA`) and demographics (`DEMO`) follow NHANES-style variable names, e.g., `LBXCRP`, `LBDSALSI`, `LBDSCRSI`, `RIAGENDR`, `RIDAGEYR`.
- Default cohort used in the app is read from:
  - `DEFAULT_QDATA = /home/ubuntu/datathon/pca-clinicalage/linAge/qDataMat.csv`
  - `DEFAULT_DATA = /home/ubuntu/datathon/pca-clinicalage/linAge/dataMat_test.csv`
- A precomputed cohort output (recommended for reference stats and model training) is stored at:
  - `PRECOMP_OUTPUT = /home/ubuntu/datathon/pca-clinicalage/linAge/dataMatrix_Normalized_With_Derived_Features_LinAge_PhenoAge.csv`

### Age filter
- When computing linAge/PhenoAge for the cohort, the pipeline filters to individuals aged 40–84 at compute time: `40 ≤ RIDAGEYR < 85`.
- The Streamlit app also applies the same 40–84 filter for the default example frames.

### Labels
- Diabetes label (`label_diabetes`) from `qDataMat.csv:DIQ010` mapping: `{1: 1, 2: 0, 3/7/9: NaN}`.
- Hypertension label (`label_hypertension`) from `qDataMat.csv:BPQ020` mapping: `{1: 1, 2: 0, 7/9: NaN}`.
- Rows with NaN labels are dropped for supervised training.

## Cleaning and feature engineering (linAge engine)

The linAge engine is implemented in `linAge.py` and is invoked via `run_linage_on_frames(data_mat, q_data_mat, lin_age_pars_path, ...)`. It produces a feature-augmented frame with linAge, PhenoAge, and BAI.

### Transformations and derived features
- Cotinine binning (`LBXCOT`) → ordinal categories via thresholds (0/1/2/3).
- LDL cholesterol (`LDLV`) via Friedewald-like formula: `TotalChol - Triglycerides/5 - HDL` using `LBDTCSI`, `LBDSTRSI`, `LBDHDLSI`.
- Urine albumin-to-creatinine ratio (`crAlbRat`) from `URXUMASI` and `URXUCRSI` using a fixed conversion factor (`1.1312e-4`).
- Log transforms:
  - `LBXCRPN` preserves original CRP.
  - `LBXCRP = ln(LBXCRP)` (used in linAge), while PhenoAge uses CRP in mg/L derived from original CRP (`original_crp * 10`, clipped to 0.01 minimum).
  - `SSBNP = ln(SSBNP)`.
- Frailty/healthcare indices (from questionnaire `q_data_mat`):
  - `fs1Score` (co-morbidity index): ratio of 22 condition flags.
  - `fs2Score` (self-health index): function of `HUQ010` and `HUQ020`.
  - `fs3Score` (healthcare use index): cleaned `HUQ050` with `77/99 → 0`.

### linAge computation
- Parameters are read from `linAge_Paras.csv` which stores per-sex medians (`medVal`), MAD (`madVal`), coefficients (`betaVal`), and flags (`parType`, `parTrans`, `sexFlag`).
- Sex-specific standardization: for each sex, inputs are normalized as `(x - medVal) / madVal` (with NaN fallbacks to 0/1).
- The per-row delta age `delAge` is computed as:
  - Male: `c1_m + beta0_m * (RIDAGEYR * 12) + Σ(beta_m ⊙ standardized_DATA)`
  - Female: `c1_f + beta0_f * (RIDAGEYR * 12) + Σ(beta_f ⊙ standardized_DATA)`
- Biological age `linAge = chronAge + delAge`, where `chronAge = RIDAGEYR`.

### PhenoAge computation
- Uses the 9-biomarker Horvath/Levine-style mortality score and inversion to biological age.
- Biomarkers: albumin, creatinine, glucose, CRP (mg/L), lymphocyte percent, MCV, RDW, ALP, WBC, and chronological age.
- Implementation details:
  - CRP for PhenoAge uses `original_crp * 10` to convert to mg/L and clamps to `[1e-6, 1 - 1e-6]` for numerical stability.
  - Notes: The code comments reference some unit conversions (e.g., albumin g/L → g/dL, creatinine µmol/L → mg/dL) but the current implementation passes values as provided by `data_mat`. Alignments can be improved in future versions if needed.

### Biological Age Index (BAI)
- BAI is a z-score of `delAge` relative to age-bin (10-year) and sex-specific cohort statistics computed from the cohort or a precomputed reference.
- Categories: Decelerated, Normal Aging (|z| ≤ 1), Accelerated (1 < z ≤ 2), Highly Accelerated (z > 2).

## Inference pipeline (single subject)

Implemented in `streamlit_app.py → run_pipeline`:

1. Input sources:
   - User-uploaded merged CSV; or
   - Predefined sample row from `DEFAULT_DATA`, `DEFAULT_QDATA` (age 40–84 subset).
2. Column assurance:
   - Questionnaire/demographics: the app ensures presence of required columns by inserting missing columns with NaN.
   - Biomarkers: read required `DATA` variables from `linAge_Paras.csv` plus curated risk model features; missing columns are added with NaN.
3. linAge + PhenoAge:
   - Call `la.run_linage_on_frames(..., compute_bai_within_sample=False)`, then compute BAI for a single subject using reference cohort stats loaded from `PRECOMP_OUTPUT`.
4. Feature frame for risk models:
   - Concatenate linAge outputs with user `data` and `q` frames; drop duplicate columns.
   - Derived feature: `agi = chronAge / BMXBMI` when available.
5. Inference:
   - Load diabetes and hypertension models (`risk-assessment/infer.py`).
   - Score and show risk categories based on simple thresholds (<10% low, <20% borderline, otherwise high).
   - Provide explanations by computing contributions in standardized space and rescaling displayed values back to original units when possible.

## Supervised risk models

Both models are trained on the precomputed cohort frame `dataMatrix_Normalized_With_Derived_Features_LinAge_PhenoAge.csv` merged with labels from `qDataMat.csv`.

### Shared preprocessing
- Train/validation split: `test_size=0.2`, `random_state=42`, stratified.
- Preprocessing pipeline (scikit-learn):
  - `SimpleImputer(strategy="median")` on numeric features
  - `StandardScaler(with_mean=True, with_std=True)`
- Column selection: all numeric columns except identifiers, target, cohort-only stats, and selected bio-age columns (see blacklist below).

### Diabetes model (`train_diabetes.py`)
- Label: `label_diabetes` from `DIQ010` mapping.
- Features: all numeric except blacklist: `Unnamed: 0`, `SEQN`, `label_diabetes`, `linAge`, `delAge`, `phenoAge`, `phenoDelAge`, `BAI`, `bai_mean`, `bai_std`, `LBXCRP`, `fs1Score`, `fs2Score`, `fs3Score`.
- Classifier: `LogisticRegression(solver="liblinear", penalty="l1", C=0.5, class_weight="balanced", max_iter=600)`.
- Threshold tuning: grid over `[0.1 .. 0.9]` to optimize F1 on validation set; report AUC, AP, precision/recall/F1 at best threshold.
- Persistence: `joblib.dump({pipeline, features, metrics}, diabetes_linage_phenoage_logreg.joblib)`.

### Hypertension model (`train_hypertension.py`)
- Label: `label_hypertension` from `BPQ020` mapping.
- Features: same numeric inclusion rule with blacklist and an additional derived feature `agi = chronAge / BMXBMI` (filled with mean during training).
- Classifier: `LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=600)`.
- Metrics: AUC and AP on validation set.
- Persistence: `joblib.dump({pipeline, features, metrics}, hypertension_broad_logreg.joblib)`.

### Model inference and explanations (`risk-assessment/infer.py`)
- Loading: wrapper classes (`DiabetesRiskModel`, `HypertensionRiskModel`) load `{pipeline, features}` bundles.
- Feature alignment: missing trained features are added as NaN so the imputer can handle them; columns are ordered to match training.
- Explanations: compute standardized vector `x`, coefficients `coef`, contributions `coef * x`, intercept, logit, and probability; return top positive and negative contributors. The UI maps feature codes to human-friendly names using `codeBook.csv`.

## EDA and data preparation (Datathon-scale datasets)

The EDA scripts analyze large multi-table Datathon datasets and produce cleaning recommendations and an aggregated `combined_dataset` for downstream modeling.

- Pandas version: `/home/ubuntu/datathon/eda/comprehensive_eda.py`
- Polars version: `/home/ubuntu/datathon/eda/comprehensive_eda_polars.py`

Key steps:
- Chunked loading (Pandas) or memory-efficient reads (Polars) across Individuals, Deaths, Steps, Medications, Lab Tests.
- Dataset overview, missingness profiling, categorical distributions, and numeric outlier detection (IQR).
- Unit standardization analysis for key labs (e.g., glucose, cholesterol) with recommendations to harmonize units.
- Aggregations per `personalid`:
  - Medications: `total_prescriptions`, `unique_drugs`
  - Lab tests: `total_lab_tests`, `unique_test_types`
  - Steps (latest year): `latest_steps`, `latest_calories`, `latest_distance`, `latest_movetimeduration`
- Feature engineering:
  - `age_group` (bins), `in_target_age` flag (40–55), `comorbidity_count`, and `healthcare_engagement` composite.
- Outputs: plots to `eda_outputs/plots`, textual summaries, and `combined_dataset.csv`/`combined_dataset_polars.csv`.

Note: These EDA outputs inform cleaning and modeling decisions but are separate from the linAge+NHANES-style pipeline used by the app and the risk models.

## Tooling choices and rationale

- Python + scikit-learn: fast iteration, stable APIs, and interpretability (linear models with coefficients and per-feature contributions).
- Logistic Regression (liblinear):
  - Interpretable coefficients; L1 regularization (`diabetes`) encourages sparsity and mitigates overfitting.
  - `class_weight="balanced"` to handle label imbalance; robust baselines under time constraints.
- Pandas/Polars: efficient data handling across different dataset sizes; Polars option for improved memory use.
- Streamlit: rapid UI to demonstrate end-to-end flow, capture uploads, and visualize results and explanations.
- Joblib: reliable model bundle persistence (`pipeline + feature list + metrics`).

## End-to-end pipeline summary

1. Prepare cohort: (Optional) Regenerate linAge/PhenoAge/BAI cohort frame
   - `uv run python /home/ubuntu/datathon/pca-clinicalage/linAge/linAge.py`
   - Produces `dataMatrix_Normalized_With_Derived_Features_LinAge_PhenoAge.csv` used by training and as reference stats.
2. Train models:
   - Diabetes: `uv run python /home/ubuntu/datathon/risk-assessment/train_diabetes.py`
   - Hypertension: `uv run python /home/ubuntu/datathon/risk-assessment/train_hypertension.py`
   - Artifacts saved under `/home/ubuntu/datathon/risk-assessment/models/`.
3. Run app:
   - `uv run streamlit run /home/ubuntu/datathon/streamlit_app.py`
   - Upload a merged CSV or select a predefined sample.
   - The app ensures required columns, computes linAge and PhenoAge, derives BAI using reference stats, and scores risk models with explanations.

## Data validation and required inputs

The app ensures presence of key questionnaire columns (set to NaN if missing) and biomarker columns from `linAge_Paras.csv` (parType `DATA`) plus curated risk features.

Examples of required columns:
- Questionnaire/demographics (subset): `BPQ020`, `DIQ010`, `RIAGENDR`, `RIDAGEYR`, `HUQ010`, `HUQ020`, `HUQ050`, `HUQ070`, `KIQ020`, `MCQ*`, `OSQ*`, `PFQ056`.
- Biomarkers (subset): `LBXCRPN`, `SSBNP`, `LBXCOT`, `LBDTCSI`, `LBDHDLSI`, `LBDSTRSI`, `URXUCRSI`, `URXUMASI`, `LBDSALSI`, `LBDSCRSI`, `LBDSGLSI`, `LBXLYPCT`, `LBXMCVSI`, `LBXRDW`, `LBXSAPSI`, `LBXWBCSI`, and `BMXBMI` for `agi`.

## Limitations and future work

- Unit harmonization: some PhenoAge biomarker unit conversions are noted in comments but not applied consistently in code; this can be addressed in a future iteration.
- Calibration: probability calibration (e.g., Platt/Isotonic) and cross-validation could further improve calibration and generalization.
- Threshold selection: app uses coarse risk tiers; consider data-driven thresholds per condition and population.
- Error handling: expand schema validation and user feedback for malformed uploads.
- Expand models: incorporate longitudinal or behavioral features (e.g., steps, medications) once harmonized with NHANES-style features.

## Reproducibility notes

- Random seeds: training uses `random_state=42` where applicable.
- Models persist their feature lists; inference aligns features (adding missing as NaN) to match training.
- Absolute paths in code intentionally reduce environment drift on the target VM.

## Appendix: Key functions and classes

- linAge engine: `run_linage_on_frames`, `compute_bai`, `calculate_pheno_age`, `pop_lin_age`, `populate_ldl`, `pop_cr_alb_rat`, `digi_cot`, and frailty indices.
- Risk training: `train_diabetes.py`, `train_hypertension.py` (ColumnTransformer + LogisticRegression).
- Inference: `DiabetesRiskModel`, `HypertensionRiskModel` and `explain_from_frame`.
- UI helpers: human label mapping via `pca-clinicalage/codeBook.csv`, value rescaling in explanations to original units when scaler stats are available.

## Diagrams

### Data cleaning and preparation

```mermaid
flowchart TD
  A[Start] --> B{Input source}
  B -->|Merged CSV upload| C[Ensure required questionnaire columns\nadd missing as NaN]
  B -->|Predefined sample| D[Filter 40–84 from default cohort\nqDataMat.csv + dataMat_test.csv]

  %% Assure biomarker/feature columns
  C --> E[Ensure required DATA features from linAge_Paras.csv\n+ curated model features; add missing as NaN]
  D --> F[Pass-through D/Q frames]

  %% linAge transforms (cleaning/derivations)
  E --> G[linAge transforms\n- digi_cot(LBXCOT)\n- LDLV via Friedewald\n- crAlbRat\n- log(LBXCRP), log(SSBNP)\n- fs1/fs2/fs3 indices]
  F --> G

  G --> H[Compute chronAge, delAge, linAge]
  H --> I[Compute PhenoAge + phenoDelAge]
  I --> J[Compute BAI\n10y age bins + sex;\nreference z-score]

  J --> K[Assemble modeling feature frame\nconcat linAge outputs + D + Q; drop duplicates; derive agi]

  K --> L{Context}
  L -->|Training| M[Merge labels (DIQ010/BPQ020)\ndrop rows with NaN labels]
  L -->|Inference| N[Align to trained feature list\nadd missing as NaN]

  M --> O[Preprocess: SimpleImputer(median) + StandardScaler]
  N --> O
  O --> P[Standardized feature matrix ready for model]
```

### End-to-end system pipeline

```mermaid
flowchart LR
  subgraph EDA[EDA over Datathon datasets]
    E1[Load Individuals/Deaths/Steps/Medications/Labs\n(chunked Pandas / Polars)] --> E2[Missingness, outliers, unit checks]
    E2 --> E3[Aggregate per personalid; engineer features\n(age_group, engagement, comorbidity)]
    E3 --> E4[Outputs: combined_dataset*.csv, plots, summaries]
  end

  subgraph L[linAge cohort computation]
    L1[Load qDataMat.csv + dataMat_test.csv\nfilter 40–84] --> L2[run_linage_on_frames\ntransforms + linAge + PhenoAge + BAI]
    L2 --> L3[Write precomputed frame\n dataMatrix_Normalized_...csv]
  end

  subgraph T[Supervised training]
    T1[Load precomputed frame + labels\n(DIQ010/BPQ020)] --> T2[Select numeric features\napply blacklist; derive agi (HTN)]
    T2 --> T3[Imputer + Scaler]
    T3 --> T4[LogisticRegression\n(balanced; L1 for diabetes; threshold tuning)]
    T4 --> T5[Persist joblib bundles\n(pipeline, features, metrics)]
  end

  subgraph S[Streamlit app inference]
    S1[Upload merged CSV or select sample] --> S2[Assure columns; run linAge;\nBAI via reference; build features; agi]
    S2 --> S3[Load joblib models; predict probabilities]
    S3 --> S4[Explain contributions; rescale values; UI cards]
  end

  E4 -. informs .-> T2
  L3 --> T1
  L3 --> S2
  T5 --> S3
```


