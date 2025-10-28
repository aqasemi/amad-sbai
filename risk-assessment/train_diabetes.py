import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


BASE_DIR = Path("/home/ubuntu/datathon")
LINAGE_DIR = BASE_DIR / "pca-clinicalage" / "linAge"
PRECOMP_FILE = LINAGE_DIR / "dataMatrix_Normalized_With_Derived_Features_LinAge_PhenoAge.csv"
MODEL_DIR = BASE_DIR / "risk-assessment" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_precomputed_linage_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_labels_and_meta(q_path: Path) -> pd.DataFrame:
    """Load qData and construct label with correct DIQ010 mapping.
    DIQ010: 1 Yes -> 1, 2 No -> 0, 3 Borderline/Unknown -> NaN, 7/9 -> NaN.
    Also returns RIAGENDR for optional feature use.
    """
    q = pd.read_csv(q_path)
    if "SEQN" not in q.columns:
        raise ValueError("SEQN not found in qDataMat.csv; cannot align with feature table.")
    if "DIQ010" not in q.columns:
        raise ValueError("DIQ010 not found in qDataMat.csv; cannot build diabetes labels.")
    mapping = {1: 1, 2: 0, 3: np.nan, 7: np.nan, 9: np.nan}
    q = q[["SEQN", "DIQ010", "RIAGENDR", "RIDAGEYR"]].copy()
    q["label_diabetes"] = q["DIQ010"].map(mapping)
    return q


def main() -> None:
    # Load features
    X = load_precomputed_linage_frame(PRECOMP_FILE)

    # Load labels and optional meta from qDataMat.csv, then left-join on SEQN
    q_path = LINAGE_DIR / "qDataMat.csv"
    q = build_labels_and_meta(q_path)
    if "SEQN" not in X.columns:
        raise ValueError("SEQN not found in precomputed linAge frame; cannot merge labels.")
    data = X.merge(q[["SEQN", "label_diabetes"]], on="SEQN", how="left")

    # Drop rows without label
    data = data[~data["label_diabetes"].isna()].reset_index(drop=True)

    # Build broad numeric feature set: take all numeric cols except identifiers/labels and
    # variables unavailable at inference time
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    blacklist = {
        "Unnamed: 0",  # artifact from saved DataFrame
        "SEQN",         # identifier
        "label_diabetes",  # target
        # Training-only cohort statistics not available at single-subject inference
        "bai_std",
        "bai_mean",
    }
    features = [c for c in numeric_cols if c not in blacklist]
    if not features:
        raise ValueError("No numeric features available after applying blacklist.")

    X_model = data[features].copy()
    y = data["label_diabetes"].astype(int)

    # Train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_model, y, test_size=0.2, random_state=42, stratify=y
    )

    # Numeric preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, features)], remainder="drop"
    )

    # Simple baseline model
    clf = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    # Evaluate
    p_valid = pipe.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, p_valid)
    ap = average_precision_score(y_valid, p_valid)
    print({"valid_auc": round(auc, 4), "valid_ap": round(ap, 4)})
    print(classification_report(y_valid, (p_valid >= 0.5).astype(int)))

    # Persist
    model_path = MODEL_DIR / "diabetes_linage_phenoage_logreg.joblib"
    joblib.dump({
        "pipeline": pipe,
        "features": features,
        "metrics": {"valid_auc": float(auc), "valid_ap": float(ap)},
    }, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()


