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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
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
        "linAge",
        "delAge",
        "phenoAge",
        "phenoDelAge",
        "BAI",
        # Cohort-only stats
        "bai_mean",
        "bai_std",
        "LBXCRP",
        "fs1Score",
        "fs2Score",
        "fs3Score",
    }
    features = [c for c in numeric_cols if c not in blacklist]
    print({"num_features": len(features)})

    if not features:
        raise ValueError("No numeric features available after applying blacklist.")

    X_model = data[features].copy()
    y = data["label_diabetes"].astype(int)

    # Train/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_model, y, test_size=0.2, random_state=42, stratify=y
    )

    # Numeric preprocessing (impute + scale)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, features)], remainder="drop"
    )

    # L1-regularized logistic to reduce overfitting; class_weight balances classes
    clf = LogisticRegression(
        max_iter=600,
        class_weight="balanced",
        solver="liblinear",
        penalty="l1",
        C=0.5,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    # Evaluate
    p_valid = pipe.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, p_valid)
    ap = average_precision_score(y_valid, p_valid)
    # Tune threshold for better balance on imbalanced data (optimize F1)
    candidate_thresholds = np.linspace(0.1, 0.9, 17)
    f1_scores = []
    for t in candidate_thresholds:
        y_pred = (p_valid >= t).astype(int)
        f1_scores.append(f1_score(y_valid, y_pred))
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(candidate_thresholds[best_idx])
    y_pred_best = (p_valid >= best_threshold).astype(int)
    prec = precision_score(y_valid, y_pred_best, zero_division=0)
    rec = recall_score(y_valid, y_pred_best, zero_division=0)
    f1 = f1_score(y_valid, y_pred_best)
    print({
        "valid_auc": round(auc, 4),
        "valid_ap": round(ap, 4),
        "best_threshold": round(best_threshold, 3),
        "precision@best": round(float(prec), 3),
        "recall@best": round(float(rec), 3),
        "f1@best": round(float(f1), 3),
    })
    print(classification_report(y_valid, y_pred_best))

    # Persist
    model_path = MODEL_DIR / "diabetes_linage_phenoage_logreg.joblib"
    joblib.dump({
        "pipeline": pipe,
        "features": features,
        "metrics": {
            "valid_auc": float(auc),
            "valid_ap": float(ap),
            "best_threshold": best_threshold,
            "precision_at_best": float(prec),
            "recall_at_best": float(rec),
            "f1_at_best": float(f1),
        },
    }, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()


