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


def load_precomputed_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_labels_hypertension(q_path: Path) -> pd.DataFrame:
    """Build hypertension label using NHANES BPQ020 (Ever told you had high blood pressure?).
    Mapping: 1 Yes -> 1, 2 No -> 0, 7/9 -> NaN. We ignore gestational qualifiers here.
    """
    q = pd.read_csv(q_path)
    if "SEQN" not in q.columns or "BPQ020" not in q.columns:
        raise ValueError("SEQN/BPQ020 not found in qDataMat.csv")
    mapping = {1: 1, 2: 0, 7: np.nan, 9: np.nan}
    out = q[["SEQN", "BPQ020"]].copy()
    out["label_hypertension"] = out["BPQ020"].map(mapping)
    return out


def main() -> None:
    X = load_precomputed_frame(PRECOMP_FILE)
    q_path = LINAGE_DIR / "qDataMat.csv"
    labels = build_labels_hypertension(q_path)

    if "SEQN" not in X.columns:
        raise ValueError("SEQN not found in precomputed frame; cannot merge labels.")

    data = X.merge(labels[["SEQN", "label_hypertension"]], on="SEQN", how="left")
    # add more features agi (age/bmi)
    data['agi'] = (data['chronAge'] / data['BMXBMI'])
    data['agi'] = data['agi'].fillna(data['agi'].mean())
    # # age / weight
    # data['awi'] = (data['chronAge'] / data['BMXWT'])
    # data['awi'] = data['awi'].fillna(data['awi'].mean())
    # # age / height
    # data['ahi'] = (data['chronAge'] / data['BMXHT'])
    # data['ahi'] = data['ahi'].fillna(data['ahi'].mean())

    data = data[~data["label_hypertension"].isna()].reset_index(drop=True)

    # Build feature set: all numeric columns except identifiers and target
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    blacklist = {
        "Unnamed: 0",
        "SEQN",
        "label_hypertension",
        # Exclude bioage and related deltas for this model
        # "chronAge",
        # "linAge",
        # "delAge",
        # "phenoAge",
        # "phenoDelAge",
        # "BAI",
        # Cohort-only stats
        # "bai_mean",
        # "bai_std",
    }
    features = [c for c in numeric_cols if c not in blacklist]
    if not features:
        raise ValueError("No features available after applying blacklist.")

    X_model = data[features].copy()
    y = data["label_hypertension"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_model, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, features)], remainder="drop")

    clf = LogisticRegression(max_iter=600, class_weight="balanced", solver="liblinear")
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    p_valid = pipe.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, p_valid)
    ap = average_precision_score(y_valid, p_valid)
    print({"valid_auc": round(auc, 4), "valid_ap": round(ap, 4)})
    print(classification_report(y_valid, (p_valid >= 0.5).astype(int)))

    model_path = MODEL_DIR / "hypertension_broad_logreg.joblib"
    joblib.dump({
        "pipeline": pipe,
        "features": features,
        "metrics": {"valid_auc": float(auc), "valid_ap": float(ap)},
    }, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()


