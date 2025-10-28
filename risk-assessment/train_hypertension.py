import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


BASE_DIR = Path("/home/ubuntu/datathon")
LINAGE_DIR = BASE_DIR / "pca-clinicalage" / "linAge"
PRECOMP_FILE = LINAGE_DIR / "dataMatrix_Normalized_With_Derived_Features_LinAge_PhenoAge.csv"
MODEL_DIR = BASE_DIR / "risk-assessment" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_precomputed_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _drop_high_missing(columns: List[str], frame: pd.DataFrame, max_missing_frac: float = 0.4) -> List[str]:
    if not columns:
        return columns
    miss = frame[columns].isna().mean()
    keep = [c for c in columns if float(miss.get(c, 0.0)) <= max_missing_frac]
    return keep


def select_htn_features(frame: pd.DataFrame) -> List[str]:
    """Select hypertension features, avoiding leakage and overly sparse columns.
    Policies:
      - Start from numeric columns
      - Remove identifiers/labels and cohort-only stats
      - Remove direct blood pressure measurements and BP questionnaire/medication variables
      - Filter columns with >40% missingness
      - Prefer a curated whitelist of cardio-metabolic risk factors when available
    """
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    # Base blacklist
    blacklist = {
        "Unnamed: 0",
        "SEQN",
        "label_hypertension",
        # Cohort-only / training-only
        "linAge",
        "delAge",
        "phenoAge",
        "phenoDelAge",
        "BAI",
        "bai_mean",
        "bai_std",
        "LBXCRP",
        "fs1Score",
        "fs2Score",
        "fs3Score",
    }

    # Regex-like exclusion via prefixes/substrings present in NHANES naming
    def _is_excluded_bp(name: str) -> bool:
        # Exclude direct BP measures and BP questionnaires/meds
        prefixes = (
            "BPX",   # examination measured BP (systolic/diastolic)
            "BPQ",   # questionnaire, includes the target and med use
        )
        if name.startswith(prefixes):
            return True
        # Any field that clearly denotes hypertension diagnosis/medication
        lowered = name.lower()
        return ("hypert" in lowered) or ("htn" in lowered)

    base = [c for c in numeric_cols if c not in blacklist and not _is_excluded_bp(c)]
    base = _drop_high_missing(base, frame, max_missing_frac=0.4)

    # Curated whitelist of cardio-metabolic factors expected to generalize
    curated = [
        "chronAge", "RIAGENDR", "BMXBMI", "BMXWAIST",
        "LBXGLU", "LBXGLUSI", "LBDSGLSI",
        "LBDLDLSI", "LBDHDLSI", "LBDTRSI",
        "LBDSCRSI",
        "URXUCRSI", "URXUMASI",
        "LBXCOT", "LBDCOTSI",
        "SSBNP", "LBXWBCSI", "LBXRDW", "LBXMCVSI", "LBXLYPCT",
        # Derived in this script or upstream
        "agi",
    ]

    # Use intersection to stay robust to availability
    curated_set = set(curated)
    features = [c for c in base if (c in curated_set)]
    # Fallback: if intersection is too small, keep base (still avoiding leakage)
    if len(features) < 8:
        features = base
    return features


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
    # add derived feature agi (age/bmi)
    with np.errstate(all='ignore'):
        data['agi'] = (data['chronAge'].astype(float) / data['BMXBMI'].astype(float))
    data['agi'] = data['agi'].fillna(data['agi'].mean())
    # # age / weight
    # data['awi'] = (data['chronAge'] / data['BMXWT'])
    # data['awi'] = data['awi'].fillna(data['awi'].mean())
    # # age / height
    # data['ahi'] = (data['chronAge'] / data['BMXHT'])
    # data['ahi'] = data['ahi'].fillna(data['ahi'].mean())

    data = data[~data["label_hypertension"].isna()].reset_index(drop=True)

    # Feature selection (avoid leakage and high-missingness)
    features = select_htn_features(data)
    if not features:
        raise ValueError("No features available after selection.")

    X_model = data[features].copy()
    y = data["label_hypertension"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_model, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, features)], remainder="drop")

    # L1-regularized logistic to reduce overfitting; class_weight balances classes
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        penalty="l1",
        C=0.5,
    )
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_valid = pipe.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, p_valid)
    ap = average_precision_score(y_valid, p_valid)
    # Diagnostics: prevalence and probability stats
    prevalence = float(y.mean())
    def _stats(a: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
        }
    # Tune threshold to maximize F1
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
        "prevalence": round(prevalence, 4),
        "train_prob": {k: round(v, 4) for k, v in _stats(p_train).items()},
        "valid_prob": {k: round(v, 4) for k, v in _stats(p_valid).items()},
        "valid_auc": round(auc, 4),
        "valid_ap": round(ap, 4),
        "best_threshold": round(best_threshold, 3),
        "precision@best": round(float(prec), 3),
        "recall@best": round(float(rec), 3),
        "f1@best": round(float(f1), 3),
    })
    print(classification_report(y_valid, y_pred_best))

    model_path = MODEL_DIR / "hypertension_broad_logreg.joblib"
    joblib.dump({
        "pipeline": pipe,
        "features": features,
        "metrics": {
            "valid_auc": float(auc),
            "valid_ap": float(ap),
            "prevalence": float(prevalence),
            "best_threshold": best_threshold,
            "train_prob_stats": _stats(p_train),
            "valid_prob_stats": _stats(p_valid),
            "precision_at_best": float(prec),
            "recall_at_best": float(rec),
            "f1_at_best": float(f1),
        },
    }, model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()


