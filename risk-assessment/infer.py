from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib


BASE_DIR = Path("/home/ubuntu/datathon")
MODEL_PATH = BASE_DIR / "risk-assessment" / "models" / "diabetes_linage_phenoage_logreg.joblib"
HTN_MODEL_PATH = BASE_DIR / "risk-assessment" / "models" / "hypertension_broad_logreg.joblib"


class DiabetesRiskModel:
    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]

    def predict_proba_from_linage_frame(self, linage_frame: pd.DataFrame) -> np.ndarray:
        # Ensure all trained features exist; add missing with NaN so the imputer can handle them
        X = linage_frame.copy()
        for col in self.features:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.features]
        return self.pipeline.predict_proba(X)[:, 1]

    def explain_from_linage_frame(self, linage_frame: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """Return per-feature contributions on the log-odds for a single row.
        The contributions sum to the model's logit (up to numerical precision).
        """
        if linage_frame is None or linage_frame.empty:
            raise ValueError("linage_frame must contain at least one row")

        X = linage_frame.copy()
        for col in self.features:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.features]

        # Access transformer and classifier
        prep = self.pipeline.named_steps.get("prep")
        clf = self.pipeline.named_steps.get("model")
        if prep is None or clf is None:
            raise RuntimeError("Pipeline must contain 'prep' and 'model' steps")

        # Transform (impute) features
        X_proc = prep.transform(X)
        x = np.asarray(X_proc)[0].astype(float)

        # Get feature names post-transform (strip transformer prefix if present)
        try:
            names = list(prep.get_feature_names_out())
        except Exception:
            # Fallback to original features if get_feature_names_out is unavailable
            names = list(self.features)

        def _strip_prefix(n: str) -> str:
            return n.split("__", 1)[1] if "__" in n else n

        names = [_strip_prefix(n) for n in names]

        coefs = np.asarray(clf.coef_).ravel()
        intercept = float(np.asarray(clf.intercept_).ravel()[0]) if hasattr(clf, "intercept_") else 0.0
        if coefs.shape[0] != len(x):
            raise RuntimeError("Coefficient vector and transformed feature vector have mismatched lengths")

        contrib = coefs * x
        logit = float(intercept + contrib.sum())
        prob = float(1.0 / (1.0 + np.exp(-logit)))

        # Build detailed frame
        details = pd.DataFrame({
            "feature": names,
            "value": x,
            "coef": coefs,
            "contribution": contrib,
        }).sort_values("contribution", ascending=False).reset_index(drop=True)

        top_pos = details.head(top_k).copy()
        top_neg = details.sort_values("contribution", ascending=True).head(top_k).copy()

        return {
            "probability": prob,
            "logit": logit,
            "bias": intercept,
            "contributions_frame": details,
            "top_positive": top_pos,
            "top_negative": top_neg,
        }


class HypertensionRiskModel:
    def __init__(self, model_path: Path = HTN_MODEL_PATH):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]

    def predict_proba_from_frame(self, features_frame: pd.DataFrame) -> np.ndarray:
        X = features_frame.copy()
        for col in self.features:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.features]
        return self.pipeline.predict_proba(X)[:, 1]

    def explain_from_frame(self, features_frame: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        if features_frame is None or features_frame.empty:
            raise ValueError("features_frame must contain at least one row")

        X = features_frame.copy()
        for col in self.features:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.features]

        prep = self.pipeline.named_steps.get("prep")
        clf = self.pipeline.named_steps.get("model")
        if prep is None or clf is None:
            raise RuntimeError("Pipeline must contain 'prep' and 'model' steps")

        X_proc = prep.transform(X)
        x = np.asarray(X_proc)[0].astype(float)

        try:
            names = list(prep.get_feature_names_out())
        except Exception:
            names = list(self.features)

        def _strip_prefix(n: str) -> str:
            return n.split("__", 1)[1] if "__" in n else n

        names = [_strip_prefix(n) for n in names]

        print(names)
        coefs = np.asarray(clf.coef_).ravel()
        intercept = float(np.asarray(clf.intercept_).ravel()[0]) if hasattr(clf, "intercept_") else 0.0
        if coefs.shape[0] != len(x):
            raise RuntimeError("Coefficient vector and transformed feature vector have mismatched lengths")

        contrib = coefs * x
        logit = float(intercept + contrib.sum())
        prob = float(1.0 / (1.0 + np.exp(-logit)))

        details = pd.DataFrame({
            "feature": names,
            "value": x,
            "coef": coefs,
            "contribution": contrib,
        }).sort_values("contribution", ascending=False).reset_index(drop=True)

        top_pos = details.head(top_k).copy()
        top_neg = details.sort_values("contribution", ascending=True).head(top_k).copy()

        return {
            "probability": prob,
            "logit": logit,
            "bias": intercept,
            "contributions_frame": details,
            "top_positive": top_pos,
            "top_negative": top_neg,
        }


def load_model() -> DiabetesRiskModel:
    return DiabetesRiskModel()


def load_hypertension_model() -> HypertensionRiskModel:
    return HypertensionRiskModel()


