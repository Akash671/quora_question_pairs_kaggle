"""
src/model.py

Loads the trained model + scaler (and, once you've exported them, the
TF-IDF vectorizer and graph lookups) and exposes a predict() function.
"""

import os
import warnings

import joblib
import pandas as pd

try:
    from .features import build_features, FEATURE_COLUMNS
except ImportError as exc:
    if "attempted relative import" not in str(exc):
        raise
    from features import build_features, FEATURE_COLUMNS  # fallback if run as a script

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))

_ARTIFACT_FILES = {
    "model": "model.joblib",
    "scaler": "scaler.joblib",
    "tfidf": "tfidf_vectorizer.joblib",          # optional, see note below
    "question_freq": "question_freq.joblib",      # optional
    "question_neighbors": "question_neighbors.joblib",  # optional
}

_REQUIRED = {"model", "scaler"}


def _load_artifacts() -> dict:
    artifacts = {}
    missing_optional = []

    for key, filename in _ARTIFACT_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        elif key in _REQUIRED:
            raise FileNotFoundError(
                f"Required artifact '{filename}' not found in {MODEL_DIR}"
            )
        else:
            missing_optional.append(filename)

    if missing_optional:
        warnings.warn(
            "Missing optional artifact(s): " + ", ".join(missing_optional) +
            ". tfidf_cosine_sim / q1_freq / q2_freq / neighbor_overlap will "
            "be computed as 0 for every prediction until these are exported "
            "from training and placed in the model/ folder.",
            stacklevel=2,
        )

    return artifacts


# Load once at import time so repeated predict() calls are fast
_artifacts = _load_artifacts()


def predict(question1: str, question2: str) -> dict:
    """
    Predict whether two questions are duplicates.

    Returns: {"is_duplicate": bool, "probability": float}
    """
    feature_row = build_features(
        question1,
        question2,
        tfidf=_artifacts.get("tfidf"),
        question_freq=_artifacts.get("question_freq"),
        question_neighbors=_artifacts.get("question_neighbors"),
    )

    X = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    X_scaled = _artifacts["scaler"].transform(X)

    proba = float(_artifacts["model"].predict_proba(X_scaled)[0, 1])
    return {"is_duplicate": proba >= 0.5, "probability": proba}


if __name__ == "__main__":
    # quick manual test: python src/model.py
    result = predict("How do I learn Python?", "What's the best way to learn Python?")
    print(result)