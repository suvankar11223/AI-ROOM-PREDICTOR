"""Validated inference helpers for the NYC Airbnb room-type model bundles."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

SUPPORTED_FORMAT = "nyc_airbnb_room_type_model_bundle_v3"


def load_model(path: str | Path) -> dict:
    """Load a trusted model bundle. Never load Joblib files from untrusted sources."""
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or bundle.get("bundle_format") != SUPPORTED_FORMAT:
        raise ValueError("Unsupported model bundle format.")
    if not bundle.get("feature_columns") or not bundle.get("estimator"):
        raise ValueError("Model bundle is missing its estimator or feature contract.")
    if list(bundle["estimator"].classes_) != list(bundle["classes"]):
        raise ValueError("Stored class order does not match the fitted estimator.")
    return bundle


def validate_input(bundle: dict, listing_data) -> pd.DataFrame:
    """Validate a dict or DataFrame and return columns in the exact model order.

    Numeric nulls are permitted because the fitted pipeline imputes them. Missing
    review rates are only permitted when number_of_reviews is zero.
    Unknown neighbourhood labels are permitted and handled by OneHotEncoder.
    """
    if isinstance(listing_data, pd.DataFrame):
        frame = listing_data.copy()
    elif isinstance(listing_data, dict):
        frame = pd.DataFrame([listing_data])
    else:
        raise TypeError("Input must be a pandas DataFrame or a dictionary.")
    if frame.empty:
        raise ValueError("At least one listing is required.")

    required = list(bundle["feature_columns"])
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    frame = frame.loc[:, required].copy()

    numeric = bundle["input_schema"].get("numeric_features", [])
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        values = frame[col].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature '{col}' contains an infinite value.")

    constraints = bundle["input_schema"].get("constraints", {})
    for col, bounds in constraints.items():
        if col not in frame.columns:
            continue
        lo, hi = bounds
        values = frame[col].dropna()
        if lo is not None and (values < lo).any():
            raise ValueError(f"Feature '{col}' must be >= {lo}.")
        if hi is not None and (values > hi).any():
            raise ValueError(f"Feature '{col}' must be <= {hi}.")

    if "reviews_per_month" in frame.columns and "number_of_reviews" in frame.columns:
        inconsistent = frame["reviews_per_month"].isna() & frame["number_of_reviews"].fillna(0).gt(0)
        if inconsistent.any():
            raise ValueError("Missing reviews_per_month is only accepted when number_of_reviews is zero.")
    return frame


def predict(bundle: dict, listing_data) -> pd.DataFrame:
    """Return predicted label, selected-class probability, and class probabilities."""
    frame = validate_input(bundle, listing_data)
    estimator = bundle["estimator"]
    probabilities = estimator.predict_proba(frame)
    classes = list(bundle["classes"])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("The model returned invalid probability values.")

    policy = bundle.get("decision_policy", {"type": "argmax_calibrated_probability"})
    if policy.get("type") == "shared_room_probability_threshold":
        threshold = float(policy["threshold"])
        shared_idx = classes.index("Shared room")
        other_idx = [i for i, label in enumerate(classes) if label != "Shared room"]
        predictions = np.asarray(classes, dtype=object)[np.argmax(probabilities[:, other_idx], axis=1)]
        predictions = np.where(probabilities[:, shared_idx] >= threshold, "Shared room", predictions)
    else:
        predictions = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]

    result = pd.DataFrame({"predicted_room_type": predictions}, index=frame.index)
    result["predicted_class_probability"] = [probabilities[i, classes.index(label)] for i, label in enumerate(predictions)]
    for i, label in enumerate(classes):
        result[f"probability_{label}"] = probabilities[:, i]
    return result
