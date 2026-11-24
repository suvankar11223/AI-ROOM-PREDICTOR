"""Small, dependency-light monitoring utilities; connect these to production telemetry."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report


def population_stability_index(reference, current, bins=10, epsilon=1e-6):
    """Calculate PSI for numeric or categorical values."""
    ref = pd.Series(reference).dropna()
    cur = pd.Series(current).dropna()
    if ref.empty or cur.empty:
        raise ValueError("Both reference and current samples must contain non-null values.")
    if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
        edges = np.unique(np.quantile(ref.astype(float), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.linspace(float(ref.min()), float(ref.max()) + epsilon, bins + 1)
        ref_bins = pd.cut(ref.astype(float), bins=edges, include_lowest=True).value_counts(sort=False)
        cur_bins = pd.cut(cur.astype(float), bins=edges, include_lowest=True).value_counts(sort=False)
        # Values outside the reference range are counted in explicit under/overflow buckets.
        ref_counts = np.r_[ref_bins.to_numpy(), 0, 0]
        cur_counts = np.r_[cur_bins.to_numpy(), (cur.astype(float) < edges[0]).sum(), (cur.astype(float) > edges[-1]).sum()]
    else:
        categories = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        ref_counts = ref.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
        cur_counts = cur.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy()
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), epsilon, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def labelled_performance(y_true, y_pred):
    """Return headline and class-wise metrics once ground truth is available."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def retraining_recommendation(max_feature_psi=None, prediction_psi=None,
                              current_macro_f1=None, baseline_macro_f1=None,
                              current_shared_recall=None, baseline_shared_recall=None):
    """Return a conservative recommendation; never triggers automatic retraining."""
    reasons = []
    critical = False
    for name, value in (("feature PSI", max_feature_psi), ("prediction PSI", prediction_psi)):
        if value is not None and value >= 0.20:
            critical = True; reasons.append(f"{name} >= 0.20")
        elif value is not None and value >= 0.10:
            reasons.append(f"{name} >= 0.10")
    if current_macro_f1 is not None and baseline_macro_f1 is not None:
        drop = baseline_macro_f1 - current_macro_f1
        if drop >= 0.10 - 1e-9:
            critical = True; reasons.append("macro F1 dropped by at least 0.10")
        elif drop >= 0.05 - 1e-9:
            reasons.append("macro F1 dropped by at least 0.05")
    if current_shared_recall is not None and baseline_shared_recall is not None:
        drop = baseline_shared_recall - current_shared_recall
        if drop >= 0.10 - 1e-9:
            critical = True; reasons.append("Shared room recall dropped by at least 0.10")
        elif drop >= 0.05 - 1e-9:
            reasons.append("Shared room recall dropped by at least 0.05")
    if critical:
        action = "HOLD_PROMOTION_AND_INVESTIGATE"
    elif reasons:
        action = "INVESTIGATE"
    else:
        action = "NO_TRIGGER_FROM_SUPPLIED_METRICS"
    return {"action": action, "reasons": reasons,
            "automatic_retraining": False,
            "note": "Retrain only after data provenance, labels, validation, calibration, and rollback checks pass."}
