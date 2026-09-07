"""Model evaluation: ROC-AUC / PR-AUC, confusion matrix, and cost-sensitive
threshold search (minimizes FN_COST_MULTIPLIER * FalseNegatives + FalsePositives
instead of using an arbitrary 0.5 cutoff)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def tune_cost_sensitive_threshold(y_true, y_prob, fn_multiplier: float = None, n_steps: int = None):
    """Grid-searches the threshold that minimizes expected cost =
    FN * fn_multiplier + FP. Returns (best_threshold, cost_curve_df)."""
    fn_multiplier = fn_multiplier if fn_multiplier is not None else config.FN_COST_MULTIPLIER
    n_steps = n_steps or config.THRESHOLD_SEARCH_STEPS

    thresholds = np.linspace(0.01, 0.99, n_steps)
    costs = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn = ((pred == 0) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        costs.append(fn * fn_multiplier + fp)
    costs = np.array(costs)
    best_idx = costs.argmin()
    return float(thresholds[best_idx]), pd.DataFrame({"threshold": thresholds, "expected_cost": costs})


def evaluate_model(model, X_val, y_val):
    """Computes the full evaluation bundle used across the ML and UI layers."""
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    ap = average_precision_score(y_val, val_pred)

    tau_optimal, cost_curve = tune_cost_sensitive_threshold(y_val.values, val_pred)
    tau_low = max(tau_optimal * config.LOW_BAND_FRACTION, 0.02)

    cm = confusion_matrix(y_val, (val_pred >= tau_optimal).astype(int))

    metrics = {
        "auc_roc": auc,
        "avg_precision": ap,
        "tau_optimal": tau_optimal,
        "tau_low": tau_low,
        "confusion_matrix": cm,
        "val_default_rate": float(y_val.mean()),
        "n_val": len(X_val),
    }
    log.info("Eval -> AUC=%.4f  AP=%.4f  tau_optimal=%.4f", auc, ap, tau_optimal)
    return metrics, cost_curve
