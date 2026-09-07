"""
Training pipeline: LightGBM classifier with class-imbalance handling
(scale_pos_weight, since the default rate is ~8%) and stratified split.

Business decision rules (Tier 1 hard knockouts, Tier 2 surrogate-tree
extraction) live in src.ml.rules, not here -- this module is training only.
"""
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.utils.config import config
from src.utils.docker_utils import get_model_path, get_metrics_path, get_artifacts_bundle_path
from src.utils.logger import get_logger
from src.ml.evaluate import evaluate_model

log = get_logger(__name__)


def train_model(df: pd.DataFrame, seed: int = None):
    """Trains the LightGBM default-prediction model. `df` must already be
    feature-engineered (see src.data.preprocessor.engineer_features).
    Returns (model, metrics, cost_curve, (X_train, X_val, y_train, y_val))."""
    seed = seed if seed is not None else config.RANDOM_SEED

    X = df[config.FEATURE_COLS]
    y = df["TARGET"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VAL_SIZE, stratify=y, random_state=seed
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    log.info("Class imbalance: %.2f%% positive -> scale_pos_weight=%.2f", 100 * pos / len(y_train), scale_pos_weight)

    model = LGBMClassifier(
        n_estimators=config.N_ESTIMATORS,
        learning_rate=config.LEARNING_RATE,
        num_leaves=config.NUM_LEAVES,
        max_depth=-1,
        min_child_samples=config.MIN_CHILD_SAMPLES,
        subsample=config.SUBSAMPLE,
        colsample_bytree=config.COLSAMPLE_BYTREE,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=config.CATEGORICAL_COLS)

    metrics, cost_curve = evaluate_model(model, X_val, y_val)
    metrics["n_train"] = len(X_train)
    metrics["scale_pos_weight"] = scale_pos_weight

    return model, metrics, cost_curve, (X_train, X_val, y_train, y_val)


def save_model_artifacts(model, metrics: dict, cost_curve: pd.DataFrame, splits: tuple):
    """Persists everything a consumer (the Streamlit app, a batch job) needs
    to skip retraining: the model itself, JSON-serializable metrics, and a
    joblib bundle with the cost curve + train/val splits. All written to
    models/ (mounted as a Docker volume in production)."""
    model_path = get_model_path()
    joblib.dump(model, model_path)
    log.info("Model saved to %s", model_path)

    serializable_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in metrics.items()}
    with open(get_metrics_path(), "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    log.info("Metrics saved to %s", get_metrics_path())

    X_train, X_val, y_train, y_val = splits
    bundle = {"cost_curve": cost_curve, "X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val}
    joblib.dump(bundle, get_artifacts_bundle_path())
    log.info("Artifacts bundle (splits + cost curve) saved to %s", get_artifacts_bundle_path())


def load_model_artifacts():
    """Loads everything save_model_artifacts wrote, so a fresh container
    start (or a fresh Streamlit session) can skip retraining entirely when a
    prior run already produced these files. Returns
    (model, metrics, cost_curve, (X_train, X_val, y_train, y_val))."""
    model = joblib.load(get_model_path())

    with open(get_metrics_path()) as f:
        metrics = json.load(f)
    metrics["confusion_matrix"] = np.array(metrics["confusion_matrix"])

    bundle = joblib.load(get_artifacts_bundle_path())
    cost_curve = bundle["cost_curve"]
    splits = (bundle["X_train"], bundle["X_val"], bundle["y_train"], bundle["y_val"])

    log.info("Loaded existing model + artifacts from %s (skipped retraining)", get_model_path())
    return model, metrics, cost_curve, splits


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.data.preprocessor import engineer_features

    raw_df, msg = load_raw_data()
    log.info(msg)
    feat_df = engineer_features(raw_df)
    model, metrics, cost_curve, splits = train_model(feat_df)
    save_model_artifacts(model, metrics, cost_curve, splits)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f} | Avg Precision: {metrics['avg_precision']:.4f}")

