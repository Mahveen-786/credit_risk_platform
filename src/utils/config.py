"""
Central configuration for the AI-Powered Credit Risk Intelligence Platform.
All tunable constants live here so nothing is hardcoded deep in a module.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Config:
    # --- Data ---
    DATA_DIR: str = os.environ.get("DATA_DIR", "data")
    DATASET_FILENAME: str = os.environ.get("DATASET_FILENAME", "application_train.csv")
    SYNTHETIC_ROWS: int = int(os.environ.get("SYNTHETIC_ROWS", "15000"))
    RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))

    # --- Model artifacts ---
    MODELS_DIR: str = os.environ.get("MODELS_DIR", "models")
    MODEL_FILENAME: str = "lgbm_credit_risk.joblib"
    METRICS_FILENAME: str = "metrics.json"
    ARTIFACTS_BUNDLE_FILENAME: str = "artifacts_bundle.joblib"

    # --- Train/val split ---
    VAL_SIZE: float = 0.2

    # --- LightGBM hyperparameters ---
    N_ESTIMATORS: int = 400
    LEARNING_RATE: float = 0.03
    NUM_LEAVES: int = 31
    MIN_CHILD_SAMPLES: int = 40
    SUBSAMPLE: float = 0.8
    COLSAMPLE_BYTREE: float = 0.8

    # --- Cost-sensitive threshold tuning ---
    FN_COST_MULTIPLIER: float = float(os.environ.get("FN_COST_MULTIPLIER", "10"))
    THRESHOLD_SEARCH_STEPS: int = 199
    LOW_BAND_FRACTION: float = 0.45  # tau_low = tau_optimal * this fraction

    # --- Expected loss ---
    LGD: float = 0.45  # Loss Given Default benchmark

    # --- Score presentation transform (NOT a FICO score) ---
    SCORE_MAX: int = 850
    SCORE_MIN: int = 300
    SCORE_PD_MULTIPLIER: int = 550

    # --- Hard knockout policy thresholds ---
    KO_MAX_ANNUITY_TO_INCOME: float = 0.50
    KO_MAX_CREDIT_TO_GOODS_RATIO: float = 1.50
    KO_MIN_EXT_SOURCE_MEAN: float = 0.10

    # --- Surrogate rule extraction ---
    SURROGATE_TREE_MAX_DEPTH: int = 4
    SURROGATE_MIN_SUPPORT: int = 15
    SURROGATE_HIGH_RISK_QUANTILE: float = 0.92

    # --- Talk-to-Data ---
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
    LLM_MAX_TOKENS: int = 400
    SQL_MAX_ATTEMPTS: int = 2
    DUCKDB_TABLE_NAME: str = "applicants"

    CATEGORICAL_COLS: List[str] = field(default_factory=lambda: [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    ])

    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE", "AGE_YEARS", "EMPLOYED_YEARS", "EMPLOYED_ANOMALY_FLAG",
        "OWN_CAR_AGE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "EXT_SOURCE_MEAN", "EXT_SOURCE_MISSING_COUNT", "AMT_REQ_CREDIT_BUREAU_QRT",
        "ANNUITY_TO_INCOME", "CREDIT_TO_INCOME", "CREDIT_TO_GOODS_RATIO", "INCOME_PER_FAM_MEMBER",
    ])


config = Config()
