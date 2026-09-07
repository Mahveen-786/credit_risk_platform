"""
Feature engineering / preprocessing for the credit risk platform:
- decodes the known DAYS_EMPLOYED == 365243 anomaly (pensioners/unemployed)
- derives the ratio features used across ML, XAI, and business rules
- casts categorical columns to pandas 'category' dtype for LightGBM
"""
import numpy as np
import pandas as pd

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # DAYS_EMPLOYED == 365243 is Home Credit's documented anomaly code for
    # pensioners / unemployed applicants with no active employment record.
    df["EMPLOYED_ANOMALY_FLAG"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).fillna(0)

    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["INCOME_PER_FAM_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_MISSING_COUNT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().sum(axis=1)

    for c in config.CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")

    log.info("Feature engineering complete: %d columns -> %d columns", len(df.columns), len(df.columns))
    return df
