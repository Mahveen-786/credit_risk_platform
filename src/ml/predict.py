"""
Inference layer: turns a raw probability of default (PD) into the
business-facing outputs -- AI Credit Risk Score, Risk Band, Expected Loss --
plus the Tier-1 hard-knockout policy gate.
"""
import numpy as np
import pandas as pd
import joblib

from src.utils.config import config
from src.utils.docker_utils import get_model_path
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_model():
    return joblib.load(get_model_path())


def pd_to_score(pd_prob):
    """AI Credit Risk Score: a 300-850 presentation transform of the model's
    predicted default probability. NOT a FICO score or a regulated credit score."""
    return np.clip(config.SCORE_MAX - pd_prob * config.SCORE_PD_MULTIPLIER, config.SCORE_MIN, config.SCORE_MAX)


def risk_band(pd_prob, tau_low, tau_optimal):
    if pd_prob < tau_low:
        return "Low"
    elif pd_prob < tau_optimal:
        return "Medium"
    return "High"


def expected_loss(pd_prob, amt_credit, lgd: float = None):
    lgd = lgd if lgd is not None else config.LGD
    return pd_prob * lgd * amt_credit


def hard_knockout_checks(row: pd.Series):
    """Tier 1: non-negotiable policy gate. Returns list of triggered knockout reasons."""
    reasons = []
    annuity_to_income = row.get("ANNUITY_TO_INCOME", np.nan)
    if pd.notna(annuity_to_income) and annuity_to_income > config.KO_MAX_ANNUITY_TO_INCOME:
        reasons.append(f"Annuity-to-Income ratio exceeds {config.KO_MAX_ANNUITY_TO_INCOME*100:.0f}% (unaffordable obligation)")
    if row.get("NAME_INCOME_TYPE") == "Unemployed":
        reasons.append("Applicant is unemployed with no verified income source")
    credit_to_goods = row.get("CREDIT_TO_GOODS_RATIO", np.nan)
    if pd.notna(credit_to_goods) and credit_to_goods > config.KO_MAX_CREDIT_TO_GOODS_RATIO:
        reasons.append(f"Requested credit exceeds {config.KO_MAX_CREDIT_TO_GOODS_RATIO*100:.0f}% of goods price (severe over-financing)")
    ext_mean = row.get("EXT_SOURCE_MEAN", np.nan)
    if pd.notna(ext_mean) and ext_mean < config.KO_MIN_EXT_SOURCE_MEAN:
        reasons.append(f"External bureau scores critically low (EXT_SOURCE_MEAN < {config.KO_MIN_EXT_SOURCE_MEAN})")
    return reasons


def score_applicant(model, row_df: pd.DataFrame, tau_low: float, tau_optimal: float):
    """Full decision trace for a single applicant row (already feature-engineered,
    matching config.FEATURE_COLS). Returns a dict with PD, score, band, expected
    loss, and any triggered hard knockouts."""
    pd_prob = float(model.predict_proba(row_df[config.FEATURE_COLS])[:, 1][0])
    score = float(pd_to_score(pd_prob))
    band = risk_band(pd_prob, tau_low, tau_optimal)
    el = float(expected_loss(pd_prob, row_df["AMT_CREDIT"].iloc[0]))
    knockouts = hard_knockout_checks(row_df.iloc[0])
    return {
        "pd_prob": pd_prob, "score": score, "band": band,
        "expected_loss": el, "knockouts": knockouts,
    }
