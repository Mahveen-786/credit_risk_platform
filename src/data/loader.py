"""
Loads the Home Credit Default Risk dataset.

Looks for the real Kaggle `application_train.csv` in the configured data
directory first. If it isn't there, generates a schema-matched SYNTHETIC
dataset (same real column names: SK_ID_CURR, TARGET, AMT_INCOME_TOTAL,
EXT_SOURCE_1/2/3, DAYS_BIRTH, DAYS_EMPLOYED, AMT_REQ_CREDIT_BUREAU_QRT, etc.)
so the platform always runs end-to-end, even before you've downloaded the
real file.
"""
import os
import numpy as np
import pandas as pd

from src.utils.config import config
from src.utils.docker_utils import get_dataset_path
from src.utils.logger import get_logger

log = get_logger(__name__)


SYNTHETIC_MARKER_SUFFIX = ".synthetic_seed"


def load_raw_data():
    """Returns (dataframe, message). message explains which source was used,
    for display in the UI. Honest even if the dataset file was auto-seeded
    by entrypoint.sh / seed_data.py: a marker file next to the CSV records
    that it's synthetic, so the UI never claims seeded data is the real
    Kaggle dataset just because a file happens to exist at that path."""
    path = get_dataset_path()
    marker_path = path + SYNTHETIC_MARKER_SUFFIX

    if os.path.exists(path):
        df = pd.read_csv(path)
        if os.path.exists(marker_path):
            msg = (
                f"No real dataset was found, so a schema-matched **synthetic** dataset "
                f"({len(df):,} rows) was auto-generated and seeded to `{path}` on "
                f"container startup. Replace that file with the real Kaggle "
                f"`{config.DATASET_FILENAME}` and restart to use real data."
            )
            log.warning(msg)
        else:
            msg = f"Loaded real dataset from `{path}` ({len(df):,} rows)."
            log.info(msg)
        return df, msg

    df = generate_synthetic_home_credit(n=config.SYNTHETIC_ROWS, seed=config.RANDOM_SEED)
    msg = (
        f"`{config.DATASET_FILENAME}` was not found in `{config.DATA_DIR}/`, so a "
        f"schema-matched **synthetic** dataset ({config.SYNTHETIC_ROWS:,} rows) was "
        f"generated instead so the platform runs end to end. Download the real Kaggle "
        f"file into that folder and refresh to use it."
    )
    log.warning(msg)
    return df, msg


def generate_synthetic_home_credit(n=15000, seed=42):
    """Synthetic data generator matching the real Kaggle application_train.csv
    schema and statistical shape (~8% default rate, known DAYS_EMPLOYED
    anomaly, realistic missingness)."""
    rng = np.random.default_rng(seed)

    sk_id = np.arange(100001, 100001 + n)

    contract_type = rng.choice(["Cash loans", "Revolving loans"], size=n, p=[0.9, 0.1])
    gender = rng.choice(["F", "M"], size=n, p=[0.66, 0.34])
    own_car = rng.choice(["Y", "N"], size=n, p=[0.34, 0.66])
    own_realty = rng.choice(["Y", "N"], size=n, p=[0.69, 0.31])
    cnt_children = rng.poisson(0.4, size=n).clip(0, 6)

    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student"],
        size=n, p=[0.51, 0.23, 0.18, 0.07, 0.005, 0.005],
    )
    education = rng.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        size=n, p=[0.71, 0.24, 0.03, 0.015, 0.005],
    )
    family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.64, 0.15, 0.10, 0.06, 0.05],
    )
    housing_type = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"],
        size=n, p=[0.88, 0.05, 0.035, 0.025, 0.01],
    )
    occupation = rng.choice(
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff",
         "Accountants", "Medicine staff", "Security staff", "Cooking staff"],
        size=n,
    )

    amt_income_total = np.round(rng.lognormal(mean=11.9, sigma=0.45, size=n), -2).clip(25000, 2000000)

    days_birth = -rng.integers(21 * 365, 69 * 365, size=n)
    age_years = -days_birth / 365.25

    is_pensioner = income_type == "Pensioner"
    days_employed = np.where(
        is_pensioner,
        365243,  # documented Home Credit anomaly code for retirees
        -rng.integers(30, 40 * 365, size=n),
    )

    ext_source_1 = rng.beta(2.2, 2.0, size=n)
    ext_source_2 = rng.beta(2.3, 1.9, size=n)
    ext_source_3 = rng.beta(2.1, 2.1, size=n)

    amt_goods_price = np.round(rng.lognormal(mean=12.1, sigma=0.55, size=n), -3).clip(45000, 4050000)
    over_finance_factor = rng.normal(1.05, 0.12, size=n).clip(0.7, 1.6)
    amt_credit = np.round(amt_goods_price * over_finance_factor, -3)
    term_months = rng.integers(6, 60, size=n)
    amt_annuity = np.round(amt_credit / term_months * rng.normal(1.02, 0.05, size=n), 1).clip(1600, None)

    cnt_fam_members = (cnt_children + rng.integers(1, 3, size=n)).clip(1, 9)
    own_car_age = np.where(own_car == "Y", rng.integers(0, 25, size=n), np.nan)

    bureau_inquiries_qrt = rng.poisson(0.7, size=n).clip(0, 12)

    annuity_to_income = amt_annuity / amt_income_total
    credit_to_goods = amt_credit / amt_goods_price
    employed_years = np.where(is_pensioner, 0, np.maximum(-days_employed, 0) / 365.25)

    ext_mean = (ext_source_1 + ext_source_2 + ext_source_3) / 3
    z = (
        -2.55
        - 6.8 * (ext_mean - 0.5)
        + 3.6 * np.clip(annuity_to_income - 0.35, 0, None)
        + 1.8 * np.clip(credit_to_goods - 1.0, 0, None)
        - 0.75 * np.tanh((employed_years - 3) / 5)
        - 0.5 * np.tanh((age_years - 40) / 15)
        + 0.22 * bureau_inquiries_qrt
        + 0.6 * (income_type == "Unemployed")
        + rng.normal(0, 0.4, size=n)
    )
    prob_default = 1 / (1 + np.exp(-z))
    target = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "SK_ID_CURR": sk_id,
        "TARGET": target,
        "NAME_CONTRACT_TYPE": contract_type,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": cnt_children,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "OCCUPATION_TYPE": occupation,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "OWN_CAR_AGE": own_car_age,
        "CNT_FAM_MEMBERS": cnt_fam_members,
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "AMT_REQ_CREDIT_BUREAU_QRT": bureau_inquiries_qrt,
    })

    for col, rate in [("EXT_SOURCE_1", 0.56), ("EXT_SOURCE_3", 0.19), ("OWN_CAR_AGE", 0.66),
                       ("AMT_GOODS_PRICE", 0.001), ("AMT_ANNUITY", 0.0004)]:
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    return df
