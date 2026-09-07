"""
Exploratory Data Analysis -- Home Credit Default Risk
This is the script conversion of notebooks/eda.ipynb (kept in sync manually;
run either one). It reproduces the "Executive EDA" tab of the Streamlit app
as a standalone, notebook-free script, and saves charts to notebooks/eda_outputs/.

Run from the project root:  python notebooks/eda.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_raw_data
from src.data.preprocessor import engineer_features

OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    raw_df, msg = load_raw_data()
    print(msg)
    df = engineer_features(raw_df)

    print("\n=== Dataset Summary ===")
    print(f"Applicants: {len(df):,}")
    print(f"Default rate: {df['TARGET'].mean()*100:.2f}%")
    print(f"Average missingness: {df.isna().mean().mean()*100:.1f}%")
    print("\nTop missing columns:")
    print(df.isna().mean().sort_values(ascending=False).head(8))

    # --- Insight 1: External Score Non-Linearity ---
    bins = pd.qcut(df["EXT_SOURCE_MEAN"].fillna(df["EXT_SOURCE_MEAN"].median()), 8, duplicates="drop")
    g1 = df.groupby(bins, observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    g1.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title("Insight 1: Default Rate by External Bureau Score Bucket")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "insight1_ext_source.png"))
    plt.close()

    # --- Insight 2: Debt-Trap Ratio ---
    dti_bins = pd.cut(df["ANNUITY_TO_INCOME"].clip(upper=1.0), bins=[0, 0.15, 0.25, 0.35, 0.5, 1.0])
    g2 = df.groupby(dti_bins, observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    g2.plot(kind="bar", ax=ax, color="#ff7f0e")
    ax.set_title("Insight 2: Default Rate by Annuity-to-Income Bucket")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "insight2_debt_trap.png"))
    plt.close()

    # --- Insight 3: Employment & Age Stability ---
    age_bins = pd.cut(df["AGE_YEARS"], bins=[20, 30, 40, 50, 60, 70])
    g3 = df.groupby(age_bins, observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    g3.plot(kind="bar", ax=ax, color="#2ca02c")
    ax.set_title("Insight 3: Default Rate by Age Bracket")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "insight3_age.png"))
    plt.close()

    # --- Insight 4: Over-Financing Risk ---
    g4 = df.assign(OVER_FINANCED=(df["CREDIT_TO_GOODS_RATIO"] > 1.0)).groupby("OVER_FINANCED", observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    g4.plot(kind="bar", ax=ax, color="#d62728")
    ax.set_title("Insight 4: Default Rate -- Over-Financed vs. Not")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "insight4_overfinancing.png"))
    plt.close()

    # --- Insight 5: Bureau Inquiry Bursts ---
    g5 = df.groupby("AMT_REQ_CREDIT_BUREAU_QRT", observed=True)["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    g5.plot(kind="bar", ax=ax, color="#9467bd")
    ax.set_title("Insight 5: Default Rate by Bureau Inquiries (last quarter)")
    ax.set_ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "insight5_bureau_inquiries.png"))
    plt.close()

    print(f"\nSaved 5 insight charts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
