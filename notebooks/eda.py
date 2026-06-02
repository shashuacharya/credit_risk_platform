"""
eda.py — Exploratory Data Analysis for Home Credit Default Risk dataset.
Run standalone: python notebooks/eda.py
Outputs charts to notebooks/eda_outputs/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = os.environ.get("DATA_DIR", "../data")


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run_eda():
    csv = os.path.join(DATA_DIR, "application_train.csv")
    if not os.path.exists(csv):
        print(f"ERROR: {csv} not found. Place dataset in {DATA_DIR}/")
        return

    print("Loading data...")
    df = pd.read_csv(csv)
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['TARGET'].mean()*100:.2f}%")

    df["age_years"] = (-df["DAYS_BIRTH"] / 365).round(0)
    df["credit_income_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["employment_years"] = df["DAYS_EMPLOYED"].apply(
        lambda x: -x / 365 if x < 0 else np.nan
    )

    sns.set_theme(style="whitegrid")

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df["TARGET"].value_counts()
    ax.bar(["No Default", "Default"], vc.values, color=["#2ecc71", "#e74c3c"])
    ax.set_title("Loan Default Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate(vc.values):
        ax.text(i, v + 500, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center")
    save(fig, "01_target_distribution.png")

    # 2. Default rate by gender
    fig, ax = plt.subplots(figsize=(6, 4))
    g = df.groupby("CODE_GENDER")["TARGET"].mean() * 100
    g.plot.bar(ax=ax, color=["#3498db", "#e91e63", "#9b59b6"])
    ax.set_title("Default Rate by Gender", fontsize=14, fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    save(fig, "02_default_by_gender.png")

    # 3. Default rate by education
    fig, ax = plt.subplots(figsize=(8, 4))
    edu = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values() * 100
    edu.plot.barh(ax=ax, color="#e67e22")
    ax.set_title("Default Rate by Education Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("Default Rate (%)")
    save(fig, "03_default_by_education.png")

    # 4. Age distribution by default
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[df["TARGET"] == 0]["age_years"], bins=40, alpha=0.7,
            color="#2ecc71", label="No Default")
    ax.hist(df[df["TARGET"] == 1]["age_years"], bins=40, alpha=0.7,
            color="#e74c3c", label="Default")
    ax.set_title("Age Distribution by Default Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (Years)")
    ax.legend()
    save(fig, "04_age_distribution.png")

    # 5. External scores boxplot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, col in enumerate(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
        if col not in df.columns:
            continue
        df.boxplot(column=col, by="TARGET", ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("Default (0=No, 1=Yes)")
    plt.suptitle("External Credit Scores by Default Status")
    plt.tight_layout()
    save(fig, "05_external_scores.png")

    # 6. Credit income ratio
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[df["TARGET"] == 0]["credit_income_ratio"].clip(0, 15),
            bins=50, alpha=0.7, color="#2ecc71", label="No Default")
    ax.hist(df[df["TARGET"] == 1]["credit_income_ratio"].clip(0, 15),
            bins=50, alpha=0.7, color="#e74c3c", label="Default")
    ax.set_title("Credit-to-Income Ratio by Default Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Credit / Income Ratio")
    ax.legend()
    save(fig, "06_credit_income_ratio.png")

    # 7. Missing values
    fig, ax = plt.subplots(figsize=(12, 5))
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False).head(30)
    null_pct.plot.bar(ax=ax, color="#c0392b")
    ax.set_title("Top 30 Columns by Missing Value %", fontsize=14, fontweight="bold")
    ax.set_ylabel("Missing %")
    plt.xticks(rotation=45, ha="right")
    save(fig, "07_missing_values.png")

    # 8. Income type default rate
    fig, ax = plt.subplots(figsize=(10, 4))
    inc = df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values() * 100
    inc.plot.barh(ax=ax, color="#8e44ad")
    ax.set_title("Default Rate by Income Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Default Rate (%)")
    save(fig, "08_default_by_income_type.png")

    # Summary
    print("\n=== KEY BUSINESS INSIGHTS ===")
    print(f"1. Overall default rate: {df['TARGET'].mean()*100:.2f}%")
    print(f"2. Gender default rates: {df.groupby('CODE_GENDER')['TARGET'].mean().to_dict()}")
    print(f"3. Highest-risk income type: {(df.groupby('NAME_INCOME_TYPE')['TARGET'].mean()*100).idxmax()}")
    print(f"4. Avg age of defaulters: {df[df['TARGET']==1]['age_years'].mean():.1f} years")
    print(f"5. Avg EXT_SOURCE_2 for defaulters: {df[df['TARGET']==1]['EXT_SOURCE_2'].mean():.3f}")
    print(f"   Avg EXT_SOURCE_2 for non-defaulters: {df[df['TARGET']==0]['EXT_SOURCE_2'].mean():.3f}")
    print(f"\nAll charts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
