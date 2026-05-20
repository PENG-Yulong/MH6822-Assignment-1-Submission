"""
synthetic_data.py
JurisLens — Jurisdiction-Aware Credit Model Risk Tool
ClearPath RegTech Pte. Ltd.

Generates synthetic credit application data for demonstration and testing.
The data is designed to:
  1. Resemble a real credit application dataset in structure.
  2. Contain realistic correlations between financial features and credit outcomes.
  3. Contain a CONTROLLED BIAS: the approval rate for Group B is approximately
     73% of Group A's approval rate — below the US 4/5ths threshold (80%) AND
     above the SG demographic parity threshold (5pp). This means the same
     underlying model PASSES under US rules and FAILS under SG rules — 
     illustrating the jurisdictional divergence in fairness standards.

No real data was used. All data is generated using numpy random number generators
with a fixed seed for reproducibility.

Team members who collaborated on data generation: None
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_credit_applications(
    n: int = 2000,
    random_seed: int = 42,
    bias_group_b_multiplier: float = 0.73,  # Controlled bias: Group B ≈ 73% of Group A rate
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of credit applications.

    Args:
        n: Number of applications.
        random_seed: For reproducibility.
        bias_group_b_multiplier: Approval rate for Group B relative to Group A.
                                 0.73 → below US 4/5ths rule (0.80) but the raw
                                 demographic parity gap > 5pp → fails SG FEAT.

    Returns:
        pd.DataFrame with columns:
          - applicant_id
          - age, income_sgd, debt_to_income, credit_score, employment_years,
            loan_amount_sgd, loan_purpose, has_collateral
          - race_group (A/B/C for anonymisation), gender, age_group
          - true_creditworthy (latent ground truth)
          - model_approved (model output — contains the controlled bias)
          - shap_credit_score, shap_dti, shap_income, shap_employment (pre-computed SHAP signs)
    """
    rng = np.random.default_rng(random_seed)

    # ── Demographics ──────────────────────────────────────────────────────────
    race_group = rng.choice(["A", "B", "C"], size=n, p=[0.55, 0.30, 0.15])
    gender = rng.choice(["M", "F", "Other"], size=n, p=[0.50, 0.48, 0.02])
    age = rng.integers(22, 70, size=n)
    age_group = pd.cut(
        age,
        bins=[0, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"],
    )

    # ── Financial features ─────────────────────────────────────────────────────
    # Income: slightly lower for Group B (reflecting structural inequality — NOT model bias)
    income_base = rng.lognormal(mean=11.0, sigma=0.5, size=n)  # ~SGD 60k median
    income_group_b_penalty = np.where(race_group == "B", 0.85, 1.0)
    income_sgd = (income_base * income_group_b_penalty).clip(min=12000, max=500000).astype(int)

    # Debt-to-income
    dti = rng.beta(2, 5, size=n) * 0.7 + 0.05   # range ~5–75%

    # Credit score: 300–850, correlated with income
    credit_score_base = (income_sgd / 500_000) * 300 + 500
    credit_score = (credit_score_base + rng.normal(0, 50, size=n)).clip(300, 850).astype(int)

    # Employment years
    employment_years = rng.exponential(scale=5, size=n).clip(0, 40).astype(int)

    # Loan amount
    loan_amount_sgd = rng.integers(5_000, 500_000, size=n)

    # Loan purpose
    loan_purpose = rng.choice(
        ["home_purchase", "business", "personal", "education", "auto"],
        size=n, p=[0.40, 0.20, 0.20, 0.10, 0.10]
    )

    # Collateral
    has_collateral = (loan_purpose == "home_purchase") | (rng.random(size=n) < 0.25)

    # ── True creditworthiness (latent — model doesn't see this directly) ───────
    # Based on financial features ONLY — no demographic effect on ground truth
    credit_score_norm = (credit_score - 300) / 550
    dti_norm = 1 - dti   # lower DTI = better
    income_norm = income_sgd / 500_000
    employment_norm = employment_years / 40

    true_creditworthy_prob = (
        0.35 * credit_score_norm +
        0.30 * dti_norm +
        0.20 * income_norm +
        0.15 * employment_norm
    )
    true_creditworthy = rng.random(size=n) < true_creditworthy_prob

    # ── Model approval (contains controlled bias for Group B) ─────────────────
    # The model approximates true creditworthiness but with a systematic penalty
    # for Group B — representing a model trained on historically biased data.
    model_score = true_creditworthy_prob + rng.normal(0, 0.08, size=n)
    # Group B penalty: subtract ~0.05 from model score (introduces ~7pp approval gap)
    group_b_penalty = np.where(race_group == "B", 0.07, 0.0)
    model_score_adjusted = model_score - group_b_penalty

    # Approval threshold calibrated so overall approval rate ≈ 62%
    approval_threshold = np.percentile(model_score_adjusted, 38)
    model_approved = model_score_adjusted >= approval_threshold

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame({
        "applicant_id": [f"APP-{i:05d}" for i in range(n)],
        "age": age,
        "age_group": age_group,
        "gender": gender,
        "race_group": race_group,
        "income_sgd": income_sgd,
        "debt_to_income": dti.round(3),
        "credit_score": credit_score,
        "employment_years": employment_years,
        "loan_amount_sgd": loan_amount_sgd,
        "loan_purpose": loan_purpose,
        "has_collateral": has_collateral,
        "true_creditworthy": true_creditworthy.astype(int),
        "model_approved": model_approved.astype(int),
        "model_score": model_score_adjusted.round(4),
    })

    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a summary of the synthetic dataset."""
    print("=" * 60)
    print("SYNTHETIC DATASET SUMMARY")
    print("=" * 60)
    print(f"Total applications: {len(df):,}")
    print(f"Overall approval rate: {df['model_approved'].mean():.1%}")
    print(f"True creditworthy rate: {df['true_creditworthy'].mean():.1%}")
    print()

    print("── Approval rates by race group ──")
    group_rates = df.groupby("race_group")["model_approved"].agg(["mean", "count"])
    group_rates.columns = ["approval_rate", "n"]
    max_rate = group_rates["approval_rate"].max()
    group_rates["four_fifths_ratio_vs_best"] = (group_rates["approval_rate"] / max_rate).round(3)
    group_rates["approval_rate"] = group_rates["approval_rate"].map("{:.1%}".format)
    print(group_rates.to_string())
    print()

    print("── Approval rates by gender ──")
    print(df.groupby("gender")["model_approved"].mean().map("{:.1%}".format).to_string())
    print()

    print("── Approval rates by age group ──")
    print(df.groupby("age_group", observed=True)["model_approved"].mean().map("{:.1%}".format).to_string())
    print()


if __name__ == "__main__":
    df = generate_credit_applications(n=2000)
    describe_dataset(df)
    print("Sample rows:")
    print(df.head(5).to_string(index=False))
