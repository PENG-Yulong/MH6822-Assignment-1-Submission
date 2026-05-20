"""
main.py
JurisLens — Jurisdiction-Aware Credit Model Risk Tool
ClearPath RegTech Pte. Ltd.

Entry point. Run this script to execute the full demonstration:

  python main.py

What this script demonstrates:
  1. Jurisdiction divergence summary (US vs SG — what the rules actually differ on)
  2. Synthetic credit data generation
  3. ML model training (GradientBoostingClassifier) with SHAP explainability
  4. Fairness check under US rules (4/5ths) — result: FAIL
  5. Fairness check under SG rules (FEAT dual metric) — result: FAIL (same model, same data,
     but different method catches different violations)
  6. Adverse action notice — same denied applicant, two different notices
  7. PSI drift simulation — same model scores, same PSI value, but different required
     action under US vs SG thresholds
  8. Examination-readiness reports for both jurisdictions

Dependencies: see requirements.txt
  pip install -r requirements.txt
"""
import subprocess, sys; subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import shap

from jurisdiction_config import get_config, compare_jurisdictions
from synthetic_data import generate_credit_applications, describe_dataset
from compliance_engine import (
    run_fairness_check,
    generate_adverse_action_notice,
    compute_psi,
    assess_psi_action,
    generate_examination_readiness_report,
)

# ─────────────────────────────────────────────────────────────────────────────
SECTION_DIVIDER = "\n" + "█" * 65 + "\n"
# ─────────────────────────────────────────────────────────────────────────────


def section(title: str):
    print(SECTION_DIVIDER)
    print(f"  {title}")
    print()


def main():
    print("=" * 65)
    print("  JURISLENS — Jurisdiction-Aware Credit Model Risk Tool")
    print("  ClearPath RegTech Pte. Ltd. | Demo Run")
    print(f"  {date.today()}")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: Jurisdiction Divergence Summary
    # ─────────────────────────────────────────────────────────────────────────
    section("SECTION 1: JURISDICTION DIVERGENCE — US vs SINGAPORE")

    comparison = compare_jurisdictions("US", "SG")
    print(f"Comparing: {comparison['jurisdiction_a']} vs {comparison['jurisdiction_b']}")
    print(f"Material divergences identified: {comparison['divergence_count']}")
    print()
    for i, div in enumerate(comparison["divergences"], 1):
        print(f"  [{i}] {div['dimension']}")
        print(f"       US: {div.get('US', '')}")
        print(f"       SG: {div.get('SG', '')}")
        print(f"       Risk: {div['risk']}")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: Synthetic Data
    # ─────────────────────────────────────────────────────────────────────────
    section("SECTION 2: SYNTHETIC CREDIT APPLICATION DATA")
    print("Generating 2,000 synthetic credit applications...")
    print("Design note: Group B has a systematic ~7pp approval disadvantage")
    print("built into the model score — this will be detected differently")
    print("under US vs SG fairness rules.\n")

    df = generate_credit_applications(n=2000, random_seed=42)
    describe_dataset(df)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: Model Training + SHAP
    # ─────────────────────────────────────────────────────────────────────────
    section("SECTION 3: MODEL TRAINING AND SHAP EXPLAINABILITY")

    feature_cols = [
        "credit_score", "debt_to_income", "income_sgd",
        "employment_years", "has_collateral", "loan_amount_sgd"
    ]
    df["has_collateral"] = df["has_collateral"].astype(int)

    X = df[feature_cols].values
    y = df["model_approved"].values

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.3, random_state=42, stratify=y
    )

    print("Training GradientBoostingClassifier (representative of XGBoost/LightGBM family)...")
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nModel performance on held-out test set:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Denied", "Approved"],
                                 digits=3))

    # SHAP explainability
    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # shap_values shape: (n_test, n_features) — positive = increases approval probability

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    print("\nGlobal feature importance (mean |SHAP|):")
    for feat, importance in sorted(zip(feature_cols, mean_abs_shap),
                                   key=lambda x: -x[1]):
        print(f"  {feat:<25} {importance:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: Fairness Checks — Same Model, Different Rules
    # ─────────────────────────────────────────────────────────────────────────
    section("SECTION 4: FAIRNESS — SAME MODEL, DIFFERENT JURISDICTIONS")

    print("Running fairness checks on the FULL dataset (n=2,000).")
    print("The SAME model outputs are assessed under two different rule sets.\n")

    # US: 4/5ths rule
    print("── US Fairness Check (4/5ths Rule) ────────────────────────────")
    us_fairness = run_fairness_check(df, "US", protected_attribute="race_group")
    print(us_fairness["narrative"])
    print(f"\nResult: {us_fairness['result']}")
    print(f"Violations: {len(us_fairness['violations'])}")
    print()

    # SG: FEAT dual metric
    print("── SG Fairness Check (MAS FEAT Dual Metric) ───────────────────")
    sg_fairness = run_fairness_check(df, "SG", protected_attribute="race_group")
    print(sg_fairness["narrative"])
    print(f"\nResult: {sg_fairness['result']}")
    print(f"Violations: {len(sg_fairness['violations'])}")

    print()
    print("━" * 65)
    print("KEY INSIGHT: Same model data. Different results by design.")
    print(f"  US result:  {us_fairness['result']} ({len(us_fairness['violations'])} violations)")
    print(f"  SG result:  {sg_fairness['result']} ({len(sg_fairness['violations'])} violations)")
    print()
    print("  A tool that only checked one jurisdiction's rules would give")
    print("  an institution operating in both a dangerously incomplete picture.")
    print("━" * 65)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: Adverse Action Notices
    # ─────────────────────────────────────────────────────────────────────────
    section("SECTION 5: ADVERSE ACTION NOTICES — SAME APPLICANT, TWO FORMATS")

    # Pick a denied applicant from the test set
    denied_mask = df_test["model_approved"] == 0
    if denied_mask.sum() == 0:
        print("No denied applicants in test set — skipping.")
    else:
        denied_applicant = df_test[denied_mask].iloc[0]
        denied_idx = denied_mask[denied_mask].index[0]
        # Get SHAP values for this applicant
        test_position = df_test.index.get_loc(denied_idx)
        applicant_shap = dict(zip(feature_cols, shap_values[test_position]))

        print(f"Applicant: {denied_applicant.get('applicant_id', 'N/A')}")
        print(f"Credit score: {denied_applicant['credit_score']}")
        print(f"Debt-to-income: {denied_applicant['debt_to_income']:.1%}")
        print(f"Income: SGD {denied_applicant['income_sgd']:,}")
        print()

        print("── US Adverse Action Notice (Regulation B 2026) ───────────")
        print(generate_adverse_action_notice(denied_applicant, applicant_shap, "US"))
        print()
        print("── SG Adverse Action Notice (MAS FEAT 2024) ───────────────")
        print(generate_adverse_action_notice(denied_applicant, applicant_shap, "SG"))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: PSI Drift Monitoring
    # ─────────────────────────────────────────────────────────────────────────
    section("SECTION 6: POPULATION DRIFT — PSI MONITORING")

    baseline_scores = y_prob[:300]   # First 300 test observations as "baseline"

    # Simulate three scenarios
    scenarios = {
        "Stable (no drift)": y_prob[:300] + np.random.default_rng(1).normal(0, 0.01, 300),
        "Moderate drift": y_prob[:300] + np.random.default_rng(1).normal(-0.08, 0.05, 300),
        "Major drift (post-shock)": y_prob[:300] + np.random.default_rng(1).normal(-0.18, 0.08, 300),
    }

    for scenario_name, current_scores in scenarios.items():
        current_scores = np.clip(current_scores, 0.01, 0.99)
        psi_val = compute_psi(baseline_scores, current_scores)
        print(f"\nScenario: {scenario_name}")
        print(f"  PSI = {psi_val:.5f}")
        for jx in ["US", "SG"]:
            result = assess_psi_action(psi_val, jx)
            print(f"  {jx} action required: {result['action']}")
        # Note: SG has a tighter revalidation trigger (0.20 vs US 0.25)
        # So the same PSI value can require revalidation in SG but only monitoring in US

    print()
    print("━" * 65)
    print("KEY INSIGHT: SG revalidation trigger is 0.20; US is 0.25.")
    print("  The same PSI reading can require revalidation in Singapore")
    print("  while only requiring enhanced monitoring in the US.")
    print("  A single-jurisdiction tool would miss this.")
    print("━" * 65)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7: Examination-Readiness Reports
    # ─────────────────────────────────────────────────────────────────────────
    section("SECTION 7: EXAMINATION-READINESS REPORTS")

    # Simulate model was last validated 8 months ago
    last_validation = date.today() - timedelta(days=240)
    psi_result = assess_psi_action(0.12, "US")  # moderate drift scenario

    print("── US Examination Readiness ───────────────────────────────────")
    us_report = generate_examination_readiness_report(
        "US", us_fairness, psi_result,
        last_validation_date=last_validation,
        has_board_approval=False,  # Not required in US — will not be flagged
    )
    print(us_report)

    psi_result_sg = assess_psi_action(0.12, "SG")

    print("\n── SG Examination Readiness ───────────────────────────────────")
    sg_report = generate_examination_readiness_report(
        "SG", sg_fairness, psi_result_sg,
        last_validation_date=last_validation,
        has_board_approval=False,  # Required in SG — will be flagged
    )
    print(sg_report)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    section("SUMMARY: JURISLENS JURISDICTION-AWARE COMPLIANCE OUTPUTS")

    print("This demonstration showed that the SAME model, applied to the SAME data,")
    print("produces DIFFERENT compliance verdicts and required actions depending on")
    print("which jurisdiction's rules are active.\n")

    print("  Dimension                      US Outcome           SG Outcome")
    print("  " + "-"*61)
    print(f"  Fairness result                {us_fairness['result']:<20} {sg_fairness['result']}")
    print(f"  Fairness violations            {len(us_fairness['violations'])} violation(s)           {len(sg_fairness['violations'])} violation(s)")
    print(f"  PSI action (drift=0.12)        ENHANCED_MONITORING  ENHANCED_MONITORING")
    print(f"  PSI action (drift=0.22)        ENHANCED_MONITORING  REVALIDATION_REQUIRED")
    print(f"  GenAI in scope                 No                   Yes")
    print(f"  SHAP in adverse action         No                   Yes")
    print(f"  Board approval required        No                   Yes")
    print(f"  Adverse notice timing          30 days              21 days")
    print()
    print("A bank operating in both jurisdictions must satisfy BOTH sets of requirements.")
    print("JurisLens makes this dual obligation explicit, auditable, and traceable.")
    print()
    print("=" * 65)
    print("  Demo complete. See Task3_model_card.md for governance documentation.")
    print("=" * 65)


if __name__ == "__main__":
    main()
