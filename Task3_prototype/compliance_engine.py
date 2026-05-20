"""
compliance_engine.py
JurisLens — Jurisdiction-Aware Credit Model Risk Tool
ClearPath RegTech Pte. Ltd.

Core compliance logic:
  - Fairness checks (jurisdiction-aware)
  - Adverse action notice generation (jurisdiction-aware)
  - Model drift / PSI monitoring
  - Examination-readiness report

The key design principle: every function that produces a compliance output
takes a jurisdiction_code argument. The same model, the same data, the same
applicant — but a different output depending on which rules are active.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Optional
from jurisdiction_config import get_config


# ─────────────────────────────────────────────────────────────────────────────
# 1. FAIRNESS CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def run_fairness_check(
    df: pd.DataFrame,
    jurisdiction_code: str,
    protected_attribute: str = "race_group",
    outcome_col: str = "model_approved",
    true_label_col: str = "true_creditworthy",
) -> dict:
    """
    Run jurisdiction-aware fairness check on model outcomes.

    US method: 4/5ths (80%) rule — compares approval rates, flags if any group
               is below 80% of the most-favoured group.

    SG method: FEAT dual metric — demographic parity (raw rate difference) AND
               equal opportunity (TPR difference). Both must be within 5pp.

    Returns a dict with:
      - jurisdiction, method, result ("PASS" / "FAIL" / "WARN")
      - per-group statistics
      - narrative explanation suitable for a compliance report
    """
    cfg = get_config(jurisdiction_code)
    fairness_cfg = cfg["fairness"]
    method = fairness_cfg["method"]

    groups = df[protected_attribute].unique()
    stats = {}
    for g in groups:
        mask = df[protected_attribute] == g
        group_df = df[mask]
        n = len(group_df)
        n_approved = group_df[outcome_col].sum()
        approval_rate = n_approved / n if n > 0 else 0.0

        # True positive rate (for equal opportunity): among truly creditworthy,
        # what fraction does the model approve?
        creditworthy_mask = group_df[true_label_col] == 1
        n_creditworthy = creditworthy_mask.sum()
        tpr = (
            group_df.loc[creditworthy_mask, outcome_col].sum() / n_creditworthy
            if n_creditworthy > 0 else np.nan
        )

        stats[g] = {
            "n": n,
            "n_approved": int(n_approved),
            "approval_rate": round(approval_rate, 4),
            "n_truly_creditworthy": int(n_creditworthy),
            "true_positive_rate": round(tpr, 4) if not np.isnan(tpr) else None,
        }

    result = {}
    violations = []

    if method == "four_fifths_rule":
        threshold = fairness_cfg["disparate_impact_threshold"]
        best_rate = max(s["approval_rate"] for s in stats.values())
        for g, s in stats.items():
            ratio = s["approval_rate"] / best_rate if best_rate > 0 else 1.0
            s["four_fifths_ratio"] = round(ratio, 4)
            s["four_fifths_pass"] = ratio >= threshold
            if not s["four_fifths_pass"]:
                violations.append({
                    "group": g,
                    "ratio": ratio,
                    "threshold": threshold,
                    "metric": "four_fifths_approval_rate",
                })

        overall_result = "PASS" if not violations else "FAIL"
        narrative = _us_fairness_narrative(stats, violations, threshold, cfg)

    elif method == "feat_dual_metric":
        dp_threshold = fairness_cfg["demographic_parity_threshold"]
        eo_threshold = fairness_cfg["equal_opportunity_threshold"]

        all_rates = [s["approval_rate"] for s in stats.values()]
        all_tprs = [s["true_positive_rate"] for s in stats.values()
                    if s["true_positive_rate"] is not None]

        # Demographic parity: check all pairs
        dp_violations = []
        eo_violations = []
        group_list = list(stats.keys())
        for i, ga in enumerate(group_list):
            for gb in group_list[i+1:]:
                dp_diff = abs(stats[ga]["approval_rate"] - stats[gb]["approval_rate"])
                stats[ga][f"dp_diff_vs_{gb}"] = round(dp_diff, 4)
                if dp_diff > dp_threshold:
                    dp_violations.append({
                        "group_a": ga, "group_b": gb,
                        "diff": round(dp_diff, 4), "threshold": dp_threshold,
                        "metric": "demographic_parity",
                    })

                # Equal opportunity
                tpr_a = stats[ga]["true_positive_rate"]
                tpr_b = stats[gb]["true_positive_rate"]
                if tpr_a is not None and tpr_b is not None:
                    eo_diff = abs(tpr_a - tpr_b)
                    if eo_diff > eo_threshold:
                        eo_violations.append({
                            "group_a": ga, "group_b": gb,
                            "diff": round(eo_diff, 4), "threshold": eo_threshold,
                            "metric": "equal_opportunity",
                        })

        violations = dp_violations + eo_violations
        overall_result = "PASS" if not violations else "FAIL"
        narrative = _sg_fairness_narrative(stats, dp_violations, eo_violations,
                                           dp_threshold, eo_threshold, cfg)
    else:
        raise ValueError(f"Unknown fairness method: {method}")

    return {
        "jurisdiction": jurisdiction_code,
        "jurisdiction_name": cfg["name"],
        "method": method,
        "protected_attribute": protected_attribute,
        "result": overall_result,
        "group_statistics": stats,
        "violations": violations,
        "narrative": narrative,
        "assessment_date": str(date.today()),
        "config_version": cfg["config_version"],
    }


def _us_fairness_narrative(stats, violations, threshold, cfg):
    lines = [
        f"FAIR LENDING ASSESSMENT — {cfg['name']} ({cfg['regulator']})",
        f"Method: Four-fifths (80%) rule under ECOA / Regulation B",
        f"Threshold: Adverse impact ratio < {threshold:.0%} triggers review",
        "",
    ]
    best_group = max(stats, key=lambda g: stats[g]["approval_rate"])
    lines.append(f"Reference group (highest approval rate): Group {best_group} "
                 f"({stats[best_group]['approval_rate']:.1%})")
    for g, s in stats.items():
        status = "✓ PASS" if s.get("four_fifths_pass", True) else "✗ FAIL"
        lines.append(f"  Group {g}: {s['approval_rate']:.1%} approval "
                     f"(ratio: {s.get('four_fifths_ratio', 1.0):.3f}) — {status}")
    if violations:
        lines.append("")
        lines.append("⚠ DISPARATE IMPACT DETECTED — examination risk elevated.")
        lines.append(f"  Note: {cfg['fairness']['note']}")
    else:
        lines.append("")
        lines.append("✓ No disparate impact detected under four-fifths rule.")
    return "\n".join(lines)


def _sg_fairness_narrative(stats, dp_violations, eo_violations, dp_threshold, eo_threshold, cfg):
    lines = [
        f"FAIRNESS ASSESSMENT — {cfg['name']} ({cfg['regulator']})",
        f"Method: MAS FEAT Dual Metric (Demographic Parity + Equal Opportunity)",
        f"Thresholds: DP diff < {dp_threshold:.0%}, EO diff < {eo_threshold:.0%}",
        "",
    ]
    for g, s in stats.items():
        lines.append(f"  Group {g}: Approval {s['approval_rate']:.1%}, "
                     f"TPR {s.get('true_positive_rate', 'N/A')}")
    lines.append("")
    if dp_violations:
        lines.append("✗ DEMOGRAPHIC PARITY VIOLATIONS:")
        for v in dp_violations:
            lines.append(f"   Groups {v['group_a']} vs {v['group_b']}: "
                         f"diff = {v['diff']:.1%} (threshold: {v['threshold']:.0%})")
    else:
        lines.append("✓ Demographic parity: no violations")

    if eo_violations:
        lines.append("✗ EQUAL OPPORTUNITY VIOLATIONS:")
        for v in eo_violations:
            lines.append(f"   Groups {v['group_a']} vs {v['group_b']}: "
                         f"TPR diff = {v['diff']:.1%} (threshold: {v['threshold']:.0%})")
    else:
        lines.append("✓ Equal opportunity: no violations")

    if dp_violations or eo_violations:
        lines.append("")
        lines.append("⚠ FEAT FAIRNESS REQUIREMENTS NOT MET.")
        lines.append("   Board attestation cannot be issued until violations are remediated.")
        lines.append("   Required action: model re-training or post-processing adjustment,")
        lines.append("   with documented methodology, within next validation cycle.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADVERSE ACTION NOTICE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_adverse_action_notice(
    applicant_row: pd.Series,
    shap_values: dict,
    jurisdiction_code: str,
) -> str:
    """
    Generate a jurisdiction-specific adverse action notice for a denied applicant.

    Args:
        applicant_row: Row from the applications DataFrame (denied applicant).
        shap_values: Dict of {feature_name: shap_value} for this applicant.
                     Negative SHAP values = features that REDUCED approval probability.
        jurisdiction_code: 'US' or 'SG'

    Returns:
        A formatted string notice. In production this would render to PDF.
    """
    cfg = get_config(jurisdiction_code)
    aa_cfg = cfg["adverse_action"]

    # Sort features by impact magnitude (most negative first = biggest negative contributors)
    sorted_features = sorted(shap_values.items(), key=lambda x: x[1])
    adverse_features = [(f, v) for f, v in sorted_features if v < 0]

    # Friendly feature names
    feature_labels = {
        "credit_score": "Credit score",
        "debt_to_income": "Debt-to-income ratio",
        "income_sgd": "Income level",
        "employment_years": "Length of employment",
        "has_collateral": "Availability of collateral",
        "loan_amount_sgd": "Loan amount requested",
    }

    notice_date = date.today().strftime("%d %B %Y")

    # ── Common header ──────────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        f"NOTICE OF CREDIT DECISION — JURISDICTION: {jurisdiction_code}",
        "=" * 60,
        f"Date: {notice_date}",
        f"Applicant ID: {applicant_row.get('applicant_id', 'N/A')}",
        f"Framework: {aa_cfg['framework']}",
        "",
        "DECISION: APPLICATION NOT APPROVED",
        "",
    ]

    if jurisdiction_code == "US":
        # ── US Reg B 2026: principal reasons, capped at 4, no SHAP disclosure ──
        max_reasons = aa_cfg["max_principal_reasons"]
        lines.append(aa_cfg["sample_required_language"])
        lines.append("")
        lines.append(f"PRINCIPAL REASONS FOR ADVERSE ACTION (max {max_reasons}):")
        lines.append("")

        for i, (feat, val) in enumerate(adverse_features[:max_reasons], 1):
            label = feature_labels.get(feat, feat.replace("_", " ").title())
            lines.append(f"  {i}. {label}")

        lines.append("")
        lines.append("You may request a statement of specific reasons for this decision")
        lines.append("within 60 days of receiving this notice.")
        lines.append("")
        lines.append("NOTE FOR COMPLIANCE RECORDS (not disclosed to applicant):")
        lines.append(f"  SHAP disclosure required under this framework: "
                     f"{aa_cfg['requires_shap_or_lime_disclosure']}")
        lines.append(f"  Digital redlining guidance active: "
                     f"{aa_cfg.get('digital_redlining_guidance_active', False)}")

    elif jurisdiction_code == "SG":
        # ── SG MAS FEAT 2024: full SHAP disclosure, human review right ─────────
        lines.append(aa_cfg["sample_required_language"])
        lines.append("")
        lines.append("FACTORS INFLUENCING THIS DECISION (in order of impact):")
        lines.append("")
        lines.append("The following factors, assessed by an automated model,")
        lines.append("contributed to this outcome. Factors are listed with their")
        lines.append("relative impact (SHAP contribution score).")
        lines.append("")

        for i, (feat, val) in enumerate(adverse_features, 1):
            label = feature_labels.get(feat, feat.replace("_", " ").title())
            # In SG, SHAP values are disclosed — directional explanation
            direction = "reduced" if val < 0 else "increased"
            lines.append(f"  {i}. {label}: {direction} approval likelihood "
                         f"(impact score: {abs(val):.3f})")

        lines.append("")
        lines.append("YOUR RIGHTS UNDER MAS GUIDELINES:")
        lines.append("  • You may request a human review of this decision within 30 days.")
        lines.append("  • You may request additional explanation of any factor listed above.")
        lines.append("  • This decision was made using a model governed under the")
        lines.append("    MAS Principles for Responsible Use of AI (FEAT).")
        lines.append("")
        lines.append("DATA USAGE NOTICE (PDPA):")
        lines.append("  Your personal data was processed for the purpose of this credit")
        lines.append("  assessment under the Personal Data Protection Act 2012.")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PSI — POPULATION STABILITY INDEX
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between baseline and current score distributions.

    PSI < 0.10: No significant change (stable)
    PSI 0.10–0.25: Moderate change (monitor)
    PSI > 0.25: Major shift (revalidation required)

    These thresholds are hardcoded as they are consistent across US and SG rules.
    The *trigger action* at each threshold differs by jurisdiction (see jurisdiction_config).
    """
    # Create bins based on baseline
    breakpoints = np.percentile(baseline_scores, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    baseline_counts = np.histogram(baseline_scores, bins=breakpoints)[0]
    current_counts = np.histogram(current_scores, bins=breakpoints)[0]

    # Avoid divide-by-zero
    baseline_pct = (baseline_counts / len(baseline_scores)).clip(min=1e-6)
    current_pct = (current_counts / len(current_scores)).clip(min=1e-6)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return round(float(psi), 5)


def assess_psi_action(psi_value: float, jurisdiction_code: str) -> dict:
    """
    Interpret PSI value in context of jurisdiction-specific trigger thresholds.
    """
    cfg = get_config(jurisdiction_code)
    mrm = cfg["model_risk"]

    reval_trigger = mrm["psi_revalidation_trigger"]
    monitor_trigger = mrm["psi_monitoring_trigger"]

    if psi_value >= reval_trigger:
        action = "REVALIDATION_REQUIRED"
        description = (
            f"PSI {psi_value:.4f} exceeds revalidation threshold {reval_trigger}. "
            f"Model must be formally revalidated before continued use. "
            f"({jurisdiction_code}: {cfg['name']})"
        )
    elif psi_value >= monitor_trigger:
        action = "ENHANCED_MONITORING"
        description = (
            f"PSI {psi_value:.4f} in monitoring zone [{monitor_trigger}, {reval_trigger}). "
            f"Increase monitoring frequency. Notify model owner. Document rationale "
            f"for continued use. ({jurisdiction_code}: {cfg['name']})"
        )
    else:
        action = "STABLE"
        description = (
            f"PSI {psi_value:.4f} below monitoring threshold {monitor_trigger}. "
            f"Model population is stable. Continue scheduled monitoring. "
            f"({jurisdiction_code}: {cfg['name']})"
        )

    return {"psi": psi_value, "action": action, "description": description}


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXAMINATION-READINESS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_examination_readiness_report(
    jurisdiction_code: str,
    fairness_result: dict,
    psi_result: dict,
    last_validation_date: date,
    has_model_inventory: bool = True,
    has_board_approval: bool = None,  # None = not checked yet
) -> str:
    """
    Generate a structured examination-readiness report for the given jurisdiction.
    This is the output a CCO would review before a regulatory examination.
    """
    cfg = get_config(jurisdiction_code)
    checklist = cfg["reporting"]["examination_readiness_checklist"]
    today = date.today()
    validation_age_months = (today - last_validation_date).days / 30

    lines = [
        "=" * 65,
        f"EXAMINATION READINESS REPORT",
        f"Jurisdiction: {cfg['name']} ({cfg['regulator']})",
        f"Config version: {cfg['config_version']}",
        f"Report generated: {today}",
        "=" * 65,
        "",
        "── MANDATORY CHECKLIST ────────────────────────────────────────",
    ]

    checks_passed = 0
    total_checks = len(checklist)

    for item in checklist:
        # We do a simplified check — in production, each item would be
        # programmatically verified against the model inventory database
        lines.append(f"  [ ] {item}")

    lines.extend([
        "",
        "── AUTOMATED CHECKS ───────────────────────────────────────────",
        "",
    ])

    # Validation age check
    req_months = cfg["model_risk"]["validation_frequency_months"]
    val_status = "✓ PASS" if validation_age_months <= req_months else "✗ FAIL"
    lines.append(f"  Model validation age: {validation_age_months:.1f} months "
                 f"(max {req_months} months) — {val_status}")

    # Fairness check
    fair_status = "✓ PASS" if fairness_result["result"] == "PASS" else "✗ FAIL"
    lines.append(f"  Fairness ({fairness_result['method']}): "
                 f"{fairness_result['result']} — {fair_status}")

    # PSI check
    psi_status = "✓" if psi_result["action"] == "STABLE" else (
        "⚠" if psi_result["action"] == "ENHANCED_MONITORING" else "✗")
    lines.append(f"  Population Stability Index: {psi_result['psi']:.4f} "
                 f"→ {psi_result['action']} — {psi_status}")

    # Board approval (SG only)
    if cfg["model_risk"]["requires_board_approval_for_deployment"]:
        board_status = ("✓ PASS" if has_board_approval else
                        "✗ FAIL" if has_board_approval is False else "? NOT CHECKED")
        lines.append(f"  Board approval on file: {board_status}")

    # GenAI scope note
    if cfg["model_risk"]["genai_in_scope"]:
        lines.append(f"  ⚠ GenAI models ARE in scope for this jurisdiction. "
                     f"Confirm GenAI inventory is complete.")
    else:
        lines.append(f"  ℹ GenAI models are NOT in scope for {jurisdiction_code} "
                     f"under current regulations (OCC 2026-13). "
                     f"Note: this differs from SG requirements.")

    lines.extend([
        "",
        "── OVERALL ASSESSMENT ─────────────────────────────────────────",
        "",
    ])

    critical_failures = []
    if fairness_result["result"] == "FAIL":
        critical_failures.append("Fairness check failed")
    if psi_result["action"] == "REVALIDATION_REQUIRED":
        critical_failures.append("Model requires revalidation")
    if validation_age_months > req_months:
        critical_failures.append("Model validation overdue")
    if cfg["model_risk"]["requires_board_approval_for_deployment"] and not has_board_approval:
        critical_failures.append("Board approval not on file")

    if critical_failures:
        lines.append(f"  STATUS: NOT EXAMINATION-READY")
        lines.append(f"  Critical issues ({len(critical_failures)}):")
        for f in critical_failures:
            lines.append(f"    • {f}")
    else:
        lines.append(f"  STATUS: EXAMINATION-READY (pending manual checklist completion)")

    lines.append("")
    lines.append("=" * 65)
    return "\n".join(lines)
