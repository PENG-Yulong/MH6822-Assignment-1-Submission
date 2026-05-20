"""
jurisdiction_config.py
JurisLens — Jurisdiction-Aware Credit Model Risk Tool
ClearPath RegTech Pte. Ltd.

This module is the core of the jurisdiction rule engine. Regulatory parameters are
stored as versioned, structured dictionaries. The key design principle: every parameter
that differs between jurisdictions must be traceable to a specific regulatory instrument.
"""

from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# JURISDICTION RULE REGISTRY
# Each entry is: config name → parameters + regulatory citations
# ─────────────────────────────────────────────────────────────────────────────

JURISDICTION_CONFIGS = {

    "US": {
        "name": "United States",
        "regulator": "OCC / CFPB / Federal Reserve",
        "config_version": "2026.04",
        "effective_date": date(2026, 4, 1),
        "last_reviewed": date(2026, 5, 15),
        "reviewer": "Peng Yulong",

        # ── Model Risk Management ──────────────────────────────────────────────
        "model_risk": {
            # OCC Bulletin 2026-13: GenAI explicitly excluded from MRM scope.
            # Political choice: Trump administration deregulatory posture.
            # Citation: https://www.occ.treas.gov/news-issuances/bulletins/2026/bulletin-2026-13.html
            "genai_in_scope": False,
            "validation_frequency_months": 12,
            "material_model_threshold_annual_decisions": 10_000,
            "requires_board_approval_for_deployment": False,  # Senior mgmt sufficient
            "requires_independent_validation": True,
            "model_inventory_required": True,
            # Stress testing: OCC expects scenario analysis for credit models
            "stress_testing_required": True,
            # PSI (Population Stability Index) trigger for revalidation
            "psi_revalidation_trigger": 0.25,   # "> 0.25 = major shift"
            "psi_monitoring_trigger": 0.10,      # "0.10-0.25 = moderate"
        },

        # ── Fairness / Fair Lending ────────────────────────────────────────────
        "fairness": {
            # ECOA / Regulation B: "Four-fifths rule" (80% rule) for disparate impact
            # Citation: 12 CFR Part 1002 (Regulation B)
            "method": "four_fifths_rule",
            # If protected group approval rate < 80% of most-favoured group → flag
            "disparate_impact_threshold": 0.80,
            "protected_attributes": ["race", "sex", "age_group", "national_origin",
                                     "marital_status", "receipt_of_public_assistance"],
            # The "reference group" is whichever group has the highest approval rate
            "reference_group_method": "most_favored",
            # Under Trump-era CFPB, disparate-impact theory enforcement is reduced
            # but the statutory obligation under ECOA remains
            "disparate_impact_enforcement_risk": "REDUCED",
            "note": ("CFPB 2026: disparate-impact enforcement deprioritised. "
                     "Statutory ECOA obligation remains. Flag for monitoring, "
                     "not automatic violation."),
        },

        # ── Adverse Action Notices ─────────────────────────────────────────────
        "adverse_action": {
            # CFPB Regulation B (2026 rewrite) — finalised largely as proposed.
            # Simplified from Biden-era AI guidance.
            # Citation: https://www.consumerfinancialserviceslawmonitor.com/2026/04/...
            "framework": "Regulation_B_2026",
            "max_principal_reasons": 4,
            "requires_specific_ai_factor_disclosure": False,  # Simplified under 2026 rule
            "requires_shap_or_lime_disclosure": False,
            "notice_timing_days": 30,
            "explainability_standard": "principal_reasons_sufficient",
            "sample_required_language": (
                "Your application for credit was denied. "
                "The principal reasons for this decision are listed below. "
                "You have the right to a statement of specific reasons "
                "within 60 days of this notice."
            ),
            "must_reference_credit_bureau": True,
            "digital_redlining_guidance_active": False,  # Biden-era guidance rescinded
        },

        # ── Reporting Obligations ──────────────────────────────────────────────
        "reporting": {
            "hmda_required": True,
            "hmda_data_fields": ["loan_type", "loan_amount", "race", "sex",
                                 "income", "census_tract", "action_taken"],
            "cra_considerations": True,
            "model_risk_report_frequency_months": 12,
            "board_report_required": False,  # Senior management report sufficient
            "examination_readiness_checklist": [
                "Model inventory current",
                "Validation reports < 12 months old",
                "Adverse action notices retained 25 months",
                "HMDA LAR submitted",
                "Fair lending monitoring documented",
            ],
        },
    },


    "SG": {
        "name": "Singapore",
        "regulator": "Monetary Authority of Singapore (MAS)",
        "config_version": "2024.11",
        "effective_date": date(2024, 11, 1),
        "last_reviewed": date(2026, 5, 15),
        "reviewer": "Peng Yulong",

        # ── Model Risk Management ──────────────────────────────────────────────
        "model_risk": {
            # MAS 2024 AI Model Risk Management paper: ALL AI in scope including GenAI.
            # Political choice: MAS takes the view that novelty increases governance need.
            # Citation: MAS 2024 AI MRM Consultation Paper
            "genai_in_scope": True,
            "validation_frequency_months": 6,   # More frequent than US
            "material_model_threshold_annual_decisions": 5_000,  # Lower bar
            "requires_board_approval_for_deployment": True,   # Board sign-off required
            "requires_independent_validation": True,
            "model_inventory_required": True,
            "stress_testing_required": True,
            "psi_revalidation_trigger": 0.20,   # Tighter than US
            "psi_monitoring_trigger": 0.10,
        },

        # ── Fairness / FEAT Principles ─────────────────────────────────────────
        "fairness": {
            # MAS FEAT Principles (Fairness, Ethics, Accountability, Transparency)
            # Dual metric approach: demographic parity + equal opportunity
            # Citation: MAS FEAT Principles 2018, operationalised in MAS 2024 guidance
            "method": "feat_dual_metric",
            # Demographic parity: |P(approve|group_A) - P(approve|group_B)| < threshold
            "demographic_parity_threshold": 0.05,  # 5% maximum difference
            # Equal opportunity: |TPR_group_A - TPR_group_B| < threshold
            "equal_opportunity_threshold": 0.05,
            "protected_attributes": ["race", "gender", "age_group", "disability_status"],
            "reference_group_method": "all_pairs",  # All group pairs compared
            "requires_board_attestation": True,
            "fairness_review_frequency_months": 6,
            "note": ("MAS FEAT requires documented fairness methodology, "
                     "board attestation, and MAS examination-ready evidence. "
                     "Applies to all AI models including GenAI components."),
        },

        # ── Adverse Action Notices ─────────────────────────────────────────────
        "adverse_action": {
            # MAS FEAT Transparency Principle + MAS 2024 AI MRM
            # Must be explainable to the individual in terms they understand.
            "framework": "MAS_FEAT_2024",
            "max_principal_reasons": None,  # Must be comprehensive, not capped
            "requires_specific_ai_factor_disclosure": True,
            "requires_shap_or_lime_disclosure": True,   # Explainability mandatory
            "notice_timing_days": 21,   # Tighter than US
            "explainability_standard": "full_audit_trail_and_consumer_explanation",
            "sample_required_language": (
                "Your credit application has been assessed using an automated system. "
                "The following factors, in order of their impact on this decision, "
                "influenced the outcome. You have the right to request a human review "
                "of this decision within 30 days."
            ),
            "must_reference_credit_bureau": True,
            "human_review_right": True,   # Required under MAS guidance
            "pdpa_disclosure_required": True,  # Personal Data Protection Act
        },

        # ── Reporting Obligations ──────────────────────────────────────────────
        "reporting": {
            "hmda_required": False,  # US-specific
            "mas_model_inventory_required": True,
            "mas_annual_review_required": True,
            "board_report_required": True,
            "board_report_frequency_months": 3,  # Quarterly board reporting
            "model_risk_report_frequency_months": 6,
            "pdpa_compliance_required": True,
            "examination_readiness_checklist": [
                "MAS model inventory current and board-approved",
                "Validation reports < 6 months old",
                "FEAT fairness assessment documented with board attestation",
                "Adverse action notices include SHAP/explainability output",
                "PDPA data usage documented for each training dataset",
                "GenAI models inventoried and governed",
                "Quarterly board model risk report produced",
            ],
        },
    },
}


def get_config(jurisdiction_code: str) -> dict:
    """
    Retrieve jurisdiction configuration by code.

    Args:
        jurisdiction_code: 'US' or 'SG' (extensible)

    Returns:
        Configuration dictionary for that jurisdiction.

    Raises:
        ValueError if the jurisdiction code is not registered.
    """
    code = jurisdiction_code.upper().strip()
    if code not in JURISDICTION_CONFIGS:
        available = list(JURISDICTION_CONFIGS.keys())
        raise ValueError(
            f"Jurisdiction '{code}' not found. "
            f"Available: {available}. "
            f"To add a new jurisdiction, extend JURISDICTION_CONFIGS in jurisdiction_config.py "
            f"with a complete parameter set and regulatory citations."
        )
    return JURISDICTION_CONFIGS[code]


def compare_jurisdictions(code_a: str, code_b: str) -> dict:
    """
    Return a structured comparison of two jurisdiction configs, highlighting divergences.
    This is the output that a CCO would use to understand dual-jurisdiction exposure.
    """
    cfg_a = get_config(code_a)
    cfg_b = get_config(code_b)

    divergences = []

    # GenAI scope
    if cfg_a["model_risk"]["genai_in_scope"] != cfg_b["model_risk"]["genai_in_scope"]:
        divergences.append({
            "dimension": "GenAI model scope",
            code_a: f"{'IN' if cfg_a['model_risk']['genai_in_scope'] else 'OUT OF'} scope",
            code_b: f"{'IN' if cfg_b['model_risk']['genai_in_scope'] else 'OUT OF'} scope",
            "risk": "HIGH — models governed in one jurisdiction may be ungoverned in the other",
        })

    # Validation frequency
    freq_a = cfg_a["model_risk"]["validation_frequency_months"]
    freq_b = cfg_b["model_risk"]["validation_frequency_months"]
    if freq_a != freq_b:
        divergences.append({
            "dimension": "Validation frequency",
            code_a: f"Every {freq_a} months",
            code_b: f"Every {freq_b} months",
            "risk": "MEDIUM — model may be compliant in lenient jurisdiction but non-compliant in stricter one",
        })

    # Board approval
    if (cfg_a["model_risk"]["requires_board_approval_for_deployment"] !=
            cfg_b["model_risk"]["requires_board_approval_for_deployment"]):
        divergences.append({
            "dimension": "Board approval for model deployment",
            code_a: str(cfg_a["model_risk"]["requires_board_approval_for_deployment"]),
            code_b: str(cfg_b["model_risk"]["requires_board_approval_for_deployment"]),
            "risk": "MEDIUM — governance process must be jurisdiction-specific",
        })

    # Fairness method
    if cfg_a["fairness"]["method"] != cfg_b["fairness"]["method"]:
        divergences.append({
            "dimension": "Fairness measurement method",
            code_a: cfg_a["fairness"]["method"],
            code_b: cfg_b["fairness"]["method"],
            "risk": ("HIGH — a model may pass the 4/5ths rule but fail demographic "
                     "parity; different methods can produce different compliance verdicts "
                     "for the same model"),
        })

    # Explainability
    if (cfg_a["adverse_action"]["requires_shap_or_lime_disclosure"] !=
            cfg_b["adverse_action"]["requires_shap_or_lime_disclosure"]):
        divergences.append({
            "dimension": "SHAP/LIME explainability in adverse action notices",
            code_a: str(cfg_a["adverse_action"]["requires_shap_or_lime_disclosure"]),
            code_b: str(cfg_b["adverse_action"]["requires_shap_or_lime_disclosure"]),
            "risk": ("HIGH — adverse action notices generated for one jurisdiction "
                     "are legally insufficient in the other"),
        })

    return {
        "jurisdiction_a": cfg_a["name"],
        "jurisdiction_b": cfg_b["name"],
        "divergence_count": len(divergences),
        "divergences": divergences,
    }


if __name__ == "__main__":
    import json
    comparison = compare_jurisdictions("US", "SG")
    print(f"\nJurisdiction Divergence Report: {comparison['jurisdiction_a']} vs {comparison['jurisdiction_b']}")
    print(f"Number of material divergences identified: {comparison['divergence_count']}\n")
    for i, d in enumerate(comparison["divergences"], 1):
        print(f"  [{i}] {d['dimension']}")
        print(f"       US: {d.get('US', d.get('jurisdiction_a', ''))}")
        print(f"       SG: {d.get('SG', d.get('jurisdiction_b', ''))}")
        print(f"       Risk: {d['risk']}")
        print()
