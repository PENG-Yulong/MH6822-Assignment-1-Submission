# Task 3 — Model Card and Governance Documentation

## JurisLens Credit Compliance Engine

**Tool:** JurisLens
**Developed by:** ClearPath RegTech Pte. Ltd. (Virtual)  
**Card last updated:** May 2026  
**Card author:** Peng Yulong



## 1\. Tool Overview

JurisLens is a jurisdiction-aware compliance tool for AI-driven credit scoring. It takes a trained credit scoring model (not itself a credit model), a dataset of applications, and a jurisdiction code (currently US or SG), and produces:

1. A fairness assessment using the rules applicable in that jurisdiction
2. Jurisdiction-specific adverse action notices for denied applicants
3. A population stability index (PSI) drift assessment with jurisdiction-calibrated triggers
4. An examination-readiness report cross-referenced against the applicable regulatory checklist

**What it is not:** JurisLens does not make credit decisions. It does not replace legal advice. It does not determine whether a particular model's design is discriminatory, but reports observable statistical disparities and required actions under the specified jurisdiction's rules.



## 2\. Intended Use

**Primary users:** Chief Compliance Officers (CCOs), model risk managers, and model validation teams at financial institutions with credit operations in the United States and/or Singapore.

**Intended use cases:**

|Use case|Supported?|
|-|-|
|Pre-deployment fairness check for a credit model|✓ Yes|
|Ongoing monitoring of a deployed credit model|✓ Yes (PSI monitoring)|
|Generation of examination-ready fairness reports|✓ Yes|
|Generation of adverse action notices|✓ Yes (template; review by legal required)|
|Making credit decisions|✗ No|
|Assessing non-credit AI models|✗ Not in this version|
|EU AI Act compliance|✗ Not in this version|
|AML/KYC compliance|✗ Not in this version|



## 3\. The Jurisdiction Configuration Layer

The jurisdiction rule engine ('\_config.py') encodes regulatory parameters as versioned structured dictionaries. Each parameter is annotated with:

* The regulatory instrument it derives from
* The effective date
* A notation where the parameter reflects a political choice (not just a technical one)

**Versioning rationale:** A compliance officer must be able to reconstruct what rules were in force when any specific fairness assessment or adverse action notice was generated. Parameters are stored with a 'config\_version' field and 'effective\_date'. When rules change, a new version is added and old versions are retained.

**Current jurisdictions supported:** US (OCC/CFPB), Singapore (MAS)  
**Jurisdictions not yet supported:** EU (AI Act), UK (FCA Consumer Duty), Hong Kong (HKMA), Canada (OSFI E-23 2027), Australia (APRA)



## 4\. Data Requirements

|Input|Required|Description|
|-|-|-|
|Applications DataFrame|Yes|One row per application; financial features + demographics|
|Protected attribute columns|Yes|Must include at least one of: race\_group, gender, age\_group|
|True creditworthiness labels|For equal-opportunity check|Ground truth labels (not always available in practice)|
|Model scores (continuous)|For PSI monitoring|Output probability from the credit model|
|SHAP values|For SG adverse action notices|Pre-computed; tool does not recompute SHAP internally in production|

**What the tool does NOT require:** Raw personal data beyond what is needed for the assessment. In production, personal identifiers would be pseudonymised before reaching JurisLens.



## 5\. Fairness Methodology

### US: Four-Fifths (80%) Rule

The four-fifths rule compares each protected group's approval rate to the highest approval rate among all groups. If any group's rate falls below 80% of the best group, disparate impact is flagged.

**Limitation:** The four-fifths rule uses approval rates, not creditworthiness-adjusted rates. A model can pass the four-fifths rule while still systematically denying creditworthy applicants from a protected group, which is exactly what the equal-opportunity metric (used in Singapore) is designed to detect.

**Political context:** The Trump-era CFPB has signalled reduced enforcement of disparate-impact theory. Our tool flags the violation with a reduced-severity rating and explanatory note, rather than treating it as an automatic regulatory breach. This reflects the current enforcement environment but not the statutory obligation, which remains.

### Singapore: MAS FEAT Dual Metric

The tool computes two metrics across all pairs of protected groups:

* **Demographic parity difference:** |Approval rate A − Approval rate B| < 5%
* **Equal opportunity difference:** |True positive rate A − True positive rate B| < 5%

Both must pass. A model can pass demographic parity while failing equal opportunity if, say, the group with a lower overall approval rate has a disproportionately low creditworthy-but-denied fraction.

**Limitation:** The 5% threshold is our operationalisation of MAS guidance. MAS does not specify a single numeric threshold; it expects documented methodology. A different institution could choose a different threshold. Users should document their rationale and have it reviewed by their compliance function.

\---

## 6\. Failure Modes and Known Limitations

### 6.1 False Negatives in Fairness Detection

If the protected attribute distribution is skewed (one group constitutes < 5% of applications), statistical power is insufficient to detect disparate impact reliably. The tool does not warn about small group sizes in this version.

**Impact:** A real disparity affecting a small group may be missed.  
**Mitigation planned:** Flag assessments where any group N < 50.

### 6.2 Jurisdictional Misconfiguration

If a user assigns the wrong jurisdiction code to a model assessment (e.g., assessing a SG model under US rules), all downstream outputs (fairness verdicts, adverse action notices, examination reports) will be incorrect.

**Impact:** An institution could believe a SG-deployed model is compliant when it is not.  
**Mitigation:** Mandatory jurisdiction confirmation at session start; audit log entry per assessment; UI warning for dual-jurisdiction institutions.

### 6.3 Regulatory Rule Lag

Regulations change. If 'jurisdiction\_config.py' is not updated when a rule changes, the tool will continue applying stale parameters silently.

**Impact:** Compliance assessments may not reflect current law.  
**Mitigation:** Each config entry has a 'last\_reviewed' date. Production version would include automated alerts when configs are > 90 days since last review, tied to a regulatory change monitoring feed.

### 6.4 PSI Thresholds Are Not Universal

The standard PSI thresholds (0.10, 0.25) are industry conventions, not regulatory mandates. MAS and OCC do not specify exact PSI thresholds. We have hardcoded the revalidation triggers at 0.25 (US) and 0.20 (SG) based on conservative interpretation of guidance documents. A prudent institution should calibrate these thresholds to their own model risk appetite.

### 6.5 Adverse Action Notices Are Templates, Not Legal Documents

The notices generated are templates. They must be reviewed by qualified legal counsel before issuance to applicants. The tool does not account for state-level variations within the US (e.g., California CCPA requirements, New York fair credit laws).

### 6.6 Equal Opportunity Requires Ground Truth

The equal-opportunity fairness metric requires knowing which applicants were truly creditworthy — a label that only becomes available at loan maturation (months or years later). In practice, the tool would use a proxy (e.g., credit bureau score decile, prior loan performance) as the ground truth proxy. The quality of the equal-opportunity assessment depends on the quality of this proxy. This is documented but not solved.

\---

## 7\. What the Tool Does Not Do — Design Boundary Choices

|Feature not included|Reason|
|-|-|
|Regulatory checkbox completion tracker|We believe this trains CCOs to optimise for documentation rather than substance. See Task 2, Question 3.|
|EU AI Act compliance module|Scope decision; EU compliance cycle is long and the Act is still being operationalised.|
|GenAI model assessment|Not needed for US under OCC 2026-13; scoped for SG.|
|Consumer-facing portal|We serve the CCO, not the applicant. A consumer portal would require different legal and UX design.|
|Automatic model retraining|Retraining decisions require human judgement. The tool flags; humans act.|
|HMDA LAR filing|Would require integration with core banking systems.|

\---

## 8\. Intended Enhancements (if time/data permitted)

1. **Conditional SHAP stability monitoring:** Track SHAP rank-order stability by income quartile over time. Detects subgroup-specific drift before aggregate metrics catch it.
2. **OSFI E-23 2027 (Canada) jurisdiction:** The 2027 effective date makes this a near-term priority.
3. **FCA Consumer Duty outcome monitoring:** UK module using FCA's four-outcome framework.
4. **Automated regulatory change alerts:** Subscription feed tying config version updates to regulatory instrument publications.
5. **Interactive sensitivity analysis:** How much would the approval rate disparity need to change before the fairness verdict flips? Visualised as a threshold frontier.

