"""
Business Decision Rules -- bridges the trained ML model and raw dataset
statistics with human-readable credit policy, for review by non-technical
credit risk officers (not just data scientists).

Two tiers, matching standard credit-policy design:

- Tier 1 (fixed thresholds / "Option B"): hard knockout rules on key
  features (EXT_SOURCE, Annuity-to-Income, Credit-to-Goods, etc.), defined
  directly against business-chosen cutoffs in config.py. Non-negotiable --
  applied before the model is even consulted.

- Tier 2 (surrogate decision tree / "Option A"): a shallow
  DecisionTreeClassifier is fit on the trained model's own top-risk-quantile
  predictions, so its splits approximate *why* the model calls something
  High Risk. Rules are read out two ways:
    1. `extract_surrogate_rules` -- structured dicts (condition, support,
       coverage, precision) for a sortable table in the UI.
    2. `get_readable_tree_text` -- the literal `sklearn.tree.export_text`
       dump, for a technical/audit view of the exact IF-THEN tree structure.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, _tree

from src.utils.config import config
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tier 1: fixed business-threshold rules (human-readable definitions only --
# the actual pass/fail check against an applicant lives in
# src.ml.predict.hard_knockout_checks, so there is a single source of truth
# for the thresholds themselves: config.py. This function exists purely to
# render those same thresholds as plain-English policy statements for the UI.)
# ---------------------------------------------------------------------------

def get_hard_knockout_rule_definitions():
    """Returns the Tier 1 policy gate as a list of plain-English rule
    strings, sourced from the same config constants the actual applicant
    check (src.ml.predict.hard_knockout_checks) uses -- so the UI can never
    drift out of sync with what's actually enforced."""
    return [
        f"IF Annuity-to-Income ratio > {config.KO_MAX_ANNUITY_TO_INCOME*100:.0f}% "
        f"THEN REJECT (unaffordable debt-service obligation)",

        f"IF Income Type = 'Unemployed' THEN REJECT (no verified income source)",

        f"IF Credit Amount > {config.KO_MAX_CREDIT_TO_GOODS_RATIO*100:.0f}% of Goods Price "
        f"THEN REJECT (severe over-financing / under-collateralized lending)",

        f"IF mean External Bureau Score < {config.KO_MIN_EXT_SOURCE_MEAN} "
        f"THEN REJECT (critically low third-party credit signal)",
    ]


# ---------------------------------------------------------------------------
# Tier 2: surrogate decision tree fit on the model's own High-Risk calls
# ---------------------------------------------------------------------------

def _fit_surrogate_tree(model, X_val, max_depth=None, min_support=None):
    """Shared fitting step used by both Tier-2 extraction functions below,
    so the structured-rules table and the raw export_text view are always
    reading the exact same tree."""
    max_depth = max_depth or config.SURROGATE_TREE_MAX_DEPTH
    min_support = min_support or config.SURROGATE_MIN_SUPPORT

    surrogate_X = X_val.copy()
    for c in config.CATEGORICAL_COLS:
        if c in surrogate_X.columns:
            surrogate_X[c] = surrogate_X[c].cat.codes

    model_pred = model.predict_proba(X_val)[:, 1]
    high_risk_label = (model_pred >= np.quantile(model_pred, config.SURROGATE_HIGH_RISK_QUANTILE)).astype(int)

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_support, random_state=config.RANDOM_SEED)
    tree.fit(surrogate_X, high_risk_label)
    return tree, surrogate_X, high_risk_label


def extract_surrogate_rules(model, X_val, max_depth: int = None, min_support: int = None):
    """Tier 2 business rules, as structured dicts for a sortable UI table:
    fits a shallow decision tree on the model's own top-risk-quantile
    predictions (a surrogate), then reads its splits out as IF-THEN business
    rules, each audited against real outcomes for support/precision/coverage."""
    tree, surrogate_X, _ = _fit_surrogate_tree(model, X_val, max_depth, min_support)
    min_support = min_support or config.SURROGATE_MIN_SUPPORT

    rules = []
    tree_ = tree.tree_
    feature_names = surrogate_X.columns

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.2f}"])
        else:
            value = tree_.value[node][0]
            support = int(tree_.n_node_samples[node])
            if support < min_support:
                return
            total = value.sum()
            leaf_high_risk_rate = (value[1] / total) if total else 0
            if leaf_high_risk_rate < 0.5:
                return
            coverage = support / len(surrogate_X)
            rules.append({
                "condition": " AND ".join(conditions) if conditions else "(root)",
                "support": support,
                "coverage_pct": round(coverage * 100, 2),
                "precision_pct": round(leaf_high_risk_rate * 100, 2),
            })

    recurse(0, [])
    rules.sort(key=lambda r: -r["precision_pct"])
    log.info("Extracted %d surrogate business rules", len(rules))
    return rules


def get_readable_rules_bullets(model, X_val, max_depth: int = None, min_support: int = None):
    """Same Tier 2 rules as `extract_surrogate_rules`, formatted as plain-
    English IF-THEN bullet strings for non-technical credit risk officers."""
    rules = extract_surrogate_rules(model, X_val, max_depth, min_support)
    bullets = []
    for r in rules:
        bullets.append(
            f"IF {r['condition']} THEN Flag as HIGH RISK "
            f"(covers {r['coverage_pct']}% of applicants, {r['precision_pct']}% precision, n={r['support']})"
        )
    return bullets


def get_readable_tree_text(model, X_val, max_depth: int = None, min_support: int = None) -> str:
    """The literal `sklearn.tree.export_text` dump of the Tier 2 surrogate
    tree -- an exact, technical IF-THEN representation of the tree structure,
    for audit/technical review alongside the plain-English bullets above."""
    tree, surrogate_X, _ = _fit_surrogate_tree(model, X_val, max_depth, min_support)
    return export_text(tree, feature_names=list(surrogate_X.columns), decimals=2)
