"""
AI-Powered Credit Risk Intelligence Platform -- Streamlit UI entrypoint.

Run with:   streamlit run app.py
(or via Docker: docker compose up --build)

On first run, if no trained model artifact exists in models/, this trains
one automatically (cached) so `docker compose up` works with a single
command. To pre-train explicitly instead, run:  python -m src.ml.train
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap

from src.utils.config import config
from src.utils.docker_utils import get_model_path
from src.data.loader import load_raw_data
from src.data.preprocessor import engineer_features
from src.ml.train import train_model, save_model_artifacts, load_model_artifacts
from src.ml.predict import load_model, pd_to_score, risk_band, expected_loss
from src.ml.rules import (
    get_hard_knockout_rule_definitions, extract_surrogate_rules,
    get_readable_rules_bullets, get_readable_tree_text,
)
from src.ml.predict import hard_knockout_checks
from src.talk_to_data.query_runner import build_duckdb_connection, answer_question, OFFLINE_QUERY_LIBRARY
from src.talk_to_data.prompt_templates import semantic_schema_prompt_block, full_raw_schema_prompt_block
from src.talk_to_data.nl_to_sql import validate_sql
from src.utils.helpers import estimate_tokens

st.set_page_config(page_title="AI Credit Risk Intelligence Platform", layout="wide", page_icon="\U0001F3E6")

FEATURE_COLS = config.FEATURE_COLS
CATEGORICAL_COLS = config.CATEGORICAL_COLS


# ----------------------------- Cached loaders -----------------------------

@st.cache_data(show_spinner="Loading dataset...")
def cached_load_raw_data():
    return load_raw_data()


@st.cache_resource(show_spinner="Training model (first run only)...")
def get_trained_model(raw_df):
    """Loads a previously trained model + artifacts bundle if one exists on
    disk (e.g. produced by `python -m src.ml.train`, the Docker entrypoint,
    or a prior container run) -- skipping retraining entirely. Otherwise
    trains one now so the app always works with a single `docker compose up`."""
    feat_df = engineer_features(raw_df)
    if os.path.exists(get_model_path()):
        try:
            model, metrics, cost_curve, splits = load_model_artifacts()
            return model, metrics, cost_curve, splits, feat_df
        except Exception:
            pass  # fall through to retrain if artifacts are partial/corrupt
    model, metrics, cost_curve, splits = train_model(feat_df)
    save_model_artifacts(model, metrics, cost_curve, splits)
    return model, metrics, cost_curve, splits, feat_df


@st.cache_data(show_spinner="Scoring applicants...")
def get_scored_dataframe(_feat_df, _model, tau_low, tau_optimal):
    feat_df = _feat_df.copy()
    probs = _model.predict_proba(feat_df[FEATURE_COLS])[:, 1]
    feat_df["PD_SCORE"] = probs
    feat_df["RISK_BAND"] = [risk_band(p, tau_low, tau_optimal) for p in probs]
    feat_df["AI_CREDIT_SCORE"] = pd_to_score(probs)
    feat_df["EXPECTED_LOSS"] = expected_loss(probs, feat_df["AMT_CREDIT"])
    return feat_df


@st.cache_resource(show_spinner="Building explainability engine...")
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_resource(show_spinner=False)
def get_duckdb_con(_scored_df):
    return build_duckdb_connection(_scored_df)


@st.cache_data(show_spinner="Extracting business rules...")
def get_surrogate_rules(_model, _X_val):
    return extract_surrogate_rules(_model, _X_val)


# ----------------------------- Load everything -----------------------------

raw_df, data_source_msg = cached_load_raw_data()
model, metrics, cost_curve, splits, feat_df_all = get_trained_model(raw_df)
X_train, X_val, y_train, y_val = splits
scored_df = get_scored_dataframe(feat_df_all, model, metrics["tau_low"], metrics["tau_optimal"])
explainer = get_shap_explainer(model)
con = get_duckdb_con(scored_df)
surrogate_rules = get_surrogate_rules(model, X_val)

st.title("\U0001F3E6 AI-Powered Credit Risk Intelligence Platform")
st.caption("Predictive, Explainable, Auditable, and Business-Readable Credit Risk Intelligence")
st.info(data_source_msg, icon="\u2139\ufe0f")

tabs = st.tabs([
    "\U0001F4CA Executive EDA", "\U0001F4B0 Loan Scorer & Decision Trace",
    "\U0001F50E XAI & Recourse", "\U0001F4CB Business Decision Rules", "\U0001F4AC Talk-to-Data",
])

# ============================== TAB 1: EDA ==============================
with tabs[0]:
    st.subheader("Dataset Summary & Data Quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{len(raw_df):,}")
    c2.metric("Default Rate", f"{raw_df['TARGET'].mean()*100:.2f}%")
    c3.metric("Features Used", len(FEATURE_COLS))
    missing_pct = raw_df.isna().mean().mean() * 100
    c4.metric("Avg. Missingness", f"{missing_pct:.1f}%")

    with st.expander("Feature categorization & data quality notes", expanded=False):
        st.markdown("""
| Category | Examples |
|---|---|
| Demographics | `CODE_GENDER`, `CNT_CHILDREN`, `NAME_FAMILY_STATUS`, `AGE_YEARS` |
| Financials | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` |
| Credit history / bureau | `EXT_SOURCE_1/2/3`, `AMT_REQ_CREDIT_BUREAU_QRT` |
| Employment & stability | `NAME_INCOME_TYPE`, `EMPLOYED_YEARS`, `OCCUPATION_TYPE` |
| Housing | `NAME_HOUSING_TYPE`, `FLAG_OWN_REALTY`, `FLAG_OWN_CAR` |

**Data quality:** `DAYS_EMPLOYED` contains the documented Home Credit anomaly
code `365243` for pensioners/unemployed applicants — decoded into an
`EMPLOYED_ANOMALY_FLAG` rather than treated as a real tenure value.
`EXT_SOURCE_1/2/3` carry meaningful missingness (bureau score not always
available), captured via `EXT_SOURCE_MISSING_COUNT` rather than naively imputed.
        """)

    st.subheader("Five Key Financial Insights")
    ins_tabs = st.tabs([
        "1. External Score Non-Linearity", "2. Debt-Trap Ratio", "3. Employment & Age Stability",
        "4. Over-Financing Risk", "5. Bureau Inquiry Bursts",
    ])

    with ins_tabs[0]:
        bins = pd.qcut(feat_df_all["EXT_SOURCE_MEAN"].fillna(feat_df_all["EXT_SOURCE_MEAN"].median()), 8, duplicates="drop")
        g = feat_df_all.groupby(bins, observed=True)["TARGET"].mean().reset_index()
        g["EXT_SOURCE_MEAN"] = g["EXT_SOURCE_MEAN"].astype(str)
        fig = px.bar(g, x="EXT_SOURCE_MEAN", y="TARGET", labels={"TARGET": "Default Rate", "EXT_SOURCE_MEAN": "External Score Bucket (low -> high)"})
        st.plotly_chart(fig, width='stretch')
        st.markdown("Default rate drops sharply — not linearly — as the mean external bureau score rises, steepest in the bottom two buckets. This is why the surrogate tree (Tab 4) splits on `EXT_SOURCE_MEAN` first.")

    with ins_tabs[1]:
        dti_bins = pd.cut(feat_df_all["ANNUITY_TO_INCOME"].clip(upper=1.0), bins=[0, 0.15, 0.25, 0.35, 0.5, 1.0])
        g = feat_df_all.groupby(dti_bins, observed=True)["TARGET"].mean().reset_index()
        g["ANNUITY_TO_INCOME"] = g["ANNUITY_TO_INCOME"].astype(str)
        fig = px.bar(g, x="ANNUITY_TO_INCOME", y="TARGET", labels={"TARGET": "Default Rate", "ANNUITY_TO_INCOME": "Annuity-to-Income Bucket"})
        st.plotly_chart(fig, width='stretch')
        st.markdown("Default risk climbs once the annuity-to-income ratio crosses ~35% — the threshold used in the hard knockout rule (Tab 4).")

    with ins_tabs[2]:
        age_bins = pd.cut(feat_df_all["AGE_YEARS"], bins=[20, 30, 40, 50, 60, 70])
        g = feat_df_all.groupby(age_bins, observed=True)["TARGET"].mean().reset_index()
        g["AGE_YEARS"] = g["AGE_YEARS"].astype(str)
        fig = px.bar(g, x="AGE_YEARS", y="TARGET", labels={"TARGET": "Default Rate", "AGE_YEARS": "Age Bracket"})
        st.plotly_chart(fig, width='stretch')
        st.markdown("Younger applicants (20-30) show a materially higher default rate than older, more employment-stable brackets.")

    with ins_tabs[3]:
        g = feat_df_all.assign(OVER_FINANCED=(feat_df_all["CREDIT_TO_GOODS_RATIO"] > 1.0)).groupby("OVER_FINANCED", observed=True)["TARGET"].mean().reset_index()
        fig = px.bar(g, x="OVER_FINANCED", y="TARGET", labels={"TARGET": "Default Rate", "OVER_FINANCED": "Credit Amount > Goods Price"})
        st.plotly_chart(fig, width='stretch')
        st.markdown("Applicants whose requested credit exceeds the goods price (under-collateralized lending) default at a visibly higher rate.")

    with ins_tabs[4]:
        g = feat_df_all.groupby("AMT_REQ_CREDIT_BUREAU_QRT", observed=True)["TARGET"].mean().reset_index()
        fig = px.bar(g, x="AMT_REQ_CREDIT_BUREAU_QRT", y="TARGET", labels={"TARGET": "Default Rate", "AMT_REQ_CREDIT_BUREAU_QRT": "Bureau Inquiries (last quarter)"})
        st.plotly_chart(fig, width='stretch')
        st.markdown("A burst of recent bureau inquiries — a signal of credit-seeking distress — tracks with higher default rates.")

# ============================== TAB 2: Scorer ==============================
with tabs[1]:
    st.subheader("Model Performance (validation set)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    c2.metric("Avg. Precision (PR-AUC)", f"{metrics['avg_precision']:.3f}")
    c3.metric("Cost-Optimal Threshold", f"{metrics['tau_optimal']:.3f}")
    c4.metric("Val. Default Rate", f"{metrics['val_default_rate']*100:.2f}%")

    with st.expander("Confusion matrix & cost-sensitive threshold derivation"):
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual: No Default", "Actual: Default"],
                              columns=["Pred: No Default", "Pred: Default"])
        st.dataframe(cm_df, width='stretch')
        st.markdown(
            f"Threshold chosen to minimize **expected cost = {config.FN_COST_MULTIPLIER:.0f} x False Negatives + False Positives** "
            f"(a missed defaulter assumed ~{config.FN_COST_MULTIPLIER:.0f}x costlier than an unnecessarily rejected good applicant), "
            f"not an arbitrary 0.5 cutoff. `scale_pos_weight={metrics['scale_pos_weight']:.1f}` handles the "
            f"underlying class imbalance during training."
        )
        fig = px.line(cost_curve, x="threshold", y="expected_cost", title="Expected Cost vs. Decision Threshold")
        fig.add_vline(x=metrics["tau_optimal"], line_dash="dash", annotation_text="optimal")
        st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Real-Time Loan Scorer")

    sample_profiles = {
        "-- Pick a sample profile --": None,
        "Prime applicant": X_val[(scored_df.loc[X_val.index, "RISK_BAND"] == "Low")].index,
        "Near-Prime applicant": X_val[(scored_df.loc[X_val.index, "RISK_BAND"] == "Medium")].index,
        "Sub-Prime applicant": X_val[(scored_df.loc[X_val.index, "RISK_BAND"] == "High")].index,
    }
    choice = st.selectbox("Load a sample applicant, or fill the form manually", list(sample_profiles.keys()))
    default_row = None
    if choice != "-- Pick a sample profile --" and len(sample_profiles[choice]) > 0:
        default_row = raw_df.loc[sample_profiles[choice][0]]

    def dv(col, fallback):
        if default_row is not None and col in default_row and pd.notna(default_row[col]):
            return default_row[col]
        return fallback

    with st.form("scorer_form"):
        colA, colB, colC = st.columns(3)
        with colA:
            income = st.number_input("Annual Income", value=float(dv("AMT_INCOME_TOTAL", 180000)), step=5000.0)
            credit = st.number_input("Requested Credit Amount", value=float(dv("AMT_CREDIT", 600000)), step=10000.0)
            annuity = st.number_input("Annuity (periodic payment)", value=float(dv("AMT_ANNUITY", 28000)), step=1000.0)
            goods_price = st.number_input("Goods Price", value=float(dv("AMT_GOODS_PRICE", 580000)), step=10000.0)
        with colB:
            age = st.slider("Age (years)", 21, 70, int(dv("DAYS_BIRTH", -13000) / -365.25) if default_row is not None else 38)
            employed_years = st.slider("Years Employed", 0, 45, 5)
            ext_mean = st.slider("External Bureau Score (mean, 0-1)", 0.0, 1.0, float(dv("EXT_SOURCE_2", 0.5)), 0.01)
            fam_members = st.slider("Family Members", 1, 9, int(dv("CNT_FAM_MEMBERS", 2)))
        with colC:
            income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student"])
            education = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
            contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
            bureau_inq = st.slider("Bureau Inquiries (last quarter)", 0, 10, 1)
        submitted = st.form_submit_button("Score Applicant", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "NAME_CONTRACT_TYPE": contract_type, "CODE_GENDER": "F", "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": max(fam_members - 2, 0), "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods_price, "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education, "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment",
            "OCCUPATION_TYPE": "Core staff", "AGE_YEARS": age, "EMPLOYED_YEARS": employed_years,
            "EMPLOYED_ANOMALY_FLAG": 1 if income_type == "Pensioner" else 0, "OWN_CAR_AGE": np.nan,
            "CNT_FAM_MEMBERS": fam_members, "EXT_SOURCE_1": ext_mean, "EXT_SOURCE_2": ext_mean, "EXT_SOURCE_3": ext_mean,
            "EXT_SOURCE_MEAN": ext_mean, "EXT_SOURCE_MISSING_COUNT": 0, "AMT_REQ_CREDIT_BUREAU_QRT": bureau_inq,
            "ANNUITY_TO_INCOME": annuity / max(income, 1), "CREDIT_TO_INCOME": credit / max(income, 1),
            "CREDIT_TO_GOODS_RATIO": credit / max(goods_price, 1), "INCOME_PER_FAM_MEMBER": income / max(fam_members, 1),
        }])
        for c in CATEGORICAL_COLS:
            row[c] = row[c].astype("category")
        row = row[FEATURE_COLS]

        pd_prob = float(model.predict_proba(row)[:, 1][0])
        score = float(pd_to_score(pd_prob))
        band = risk_band(pd_prob, metrics["tau_low"], metrics["tau_optimal"])
        el = float(expected_loss(pd_prob, credit))
        knockouts = hard_knockout_checks(row.iloc[0])

        st.session_state["last_scored_row"] = row
        st.session_state["last_score_result"] = dict(pd_prob=pd_prob, score=score, band=band, el=el, knockouts=knockouts)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Probability of Default", f"{pd_prob*100:.1f}%")
        r2.metric("AI Credit Risk Score", f"{score:.0f}", help="300-850 presentation transform of PD. NOT a FICO score.")
        band_color = {"Low": "\U0001F7E2", "Medium": "\U0001F7E1", "High": "\U0001F534"}[band]
        r3.metric("Risk Band", f"{band_color} {band}")
        r4.metric("Expected Loss", f"${el:,.0f}", help="PD x LGD(45%) x Exposure at Default")

        if knockouts:
            st.error("**Tier 1 Hard Knockout triggered:**\n" + "\n".join(f"- {k}" for k in knockouts))
        else:
            st.success("No Tier 1 hard-knockout rules triggered.")

# ============================== TAB 3: XAI ==============================
with tabs[2]:
    st.subheader("Global Feature Importance")
    sv_sample = X_val.sample(min(500, len(X_val)), random_state=1)
    shap_values_global = explainer.shap_values(sv_sample)
    if isinstance(shap_values_global, list):
        shap_values_global = shap_values_global[1]
    mean_abs = np.abs(shap_values_global).mean(axis=0)
    imp_df = pd.DataFrame({"feature": sv_sample.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=True).tail(15)
    fig = px.bar(imp_df, x="mean_abs_shap", y="feature", orientation="h", title="Top 15 Global Risk Drivers (mean |SHAP|)")
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Local Explanation & Counterfactual Recourse")

    if "last_scored_row" not in st.session_state:
        st.info("Score an applicant in the **Loan Scorer** tab first to see their individual explanation here.")
    else:
        row = st.session_state["last_scored_row"]
        result = st.session_state["last_score_result"]
        sv_row = explainer.shap_values(row)
        if isinstance(sv_row, list):
            sv_row = sv_row[1]
        sv_row = sv_row[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else float(np.atleast_1d(base_value)[0])

        contrib_df = pd.DataFrame({"feature": row.columns, "value": row.iloc[0].astype(str), "shap": sv_row})
        contrib_df["abs_shap"] = contrib_df["shap"].abs()
        contrib_df = contrib_df.sort_values("abs_shap", ascending=False).head(10).sort_values("shap")

        fig = go.Figure(go.Bar(
            x=contrib_df["shap"], y=[f"{f} = {v}" for f, v in zip(contrib_df["feature"], contrib_df["value"])],
            orientation="h", marker_color=["#d62728" if s > 0 else "#2ca02c" for s in contrib_df["shap"]],
        ))
        fig.update_layout(title=f"Top drivers for this applicant (base log-odds: {base_value:.2f})",
                           xaxis_title="SHAP contribution (pushes risk up / red = higher risk, green = lower risk)")
        st.plotly_chart(fig, width='stretch')

        if result["band"] == "High":
            st.markdown("**What-if recourse:** adjust the two most controllable levers below to see what it would take to move out of High Risk.")
            c1, c2 = st.columns(2)
            new_credit = c1.slider("Adjusted Credit Amount", int(row["AMT_CREDIT"].iloc[0] * 0.4), int(row["AMT_CREDIT"].iloc[0]), int(row["AMT_CREDIT"].iloc[0]))
            new_annuity = c2.slider("Adjusted Annuity", int(row["AMT_ANNUITY"].iloc[0] * 0.4), int(row["AMT_ANNUITY"].iloc[0]), int(row["AMT_ANNUITY"].iloc[0]))
            what_if_row = row.copy()
            what_if_row["AMT_CREDIT"] = new_credit
            what_if_row["AMT_ANNUITY"] = new_annuity
            what_if_row["CREDIT_TO_INCOME"] = new_credit / max(row["AMT_INCOME_TOTAL"].iloc[0], 1)
            what_if_row["ANNUITY_TO_INCOME"] = new_annuity / max(row["AMT_INCOME_TOTAL"].iloc[0], 1)
            what_if_row["CREDIT_TO_GOODS_RATIO"] = new_credit / max(row["AMT_GOODS_PRICE"].iloc[0], 1)
            new_pd = float(model.predict_proba(what_if_row)[:, 1][0])
            new_band = risk_band(new_pd, metrics["tau_low"], metrics["tau_optimal"])
            st.metric("Recomputed Probability of Default", f"{new_pd*100:.1f}%", delta=f"{(new_pd - result['pd_prob'])*100:.1f} pts")
            st.write(f"Resulting risk band: **{new_band}**")
        else:
            st.caption("Recourse simulator applies to High Risk applicants.")

# ============================== TAB 4: Business Decision Rules ==============================
with tabs[3]:
    st.header("Business Decision Rules")
    st.caption(
        "Human-readable credit policy logic, bridging the trained ML model and raw dataset "
        "statistics for review by non-technical credit risk officers."
    )

    st.subheader("Tier 1 — Hard Knockout Rules (Policy Gate)")
    st.caption("Fixed business thresholds. Non-negotiable — applied before the model is even consulted.")
    for rule_text in get_hard_knockout_rule_definitions():
        st.markdown(f"- {rule_text}")
    n_knockout = feat_df_all.apply(lambda r: len(hard_knockout_checks(r)) > 0, axis=1).sum()
    st.caption(f"{n_knockout:,} of {len(feat_df_all):,} applicants in the current dataset would trigger at least one Tier 1 knockout.")

    st.divider()
    st.subheader("Tier 2 — Model-Derived Business Rules (surrogate decision tree)")
    st.caption("A shallow decision tree fit on the trained model's own High-Risk calls, read out as IF-THEN rules and audited against real outcomes.")

    rule_view = st.radio("View as:", ["Plain-English bullets", "Structured table", "Raw decision tree (sklearn.tree.export_text)"], horizontal=True)

    if rule_view == "Plain-English bullets":
        bullets = get_readable_rules_bullets(model, X_val)
        if bullets:
            for b in bullets:
                st.markdown(f"- {b}")
        else:
            st.warning("No surrogate rules met the minimum support/precision threshold for this run.")
    elif rule_view == "Structured table":
        if surrogate_rules:
            rules_df = pd.DataFrame(surrogate_rules).rename(columns={
                "condition": "IF (condition)", "support": "Support (n)",
                "coverage_pct": "Coverage (%)", "precision_pct": "Precision (%)",
            })
            st.dataframe(rules_df, width='stretch', hide_index=True)
        else:
            st.warning("No surrogate rules met the minimum support/precision threshold for this run.")
    else:
        st.code(get_readable_tree_text(model, X_val), language="text")
        st.caption("Exact IF-THEN structure of the surrogate tree, via sklearn.tree.export_text — for technical/audit review alongside the plain-English view above.")

# ============================== TAB 5: Talk-to-Data ==============================
with tabs[4]:
    st.subheader("Talk-to-Data Assistant")

    api_key = os.environ.get("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY)
    key_input = st.text_input("Any API key (optional — leave blank to use offline verified query library)",
                               value=api_key, type="password")
    use_offline_toggle = st.toggle("Force offline query-library mode", value=(not bool(key_input)))
    mode_label = "\U0001F4DA Offline Query Library" if (use_offline_toggle or not key_input) else "\U0001F7E2 Live LLM (Claude)"
    st.caption(f"Current mode: **{mode_label}**")

    with st.expander("Prompt engineering: semantic schema dictionary & token reduction"):
        pruned = semantic_schema_prompt_block()
        full = full_raw_schema_prompt_block(scored_df)
        pt, ft = estimate_tokens(pruned), estimate_tokens(full)
        c1, c2, c3 = st.columns(3)
        c1.metric("Naive full-schema dump", f"~{ft} tokens")
        c2.metric("Curated semantic dictionary", f"~{pt} tokens")
        c3.metric("Reduction", f"{(1 - pt/ft)*100:.0f}%")
        st.caption("Measured on this demo's ~29-column feature set. The real Kaggle dataset has 122 raw columns, so curation saves proportionally more there.")
        st.code(pruned, language="text")

    st.markdown("Try a question, or pick one of the 5 verified queries below:")
    preset = st.selectbox("Verified query library", ["-- free text below instead --"] + [q["question"] for q in OFFLINE_QUERY_LIBRARY])
    question = st.text_input("Or ask your own question in plain English",
                              value="" if preset == "-- free text below instead --" else preset)

    if st.button("Ask", type="primary") and question:
        result = answer_question(question, con, api_key=key_input, use_offline=use_offline_toggle)
        st.write(f"**Mode:** `{result['mode']}` | **Attempts:** {result['attempts']}")
        if result["sql"]:
            st.code(result["sql"], language="sql")
        if result["dataframe"] is not None:
            st.dataframe(result["dataframe"], width='stretch')
            st.success(result["narrative"])
        if result["error"] and result["dataframe"] is None:
            st.error(result["error"])

st.divider()
st.caption(
    "AI Credit Risk Score is a presentation-layer transform of the model's predicted default "
    "probability and is not a FICO score or regulated credit score."
)
