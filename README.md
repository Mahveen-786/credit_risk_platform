# AI-Powered Credit Risk Intelligence Platform

Predictive, explainable,
auditable, and business-readable credit risk intelligence on the Kaggle
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
dataset.

## Quick start

```bash
# 1. (Optional) put the real dataset in place
#    Download application_train.csv from the Kaggle link above and drop it in data/
#    If it's not there, the container auto-seeds a schema-matched synthetic
#    dataset (15,000 rows, ~8% default rate) to data/application_train.csv
#    on first startup, and auto-trains a model into models/ -- so the app
#    always works with zero manual setup.

# 2. (Optional) configure environment variables
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY if you want live Talk-to-Data mode

# 3. Run with Docker (recommended -- single command)
docker compose up --build

# 4. Open http://localhost:8501
```

**Without Docker:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**To pre-train the model explicitly** (otherwise app.py trains automatically on first run):
```bash
python -m src.ml.train
```

**To seed a synthetic dataset file to disk explicitly** (otherwise done automatically by the Docker entrypoint, or generated in-memory by the app if no file is ever written):
```bash
python -m src.data.seed_data
```

**To run the EDA notebook as a script:**
```bash
python notebooks/eda.py
```

## Architecture

```
Applicant Decision Trace:
  Raw applicant data -> feature engineering (src/data/preprocessor.py)
  -> LightGBM inference (src/ml/predict.py) -> calibrated PD
  -> cost-sensitive threshold -> Risk Band + AI Credit Score + Expected Loss
  -> SHAP local explanation -> Tier 1/2 rule check -> decision

Talk-to-Data Trace:
  NL question -> semantic schema dictionary (pruned column list)
  -> Claude generates SQL -> sqlglot AST validator
  -> DuckDB execution -> (on failure) reflection retry, max 2 attempts
  -> (on exhausted attempts) safe fallback to verified offline query library
```

## Project structure

```
credit_risk_platform/
├── data/                     # dataset files (gitignored; add application_train.csv here)
├── notebooks/
│   ├── eda.ipynb             # exploratory analysis, interactive
│   └── eda.py                # same analysis, script form
├── src/
│   ├── data/
│   │   ├── loader.py         # loads real CSV or generates synthetic fallback; honest synthetic/real messaging
│   │   ├── preprocessor.py   # anomaly handling, ratio features, encoding
│   │   └── seed_data.py      # writes a synthetic dataset to data/ on container startup if none exists
│   ├── ml/
│   │   ├── train.py          # LightGBM training + artifact persistence/reload
│   │   ├── predict.py        # PD -> score/band/expected-loss, hard-knockout applicant check
│   │   ├── evaluate.py       # AUC/PR-AUC, confusion matrix, cost-sensitive threshold
│   │   └── rules.py          # Business Decision Rules: Tier 1 threshold policy + Tier 2 surrogate-tree extraction
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py      # Claude SQL generation + AST security validator
│   │   ├── query_runner.py   # DuckDB execution, self-healing loop, offline library
│   │   └── prompt_templates.py  # versioned prompts + semantic schema dictionary
│   └── utils/
│       ├── logger.py
│       ├── config.py         # all tunable constants in one place
│       ├── helpers.py
│       └── docker_utils.py   # data/model/artifact path resolution (local vs. container)
├── sql/
│   └── schema.sql            # DDL: full raw application_train (122 cols) + curated applicants working table
├── models/                   # saved model artifacts (.joblib), gitignored -- auto-populated on first run
├── app.py                    # Streamlit UI entrypoint
├── entrypoint.sh             # Docker container startup: seed data -> train model -> launch Streamlit
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

## Model selection & class imbalance strategy

**LightGBM** classifier, chosen for native categorical feature support and
fast training on tabular data. Class imbalance (~8% default rate) is handled
via `scale_pos_weight` computed from the training split rather than
oversampling (SMOTE, etc.) -- this avoids synthetic-minority artifacts and
keeps predicted probabilities better calibrated for the score/threshold math
downstream.

**Decision threshold** is not the default 0.5. It's chosen by grid search to
minimize expected cost = `FN_COST_MULTIPLIER x False Negatives + False Positives`
(default multiplier: 10, i.e. missing a defaulter is assumed ~10x costlier
than rejecting a good applicant) -- configurable via `FN_COST_MULTIPLIER` in
`.env`.

## Evaluation metrics

On the validation split (20%, stratified):
- **AUC-ROC** and **Average Precision (PR-AUC)** -- PR-AUC is reported
  alongside AUC-ROC because AUC-ROC alone is optimistic under ~8% class
  imbalance.
- **Confusion matrix** at the cost-optimal threshold (not at 0.5).
- Both are shown live in the Streamlit app's "Loan Scorer & Decision Trace"
  tab, computed on your actual dataset (real or synthetic) at runtime --
  see the app for current numbers rather than a static figure here, since
  they'll differ between the synthetic fallback and the real Kaggle data.

## Prompt engineering & token optimization

The Talk-to-Data system does **not** dump the full dataframe schema into the
LLM prompt. `src/talk_to_data/prompt_templates.py` defines a curated
**Semantic Schema Dictionary** -- only business-relevant columns, each with a
plain-English description -- which the app measures against a naive
full-schema-dump baseline (dtypes + sample values for every column) and
displays the real, measured token reduction in the Talk-to-Data tab (not a
guessed number). Note this demo dataset already carries a pre-trimmed ~29
columns; the real Kaggle `application_train.csv` has 122, so curation saves
proportionally more there.

**Hallucination control:** every LLM-generated query passes through a
sqlglot AST validator before execution -- single read-only `SELECT`
statements only, no DDL/DML, no multi-statement injection. If generation or
validation fails, the system retries once with the error fed back to the
model (self-healing), and after 2 failed attempts falls back to the closest
verified query in the offline library rather than surfacing a broken state.

## Rule derivation logic

All business decision rules live in `src/ml/rules.py` and are exposed in the
Streamlit app's **"Business Decision Rules"** tab (three interchangeable
views: plain-English bullets, a structured/sortable table, and the raw
decision tree).

- **Tier 1 (hard knockouts, threshold-based / "Option B"):** a fixed policy
  gate -- annuity-to-income > 50%, unemployed with no verified income,
  credit > 150% of goods price, or bureau score critically low. Thresholds
  are defined once in `config.py`; `src/ml/rules.py::get_hard_knockout_rule_definitions()`
  renders them as plain-English policy statements for the UI, and
  `src/ml/predict.py::hard_knockout_checks()` applies the same constants
  against a real applicant -- single source of truth, so the two can't drift.
- **Tier 2 (model-derived rules, decision-tree extraction / "Option A"):** a
  shallow `DecisionTreeClassifier` is fit on the trained model's own
  top-risk-quantile predictions (a surrogate), so its splits approximate
  *why* the model calls something High Risk. Read out three ways:
  - `extract_surrogate_rules()` -- structured dicts (condition, support,
    coverage, precision) for the sortable UI table, each audited against
    real outcomes.
  - `get_readable_rules_bullets()` -- the same rules as plain-English
    `IF ... THEN Flag as HIGH RISK` bullets for non-technical review.
  - `get_readable_tree_text()` -- the literal `sklearn.tree.export_text`
    dump of the tree, for exact technical/audit review.

## Known limitations & possible improvements

- Only the main `application_train.csv` table is used. The related tables
  (`bureau.csv`, `previous_application.csv`, `POS_CASH_balance.csv`,
  `instalments_payments.csv`, `credit_card_balance.csv`) contain genuine
  behavioral history (payment punctuality, revolving utilization, past loan
  performance) that would meaningfully improve model accuracy, but require
  aggregation/join feature engineering that's out of scope for this build.
- The counterfactual recourse simulator is a simple 2-lever what-if slider,
  not a formal minimal-distance optimizer.
- Token counts in the UI are estimated at ~4 characters/token (a standard
  approximation), not an exact tokenizer count.
- No automated test suite is included in this structure (kept out
  deliberately to match the exact folder layout requested); recommend adding
  one (e.g. `tests/` with pytest covering the preprocessor, threshold
  tuning, and the SQL validator) before treating this as production-ready.
- `AI_CREDIT_SCORE` is an explicit presentation-layer transform of PD, not a
  regulated credit score -- labeled as such throughout the UI and code.
- `sql/schema.sql`'s `application_train` DDL is reconstructed from the
  public Kaggle data dictionary for documentation/audit completeness;
  cross-check against the official file if exact column-for-column fidelity
  is required.
- The Docker entrypoint (`entrypoint.sh`) and Dockerfile were validated by
  running the same seed -> train -> serve sequence directly (outside a
  container) and confirming each step and the final Streamlit process come
  up cleanly; an actual `docker build`/`docker compose up` was not run in
  the environment this was built in, so verify that on your machine before
  treating it as final.
