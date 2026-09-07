"""
Versioned prompt templates for the Talk-to-Data system.

PROMPT_VERSION bump this whenever SEMANTIC_SCHEMA or SYSTEM_PROMPT_TEMPLATE
changes, so prompt regressions are traceable in logs/evaluations.
"""
from src.utils.config import config

PROMPT_VERSION = "v1.0"

# The curated, business-readable Semantic Schema Dictionary -- deliberately
# NOT the full raw column dump. This is what keeps the LLM prompt small and
# on-topic (see query_runner.full_raw_schema_prompt_block for the naive
# baseline this is measured against).
SEMANTIC_SCHEMA = {
    "TARGET": "1 if the applicant defaulted on the loan, 0 if repaid on time (integer)",
    "NAME_CONTRACT_TYPE": "Loan type: 'Cash loans' or 'Revolving loans'",
    "CODE_GENDER": "Applicant gender: 'M' or 'F'",
    "CNT_CHILDREN": "Number of children the applicant has (integer)",
    "AMT_INCOME_TOTAL": "Applicant's total annual income (currency)",
    "AMT_CREDIT": "Total credit amount of the loan (currency)",
    "AMT_ANNUITY": "Loan annuity / periodic payment amount (currency)",
    "AMT_GOODS_PRICE": "Price of the goods the loan is financing (currency)",
    "NAME_INCOME_TYPE": "Income source category, e.g. 'Working', 'Pensioner', 'Unemployed'",
    "NAME_EDUCATION_TYPE": "Highest education level attained",
    "NAME_FAMILY_STATUS": "Marital / family status",
    "NAME_HOUSING_TYPE": "Housing situation, e.g. 'House / apartment', 'Rented apartment'",
    "OCCUPATION_TYPE": "Applicant's occupation category",
    "AGE_YEARS": "Applicant age in years (derived, float)",
    "EMPLOYED_YEARS": "Years at current employment (derived, float; 0 for pensioners)",
    "CNT_FAM_MEMBERS": "Number of family members in the household (integer)",
    "EXT_SOURCE_1": "Normalized external credit bureau score #1, range 0-1 (higher = safer)",
    "EXT_SOURCE_2": "Normalized external credit bureau score #2, range 0-1 (higher = safer)",
    "EXT_SOURCE_3": "Normalized external credit bureau score #3, range 0-1 (higher = safer)",
    "EXT_SOURCE_MEAN": "Mean of the three external bureau scores (derived, 0-1)",
    "ANNUITY_TO_INCOME": "AMT_ANNUITY / AMT_INCOME_TOTAL, debt-service burden ratio (derived)",
    "CREDIT_TO_INCOME": "AMT_CREDIT / AMT_INCOME_TOTAL (derived)",
    "CREDIT_TO_GOODS_RATIO": "AMT_CREDIT / AMT_GOODS_PRICE, over-financing indicator (derived)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Number of credit bureau inquiries in the preceding quarter (integer)",
    "PD_SCORE": "Model-predicted probability of default, 0-1 (derived, added after scoring)",
    "RISK_BAND": "'Low' / 'Medium' / 'High' risk classification (derived, added after scoring)",
    "AI_CREDIT_SCORE": "300-850 presentation score derived from PD_SCORE (derived, added after scoring)",
}


def semantic_schema_prompt_block() -> str:
    lines = [f"- {col}: {desc}" for col, desc in SEMANTIC_SCHEMA.items()]
    return f"Table `{config.DUCKDB_TABLE_NAME}` columns:\n" + "\n".join(lines)


def full_raw_schema_prompt_block(df, n_samples: int = 3) -> str:
    """Naive baseline an un-pruned prompt would actually send: every dataframe
    column with its dtype and a few sample values, no curation. Used only for
    the token-reduction comparison shown in the UI -- never sent to the LLM.

    Note: the real Kaggle application_train.csv has 122 raw columns; this
    dataset carries a pre-trimmed ~24-29, so the reduction shown here is a
    conservative lower bound on what curation saves against the full
    competition dataset."""
    lines = [f"Table `{config.DUCKDB_TABLE_NAME}` -- full raw schema dump (no curation):"]
    for col in df.columns:
        samples = df[col].dropna().unique()[:n_samples]
        sample_str = ", ".join(str(s) for s in samples)
        lines.append(f"- {col} ({df[col].dtype}): e.g. [{sample_str}]")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are a SQL analyst for a bank's credit risk platform. Convert the user's "
        "question into a single read-only DuckDB SELECT statement.\n\n"
        f"{semantic_schema_prompt_block()}\n\n"
        "Rules:\n"
        "- Output ONLY the SQL inside a ```sql code block, nothing else.\n"
        "- Exactly one SELECT statement. No DDL/DML, no semicolon-separated statements.\n"
        f"- Always query the table `{config.DUCKDB_TABLE_NAME}`.\n"
        "- Prefer ROUND(..., 2) for ratios/percentages and give columns readable aliases."
    )


def build_reflection_prompt(question: str, error_context: str) -> str:
    return (
        f"Your previous SQL failed with this error:\n{error_context}\n\n"
        f"Fix it and re-answer the original question: {question}"
    )
