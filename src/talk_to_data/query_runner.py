"""
Executes validated SQL against DuckDB and orchestrates the full Talk-to-Data
trace: LLM SQL generation -> AST validation -> execution -> (on failure)
reflection retry -> (on exhausted attempts) safe fallback to the verified
offline query library. Also powers pure offline mode when no API key is set.
"""
import duckdb
import pandas as pd

from src.utils.config import config
from src.utils.logger import get_logger
from src.talk_to_data.nl_to_sql import validate_sql, call_claude_for_sql

log = get_logger(__name__)

TABLE_NAME = config.DUCKDB_TABLE_NAME

# Verified, hand-checked queries -- used in offline mode and as the safe
# fallback after the LLM exhausts its self-healing attempts, so the app
# never breaks during evaluation even without an API key.
OFFLINE_QUERY_LIBRARY = [
    {
        "question": "What is the overall default rate in the dataset?",
        "sql": f"SELECT ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct, COUNT(*) AS n_applicants FROM {TABLE_NAME}",
    },
    {
        "question": "What is the average income by education level?",
        "sql": f"SELECT NAME_EDUCATION_TYPE, ROUND(AVG(AMT_INCOME_TOTAL), 0) AS avg_income, COUNT(*) AS n "
               f"FROM {TABLE_NAME} GROUP BY NAME_EDUCATION_TYPE ORDER BY avg_income DESC",
    },
    {
        "question": "How many applicants fall in each risk band?",
        "sql": f"SELECT RISK_BAND, COUNT(*) AS n_applicants, ROUND(AVG(TARGET)*100,2) AS actual_default_rate_pct "
               f"FROM {TABLE_NAME} GROUP BY RISK_BAND ORDER BY actual_default_rate_pct DESC",
    },
    {
        "question": "What is the default rate for applicants with a high annuity-to-income ratio (over 35%)?",
        "sql": f"SELECT ROUND(AVG(TARGET)*100, 2) AS default_rate_pct, COUNT(*) AS n "
               f"FROM {TABLE_NAME} WHERE ANNUITY_TO_INCOME > 0.35",
    },
    {
        "question": "Which income type has the highest default rate?",
        "sql": f"SELECT NAME_INCOME_TYPE, ROUND(AVG(TARGET)*100, 2) AS default_rate_pct, COUNT(*) AS n "
               f"FROM {TABLE_NAME} GROUP BY NAME_INCOME_TYPE HAVING COUNT(*) > 20 ORDER BY default_rate_pct DESC",
    },
    {
        "question": "What is the average external bureau score for defaulters vs non-defaulters?",
        "sql": f"SELECT TARGET, ROUND(AVG(EXT_SOURCE_MEAN), 3) AS avg_ext_source_mean, COUNT(*) AS n "
               f"FROM {TABLE_NAME} GROUP BY TARGET",
    },
]


def build_duckdb_connection(df: pd.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register(TABLE_NAME, df)
    return con


def _match_offline_query(question: str):
    """Lightweight keyword match against the verified library -- used for
    offline mode and as the safe fallback after 2 failed LLM attempts."""
    q = question.lower()
    best, best_score = OFFLINE_QUERY_LIBRARY[0], -1
    for item in OFFLINE_QUERY_LIBRARY:
        overlap = len(set(q.split()) & set(item["question"].lower().split()))
        if overlap > best_score:
            best, best_score = item, overlap
    return best


def _summarize_result(df: pd.DataFrame, question: str) -> str:
    if df is None or df.empty:
        return "The query returned no rows."
    n_rows, n_cols = df.shape
    if n_rows == 1 and n_cols <= 3:
        parts = [f"{col} = {df.iloc[0][col]}" for col in df.columns]
        return "Result: " + ", ".join(parts)
    return f"Returned {n_rows} rows x {n_cols} columns. Top row: " + \
        ", ".join(f"{c}={df.iloc[0][c]}" for c in df.columns)


def answer_question(question: str, con, api_key: str = None, use_offline: bool = False):
    """Runs the full talk-to-data trace and returns a dict describing what
    happened: mode, sql used, attempt count, result dataframe, business
    narrative, and any error."""
    result = {
        "question": question, "mode": None, "sql": None, "attempts": 0,
        "dataframe": None, "error": None, "narrative": None,
    }

    if use_offline or not api_key:
        result["mode"] = "offline_query_library"
        best = _match_offline_query(question)
        result["sql"] = best["sql"]
        result["attempts"] = 1
        try:
            result["dataframe"] = con.execute(best["sql"]).fetchdf()
            result["narrative"] = _summarize_result(result["dataframe"], question)
        except Exception as e:
            result["error"] = str(e)
            log.error("Offline query execution failed: %s", e)
        return result

    result["mode"] = "llm_online"
    error_context = None
    for attempt in range(1, config.SQL_MAX_ATTEMPTS + 1):
        result["attempts"] = attempt
        try:
            sql = call_claude_for_sql(question, api_key, error_context)
        except Exception as e:
            result["error"] = f"LLM call failed: {e}"
            log.error(result["error"])
            break
        result["sql"] = sql
        ok, err = validate_sql(sql)
        if not ok:
            error_context = f"SQL rejected by validator: {err}\nSQL was: {sql}"
            result["error"] = error_context
            log.warning("Attempt %d rejected by validator: %s", attempt, err)
            continue
        try:
            result["dataframe"] = con.execute(sql).fetchdf()
            result["error"] = None
            result["narrative"] = _summarize_result(result["dataframe"], question)
            log.info("Attempt %d succeeded", attempt)
            return result
        except Exception as e:
            error_context = f"DuckDB execution error: {e}\nSQL was: {sql}"
            result["error"] = error_context
            log.warning("Attempt %d execution error: %s", attempt, e)
            continue

    # Exhausted attempts -> safe fallback, so the app never crashes on stage.
    result["mode"] = "llm_failed_fallback_to_library"
    best = _match_offline_query(question)
    result["sql"] = best["sql"]
    try:
        result["dataframe"] = con.execute(best["sql"]).fetchdf()
        result["narrative"] = _summarize_result(result["dataframe"], question) + \
            " (Note: the live LLM query failed validation/execution twice, so this is the closest verified fallback query.)"
    except Exception as e:
        result["error"] = str(e)
        log.error("Fallback query execution failed: %s", e)
    return result
