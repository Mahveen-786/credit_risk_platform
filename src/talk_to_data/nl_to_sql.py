"""
Natural-language-to-SQL generation (via the Anthropic API) and the
AST-based read-only SQL security guardrail (via sqlglot).
"""
import re
import sqlglot
from sqlglot import exp

from src.utils.config import config
from src.utils.logger import get_logger
from src.talk_to_data.prompt_templates import build_system_prompt, build_reflection_prompt

log = get_logger(__name__)


def validate_sql(sql: str):
    """AST-level guardrail: single, read-only SELECT statement only.
    Returns (is_valid, error_message)."""
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except Exception as e:
        return False, f"SQL failed to parse: {e}"

    if len(statements) != 1:
        return False, "Only a single SQL statement is allowed."

    stmt = statements[0]
    if stmt is None or not isinstance(stmt, exp.Select):
        return False, "Only read-only SELECT statements are permitted."

    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create, exp.TruncateTable)
    for node in stmt.walk():
        n = node[0] if isinstance(node, tuple) else node
        if isinstance(n, forbidden):
            return False, f"Blocked statement type: {type(n).__name__}"

    sql_lower = sql.lower()
    for kw in ["drop ", "delete ", "insert ", "update ", "alter ", "attach ", "pragma ", ";--"]:
        if kw in sql_lower:
            return False, f"Blocked keyword detected: '{kw.strip()}'"

    if sql.count(";") > 1 or (sql.count(";") == 1 and not sql.strip().endswith(";")):
        return False, "Multiple statements are not allowed."

    return True, None


def _extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def call_claude_for_sql(question: str, api_key: str, error_context: str = None) -> str:
    """Calls the Anthropic API to translate a natural-language question into
    a single read-only DuckDB SELECT statement, using the pruned semantic
    schema dictionary rather than a full raw column dump."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = build_system_prompt()
    user_msg = build_reflection_prompt(question, error_context) if error_context else question

    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    sql = _extract_sql(text)
    log.info("LLM generated SQL (error_context=%s): %s", bool(error_context), sql)
    return sql
