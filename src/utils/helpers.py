"""Small, generic, dependency-light helper functions shared across modules."""


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token). Good enough
    for a directional token-reduction benchmark; not a tokenizer-accurate count."""
    return max(1, len(text) // 4)


def format_currency(value: float) -> str:
    try:
        return f"${value:,.0f}"
    except (TypeError, ValueError):
        return str(value)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator in (0, None):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default
