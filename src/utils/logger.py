"""Standard logging setup for the platform. Import get_logger(__name__) anywhere."""
import logging
import os
import sys

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_CONFIGURED = False


def _configure_root():
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=_LOG_LEVEL,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
