"""
Resolves data/model paths so the same code works whether it's run locally
(`streamlit run app.py` from the project root) or inside the Docker
container, where `data/` and `models/` are mounted volumes.
"""
import os
from src.utils.config import config


def is_running_in_docker() -> bool:
    """Best-effort detection: /.dockerenv is created by the Docker runtime."""
    return os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "1"


def get_data_dir() -> str:
    """Returns the directory to look for dataset files in. Respects the
    DATA_DIR env var (set in docker-compose.yml) so the same code works
    unmodified inside or outside the container."""
    return config.DATA_DIR


def get_dataset_path(filename: str = None) -> str:
    filename = filename or config.DATASET_FILENAME
    return os.path.join(get_data_dir(), filename)


def get_models_dir() -> str:
    path = config.MODELS_DIR
    os.makedirs(path, exist_ok=True)
    return path


def get_model_path() -> str:
    return os.path.join(get_models_dir(), config.MODEL_FILENAME)


def get_metrics_path() -> str:
    return os.path.join(get_models_dir(), config.METRICS_FILENAME)


def get_artifacts_bundle_path() -> str:
    return os.path.join(get_models_dir(), config.ARTIFACTS_BUNDLE_FILENAME)
