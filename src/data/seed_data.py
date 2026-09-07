"""
Seed script: generates the synthetic Home Credit dataset and writes it to
disk at the configured dataset path, so the container always has a concrete
data/application_train.csv file to point to (rather than only ever
generating data in-memory on the fly). Run automatically by the Docker
entrypoint (entrypoint.sh) on container startup if no dataset is present;
can also be run manually:

    python -m src.data.seed_data
"""
import os

from src.utils.config import config
from src.utils.docker_utils import get_dataset_path
from src.utils.logger import get_logger
from src.data.loader import generate_synthetic_home_credit, SYNTHETIC_MARKER_SUFFIX

log = get_logger(__name__)


def seed():
    path = get_dataset_path()
    marker_path = path + SYNTHETIC_MARKER_SUFFIX

    if os.path.exists(path):
        log.info("Dataset already present at %s -- skipping seed.", path)
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log.info("Seeding synthetic dataset (%d rows) to %s ...", config.SYNTHETIC_ROWS, path)
    df = generate_synthetic_home_credit(n=config.SYNTHETIC_ROWS, seed=config.RANDOM_SEED)
    df.to_csv(path, index=False)
    # Marks the seeded file as synthetic so src.data.loader never reports it
    # as the real dataset. If you later replace `path` with the real Kaggle
    # file, also delete this marker file (or it will keep flagging the
    # dataset as synthetic even after being replaced).
    with open(marker_path, "w") as f:
        f.write("synthetic\n")
    log.info("Seed complete: %s (%d rows, %d columns).", path, len(df), len(df.columns))
    return path


if __name__ == "__main__":
    seed()
