#!/bin/sh
# Container entrypoint: ensures a dataset and a trained model artifact exist
# BEFORE Streamlit starts serving, so `docker compose up --build` works as a
# single command with no manual setup step -- and so the first page load
# isn't the moment training happens (app.py still has that as a fallback,
# but this makes container startup logs show the work explicitly).
set -e

DATA_DIR="${DATA_DIR:-data}"
DATASET_FILENAME="${DATASET_FILENAME:-application_train.csv}"
MODELS_DIR="${MODELS_DIR:-models}"
MODEL_FILENAME="lgbm_credit_risk.joblib"

echo "=== Credit Risk Platform: container startup ==="

if [ ! -f "${DATA_DIR}/${DATASET_FILENAME}" ]; then
    echo "-> No dataset found at ${DATA_DIR}/${DATASET_FILENAME}. Seeding synthetic dataset..."
    python -m src.data.seed_data
else
    echo "-> Dataset already present at ${DATA_DIR}/${DATASET_FILENAME}."
fi

if [ ! -f "${MODELS_DIR}/${MODEL_FILENAME}" ]; then
    echo "-> No trained model found. Training now (one-time, ~1-2 min)..."
    python -m src.ml.train
else
    echo "-> Trained model already present at ${MODELS_DIR}/${MODEL_FILENAME}. Skipping training."
fi

echo "=== Startup checks complete. Launching Streamlit. ==="
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
