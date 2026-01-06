#!/bin/bash
# =============================================================================
# run_full_pipeline.sh - Run Complete E-Commerce Analytics Pipeline
#
# This script runs the entire pipeline on Dataproc:
#   1. Preprocessing (Leakage-Free Feature Engineering)
#   2. Intent Model Training (Random Forest)
#   3. Recommender Training (ALS)
#
# Usage:
#   ./scripts/run_full_pipeline.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

echo "=========================================="
echo "E-Commerce Analytics Pipeline"
echo "=========================================="
echo "Project Root: ${PROJECT_ROOT}"
echo ""

# Step 1: Preprocessing
echo "[1/3] Running Preprocessing..."
echo "------------------------------------------"
bash "${SCRIPT_DIR}/submit_preprocessing.sh"
echo ""

# Wait for job completion (in production, use proper job monitoring)
echo "Waiting for preprocessing to complete..."
sleep 10

# Step 2: Intent Model Training
echo "[2/3] Training Intent Prediction Model..."
echo "------------------------------------------"
bash "${SCRIPT_DIR}/submit_intent_training.sh"
echo ""

# Step 3: Recommender Training
echo "[3/3] Training Recommender System..."
echo "------------------------------------------"
bash "${SCRIPT_DIR}/submit_recsys_training.sh"
echo ""

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Models saved to GCS bucket."
echo "Check Dataproc job history for details."
