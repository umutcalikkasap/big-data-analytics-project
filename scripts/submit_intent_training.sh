#!/bin/bash
# =============================================================================
# submit_intent_training.sh - Submit Intent Model Training to Dataproc
#
# This script submits the Random Forest training job to a Dataproc cluster.
#
# Usage:
#   ./scripts/submit_intent_training.sh
# =============================================================================

set -e

# Configuration - MODIFY THESE VALUES
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
CLUSTER_NAME="ecommerce-cluster"
GCS_BUCKET="gs://your-bucket-name"

# Input/Output paths
INPUT_PATH="${GCS_BUCKET}/processed/session_features"
MODEL_OUTPUT="${GCS_BUCKET}/models/intent_model"

# Model hyperparameters
NUM_TREES=50
MAX_DEPTH=10

echo "Submitting Intent Model Training job..."
echo "Cluster: ${CLUSTER_NAME}"
echo "Features: ${INPUT_PATH}"
echo "Model Output: ${MODEL_OUTPUT}"
echo "Trees: ${NUM_TREES}, Depth: ${MAX_DEPTH}"

gcloud dataproc jobs submit pyspark \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --cluster="${CLUSTER_NAME}" \
    src/train_intent.py \
    -- \
    --input "${INPUT_PATH}" \
    --model-output "${MODEL_OUTPUT}" \
    --trees "${NUM_TREES}" \
    --depth "${MAX_DEPTH}"

echo "Intent training job submitted successfully!"
