#!/bin/bash
# =============================================================================
# submit_preprocessing.sh - Submit Preprocessing Job to Google Cloud Dataproc
#
# This script submits the PySpark preprocessing job to a Dataproc cluster.
#
# Usage:
#   ./scripts/submit_preprocessing.sh
#
# Prerequisites:
#   - Google Cloud SDK installed and configured
#   - Dataproc cluster created
#   - Data uploaded to GCS bucket
# =============================================================================

set -e

# Configuration - MODIFY THESE VALUES
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
CLUSTER_NAME="ecommerce-cluster"
GCS_BUCKET="gs://your-bucket-name"

# Input/Output paths
INPUT_PATH="${GCS_BUCKET}/data/2019-Oct.csv"
OUTPUT_PATH="${GCS_BUCKET}/processed/session_features"

# Submit the job
echo "Submitting preprocessing job to Dataproc..."
echo "Cluster: ${CLUSTER_NAME}"
echo "Input: ${INPUT_PATH}"
echo "Output: ${OUTPUT_PATH}"

gcloud dataproc jobs submit pyspark \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --cluster="${CLUSTER_NAME}" \
    src/preprocessing.py \
    -- \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_PATH}"

echo "Preprocessing job submitted successfully!"
