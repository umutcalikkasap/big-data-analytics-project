#!/bin/bash
# =============================================================================
# create_cluster.sh - Create Google Cloud Dataproc Cluster
#
# This script creates an optimized Dataproc cluster for the e-commerce
# analytics pipeline.
#
# Usage:
#   ./scripts/create_cluster.sh
# =============================================================================

set -e

# Configuration - MODIFY THESE VALUES
PROJECT_ID="your-gcp-project-id"
REGION="us-central1"
ZONE="${REGION}-a"
CLUSTER_NAME="ecommerce-cluster"

# Cluster specifications
MASTER_MACHINE_TYPE="n1-standard-4"
WORKER_MACHINE_TYPE="n1-standard-4"
NUM_WORKERS=2

echo "Creating Dataproc cluster: ${CLUSTER_NAME}"
echo "Region: ${REGION}"
echo "Master: ${MASTER_MACHINE_TYPE}"
echo "Workers: ${NUM_WORKERS} x ${WORKER_MACHINE_TYPE}"

gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --master-machine-type="${MASTER_MACHINE_TYPE}" \
    --master-boot-disk-size=100GB \
    --num-workers="${NUM_WORKERS}" \
    --worker-machine-type="${WORKER_MACHINE_TYPE}" \
    --worker-boot-disk-size=100GB \
    --image-version=2.1-debian11 \
    --optional-components=JUPYTER \
    --enable-component-gateway \
    --properties="spark:spark.sql.adaptive.enabled=true,spark:spark.serializer=org.apache.spark.serializer.KryoSerializer"

echo ""
echo "Cluster created successfully!"
echo "Access Jupyter at: https://console.cloud.google.com/dataproc/clusters/${CLUSTER_NAME}/web-interfaces"
