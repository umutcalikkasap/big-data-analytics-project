#!/bin/bash
# start_streaming.sh - Start All Streaming Components
#
# Bu script tum streaming bilesenlerini sirayla baslatir:
# 1. Docker services (Kafka, Zookeeper, Redis)
# 2. Kafka Producer (CSV -> Kafka)
# 3. Spark Streaming Processor
# 4. Streamlit Dashboard
#
# Usage:
#   ./scripts/start_streaming.sh          # Normal start
#   ./scripts/start_streaming.sh --demo   # Quick demo (10K events)
#
# Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
# Course: YZV411E Big Data Analytics - Istanbul Technical University

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
cd "$PROJECT_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  E-COMMERCE STREAMING PIPELINE - START    ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Project: ${GREEN}$PROJECT_DIR${NC}"
echo ""

# Parse arguments
DEMO_MODE=false
EVENT_LIMIT=""

if [[ "$1" == "--demo" ]]; then
    DEMO_MODE=true
    EVENT_LIMIT="--limit 10000"
    echo -e "${YELLOW}Running in DEMO mode (10K events)${NC}"
fi

# Check Docker
echo -e "${BLUE}[1/4] Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found! Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Docker daemon not running! Please start Docker.${NC}"
    exit 1
fi

echo -e "${GREEN}Docker OK${NC}"

# Start Docker services
echo ""
echo -e "${BLUE}[2/4] Starting Docker services (Kafka, Zookeeper, Redis)...${NC}"

if [ -f "docker/docker-compose.yml" ]; then
    docker-compose -f docker/docker-compose.yml up -d

    echo "Waiting for Kafka to be ready..."
    sleep 15

    # Check if Kafka is healthy
    if docker ps | grep -q "kafka"; then
        echo -e "${GREEN}Kafka started successfully${NC}"
    else
        echo -e "${YELLOW}Warning: Kafka may not be fully ready${NC}"
    fi
else
    echo -e "${YELLOW}docker-compose.yml not found. Skipping Docker setup.${NC}"
    echo -e "${YELLOW}You can run in mock mode with --mock flag${NC}"
fi

# Start Kafka Producer
echo ""
echo -e "${BLUE}[3/4] Starting Kafka Producer...${NC}"
echo "Producer will simulate events from CSV to Kafka"

# Run producer in background
python -m src.streaming.kafka_producer $EVENT_LIMIT &
PRODUCER_PID=$!
echo -e "Producer PID: ${GREEN}$PRODUCER_PID${NC}"

# Wait a bit for producer to start
sleep 5

# Start Spark Streaming Processor
echo ""
echo -e "${BLUE}[4/4] Starting Spark Streaming Processor...${NC}"

# Run processor in background
python -m src.streaming.stream_processor &
SPARK_PID=$!
echo -e "Spark PID: ${GREEN}$SPARK_PID${NC}"

# Wait for processing to start
sleep 10

# Start Streamlit Dashboard
echo ""
echo -e "${BLUE}Starting Streamlit Dashboard...${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Dashboard URL: http://localhost:8501${NC}"
echo -e "${GREEN}Kafka UI:      http://localhost:8080${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all components${NC}"

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo -e "${BLUE}Stopping all components...${NC}"

    # Kill background processes
    kill $PRODUCER_PID 2>/dev/null || true
    kill $SPARK_PID 2>/dev/null || true

    # Stop Docker
    if [ -f "docker/docker-compose.yml" ]; then
        docker-compose -f docker/docker-compose.yml down
    fi

    echo -e "${GREEN}All components stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Streamlit (foreground)
streamlit run dashboard/app.py --server.port 8501

# If streamlit exits, cleanup
cleanup
