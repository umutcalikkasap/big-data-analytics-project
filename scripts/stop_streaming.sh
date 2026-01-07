#!/bin/bash
# stop_streaming.sh - Stop All Streaming Components
#
# Bu script tum streaming bilesenlerini durdurur.
#
# Usage:
#   ./scripts/stop_streaming.sh
#
# Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
# Course: YZV411E Big Data Analytics - Istanbul Technical University

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  E-COMMERCE STREAMING PIPELINE - STOP     ${NC}"
echo -e "${BLUE}============================================${NC}"

# Project directory
PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"

# Kill Python processes
echo -e "${BLUE}Stopping Python processes...${NC}"

pkill -f "kafka_producer" 2>/dev/null && echo "Stopped: kafka_producer" || echo "kafka_producer not running"
pkill -f "stream_processor" 2>/dev/null && echo "Stopped: stream_processor" || echo "stream_processor not running"
pkill -f "streamlit" 2>/dev/null && echo "Stopped: streamlit" || echo "streamlit not running"

# Stop Docker
echo ""
echo -e "${BLUE}Stopping Docker services...${NC}"

if [ -f "$PROJECT_DIR/docker/docker-compose.yml" ]; then
    docker-compose -f "$PROJECT_DIR/docker/docker-compose.yml" down
    echo -e "${GREEN}Docker services stopped${NC}"
else
    echo "docker-compose.yml not found"
fi

# Clear temp files
echo ""
echo -e "${BLUE}Cleaning up temp files...${NC}"

rm -rf /tmp/spark-streaming-checkpoints 2>/dev/null || true
rm -f /tmp/streaming_metrics.json 2>/dev/null || true

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}All streaming components stopped!${NC}"
echo -e "${GREEN}============================================${NC}"
