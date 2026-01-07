"""
Streaming Module - Real-time E-Commerce Analytics

Bu modul Kafka + Spark Streaming ile gercek zamanli
purchase intent prediction saglar.

Components:
    - config: Configuration settings
    - kafka_producer: CSV -> Kafka event simulator
    - stream_processor: Spark Structured Streaming
    - online_model: Incremental learning model
    - metrics_store: Shared state between components

Usage:
    # Start producer
    python -m src.streaming.kafka_producer --limit 10000

    # Start processor
    python -m src.streaming.stream_processor

    # Or use convenience scripts
    ./scripts/start_streaming.sh

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

from .config import (
    KAFKA_CONFIG,
    SPARK_CONFIG,
    FEATURE_COLS,
    PRODUCER_CONFIG,
    MODEL_CONFIG,
    METRICS_CONFIG
)

from .metrics_store import (
    MetricsStore,
    get_metrics,
    get_history,
    get_predictions,
    update_metrics
)

from .online_model import (
    OnlinePredictor,
    HeuristicPredictor,
    create_predictor
)

__all__ = [
    # Config
    'KAFKA_CONFIG',
    'SPARK_CONFIG',
    'FEATURE_COLS',
    'PRODUCER_CONFIG',
    'MODEL_CONFIG',
    'METRICS_CONFIG',
    # Metrics Store
    'MetricsStore',
    'get_metrics',
    'get_history',
    'get_predictions',
    'update_metrics',
    # Online Model
    'OnlinePredictor',
    'HeuristicPredictor',
    'create_predictor',
]
