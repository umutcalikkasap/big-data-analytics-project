"""
config.py - Streaming Pipeline Configuration

Tum streaming bilesenleri icin merkezi konfigurasyon.
Kafka, Spark, Redis ve Model ayarlarini icerir.

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# DATA FILE AUTO-DETECTION
# Oncelik sirasi: Sample data (Nov first week) > Full data (Oct/Nov)
# Bu sayede repoyu klonlayan biri buyuk dosya indirmeden test yapabilir
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Desteklenen veri dosyalari (oncelik sirasina gore)
DATA_FILES_PRIORITY = [
    "2019-Nov-first-week.csv",  # Sample data - quick testing (1.5GB, 11.5M rows)
    "2019-Nov.csv",             # Full November data
    "2019-Oct.csv",             # Full October data (5.3GB, 42M rows)
]

def get_default_data_file():
    """
    Mevcut veri dosyasini otomatik tespit et.
    Oncelik: Sample data > Full data

    Returns:
        str: Bulunan veri dosyasinin tam yolu
    """
    for filename in DATA_FILES_PRIORITY:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            return filepath

    # Fallback: data/ klasorundeki herhangi bir CSV
    if os.path.exists(DATA_DIR):
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if csv_files:
            return os.path.join(DATA_DIR, csv_files[0])

    # Hicbir dosya bulunamadi - varsayilan yol dondur
    return os.path.join(DATA_DIR, "2019-Nov-first-week.csv")

# Auto-detected data file
DEFAULT_DATA_FILE = get_default_data_file()

# =============================================================================
# KAFKA CONFIGURATION
# =============================================================================
KAFKA_CONFIG = {
    "bootstrap_servers": "localhost:9092",
    "topic": "ecommerce-events",
    "group_id": "ecommerce-consumer-group",
    "auto_offset_reset": "earliest"
}

# =============================================================================
# KAFKA PRODUCER CONFIGURATION
# =============================================================================
PRODUCER_CONFIG = {
    "events_per_second": 1000,      # Simulation speed (events/sec)
    "batch_size": 100,              # Events per batch
    "csv_path": DEFAULT_DATA_FILE   # Auto-detected data file
}

# =============================================================================
# SPARK STREAMING CONFIGURATION
# =============================================================================
SPARK_CONFIG = {
    "app_name": "ECommerce-Streaming",
    "master": "local[*]",
    "driver_memory": "4g",
    "shuffle_partitions": "10",
    "trigger_interval": "5 seconds",    # Micro-batch interval
    "watermark_delay": "10 minutes",    # Late data tolerance
    "checkpoint_location": "/tmp/spark-streaming-checkpoints"
}

# =============================================================================
# FEATURE CONFIGURATION
# Mevcut batch processing ile ayni feature'lar
# =============================================================================
FEATURE_COLS = [
    'view_count',
    'cart_count',
    'session_duration',
    'avg_price',
    'max_price',
    'unique_items'
]

# Event types
EVENT_TYPES = ['view', 'cart', 'purchase']

# =============================================================================
# REDIS CONFIGURATION (Optional - for production)
# =============================================================================
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0
}

# =============================================================================
# METRICS STORE CONFIGURATION
# =============================================================================
METRICS_CONFIG = {
    "backend": "file",  # "file" or "redis"
    "file_path": "/tmp/streaming_metrics.json",
    "history_limit": 1000
}

# =============================================================================
# ONLINE MODEL CONFIGURATION
# =============================================================================
MODEL_CONFIG = {
    "learning_rate": 0.01,
    "update_interval": 100,     # Update model every N samples
    "model_path": os.path.join(PROJECT_ROOT, "models", "online_model.pkl"),
    "model_type": "sgd"         # "sgd" or "river"
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
DASHBOARD_CONFIG = {
    "refresh_rate": 3,          # Default refresh rate (seconds)
    "max_predictions_display": 50,
    "chart_history_limit": 100
}

# =============================================================================
# DATA SCHEMA
# CSV kolonlari (Kafka message format icin)
# =============================================================================
DATA_SCHEMA = {
    "columns": [
        "event_time",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_id",
        "user_session"
    ],
    "timestamp_format": "yyyy-MM-dd HH:mm:ss 'UTC'"
}
