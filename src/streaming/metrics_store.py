"""
metrics_store.py - Shared State Between Spark and Streamlit

Bu modul Spark Streaming ve Streamlit Dashboard arasinda
paylasilan metrikleri depolar.

Backends:
    - file: JSON dosyasina yazar (demo icin)
    - redis: Redis'e yazar (production icin)

Usage:
    from src.streaming.metrics_store import MetricsStore, get_metrics

    # Write (from Spark)
    store = MetricsStore()
    store.update_metrics({"views": 100, "carts": 10})

    # Read (from Streamlit)
    metrics = get_metrics()

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import json
import os
from datetime import datetime
from collections import deque
import threading
import logging

from .config import METRICS_CONFIG, REDIS_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsStore:
    """
    Metrikleri depolayan ve paylasan store.

    Attributes:
        backend: "file" veya "redis"
        metrics_file: JSON dosya yolu (file backend icin)
    """

    def __init__(self, backend=None):
        """
        MetricsStore'u initialize et.

        Args:
            backend: "file" veya "redis" (None = config'den)
        """
        self.backend = backend or METRICS_CONFIG["backend"]
        self.metrics_file = METRICS_CONFIG["file_path"]
        self.history_limit = METRICS_CONFIG["history_limit"]
        self.lock = threading.Lock()

        # In-memory cache
        self.current_metrics = {}
        self.metrics_history = deque(maxlen=self.history_limit)

        # Redis client (lazy init)
        self.redis_client = None
        if self.backend == "redis":
            self._init_redis()

    def _init_redis(self):
        """Redis client'i initialize et."""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=REDIS_CONFIG["host"],
                port=REDIS_CONFIG["port"],
                db=REDIS_CONFIG["db"],
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except ImportError:
            logger.warning("redis package not installed. Using file backend.")
            self.backend = "file"
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using file backend.")
            self.backend = "file"

    def update_metrics(self, metrics):
        """
        Metrikleri guncelle.

        Args:
            metrics: dict - Yeni metrikler
        """
        metrics["timestamp"] = datetime.now().isoformat()

        with self.lock:
            self.current_metrics = metrics
            self.metrics_history.append(metrics)

        if self.backend == "redis" and self.redis_client:
            self._update_redis(metrics)
        else:
            self._update_file(metrics)

    def _update_redis(self, metrics):
        """Redis'e yaz."""
        try:
            # Current metrics
            self.redis_client.set("current_metrics", json.dumps(metrics))

            # History (LPUSH for recent first)
            self.redis_client.lpush("metrics_history", json.dumps(metrics))
            self.redis_client.ltrim("metrics_history", 0, self.history_limit - 1)

            # Individual metrics for easy access
            self.redis_client.set("total_views", metrics.get("total_views", 0))
            self.redis_client.set("total_carts", metrics.get("total_carts", 0))
            self.redis_client.set("total_purchases", metrics.get("total_purchases", 0))
            self.redis_client.set("conversion_rate", metrics.get("conversion_rate", 0))

        except Exception as e:
            logger.error(f"Redis update error: {e}")

    def _update_file(self, metrics):
        """Dosyaya yaz."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)

            # Mevcut history'yi oku
            history = []
            if os.path.exists(self.metrics_file):
                try:
                    with open(self.metrics_file, 'r') as f:
                        data = json.load(f)
                        history = data.get("history", [])
                except (json.JSONDecodeError, IOError):
                    history = []

            # Yeni metrics ekle
            history.append(metrics)
            history = history[-self.history_limit:]

            # Yaz (atomic write with temp file)
            temp_file = self.metrics_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump({
                    "current": metrics,
                    "history": history
                }, f, indent=2)

            os.replace(temp_file, self.metrics_file)

        except Exception as e:
            logger.error(f"File update error: {e}")

    def get_current_metrics(self):
        """
        Guncel metrikleri al.

        Returns:
            dict: Current metrics
        """
        if self.backend == "redis" and self.redis_client:
            try:
                data = self.redis_client.get("current_metrics")
                return json.loads(data) if data else {}
            except Exception:
                pass

        # File fallback
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f).get("current", {})
        except Exception:
            pass

        return self.current_metrics

    def get_metrics_history(self, limit=100):
        """
        Metrik gecmisini al.

        Args:
            limit: Maksimum kayit sayisi

        Returns:
            list: Metrics history
        """
        if self.backend == "redis" and self.redis_client:
            try:
                data = self.redis_client.lrange("metrics_history", 0, limit - 1)
                return [json.loads(d) for d in data]
            except Exception:
                pass

        # File fallback
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    history = json.load(f).get("history", [])
                    return history[-limit:]
        except Exception:
            pass

        return list(self.metrics_history)[-limit:]

    def get_predictions(self, limit=50):
        """
        Son prediction'lari al.

        Args:
            limit: Maksimum prediction sayisi

        Returns:
            list: Recent predictions
        """
        current = self.get_current_metrics()
        return current.get("predictions", [])[:limit]

    def clear(self):
        """Tum metrikleri temizle."""
        with self.lock:
            self.current_metrics = {}
            self.metrics_history.clear()

        if self.backend == "redis" and self.redis_client:
            try:
                self.redis_client.delete("current_metrics", "metrics_history")
            except Exception:
                pass
        else:
            try:
                if os.path.exists(self.metrics_file):
                    os.remove(self.metrics_file)
            except Exception:
                pass


# =============================================================================
# GLOBAL CONVENIENCE FUNCTIONS
# Streamlit'ten kolay erisim icin
# =============================================================================

_global_store = None

def _get_store():
    """Global store instance'i al veya olustur."""
    global _global_store
    if _global_store is None:
        _global_store = MetricsStore()
    return _global_store


def get_metrics():
    """
    Guncel metrikleri al.

    Returns:
        dict: Current metrics
    """
    return _get_store().get_current_metrics()


def get_history(limit=100):
    """
    Metrik gecmisini al.

    Args:
        limit: Maksimum kayit sayisi

    Returns:
        list: Metrics history
    """
    return _get_store().get_metrics_history(limit)


def get_predictions(limit=50):
    """
    Son prediction'lari al.

    Args:
        limit: Maksimum prediction sayisi

    Returns:
        list: Recent predictions
    """
    return _get_store().get_predictions(limit)


def update_metrics(metrics):
    """
    Metrikleri guncelle.

    Args:
        metrics: dict - Yeni metrikler
    """
    _get_store().update_metrics(metrics)
