"""
stream_processor.py - Spark Structured Streaming Processor

Kafka'dan eventleri okur, session-level feature'lar hesaplar,
online model ile prediction yapar.

Mevcut preprocessing.py mantigi streaming'e adapte edilmistir.

Usage:
    python -m src.streaming.stream_processor

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import os
import sys
import logging
from datetime import datetime

# Spark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType,
        LongType, IntegerType
    )
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Warning: PySpark not installed")

from .config import KAFKA_CONFIG, SPARK_CONFIG, FEATURE_COLS
from .online_model import OnlinePredictor, HeuristicPredictor
from .metrics_store import MetricsStore

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# EVENT SCHEMA
# Kafka'dan gelecek JSON event yapisi
# =============================================================================
EVENT_SCHEMA = StructType([
    StructField("event_time", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("product_id", LongType(), True),
    StructField("category_id", LongType(), True),
    StructField("category_code", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("user_id", LongType(), True),
    StructField("user_session", StringType(), True),
    StructField("ingestion_time", StringType(), True)
])


class StreamingProcessor:
    """
    Spark Structured Streaming ile real-time processing.

    Pipeline:
        1. Kafka'dan event oku
        2. JSON parse et
        3. Windowed aggregation ile feature hesapla
        4. Online model ile prediction yap
        5. Metrics store'a yaz
    """

    def __init__(self):
        """StreamingProcessor'i initialize et."""
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark not installed. Run: pip install pyspark")

        self.spark = self._create_spark_session()
        self.metrics_store = MetricsStore()
        self.predictor = OnlinePredictor()

        # Batch counter
        self.batch_count = 0

        logger.info("StreamingProcessor initialized")

    def _create_spark_session(self):
        """
        Spark Session olustur (Kafka destekli).

        Returns:
            SparkSession instance
        """
        # Kafka package for Spark 3.4+
        kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0"

        spark = SparkSession.builder \
            .appName(SPARK_CONFIG["app_name"]) \
            .master(SPARK_CONFIG["master"]) \
            .config("spark.jars.packages", kafka_package) \
            .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
            .config("spark.sql.shuffle.partitions", SPARK_CONFIG["shuffle_partitions"]) \
            .config("spark.streaming.stopGracefullyOnShutdown", "true") \
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")

        logger.info(f"Spark Session created: {spark.version}")
        return spark

    def read_kafka_stream(self):
        """
        Kafka'dan streaming DataFrame oku.

        Returns:
            DataFrame: Raw Kafka messages
        """
        logger.info(f"Connecting to Kafka: {KAFKA_CONFIG['bootstrap_servers']}")
        logger.info(f"Topic: {KAFKA_CONFIG['topic']}")

        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", KAFKA_CONFIG["bootstrap_servers"]) \
            .option("subscribe", KAFKA_CONFIG["topic"]) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()

    def parse_events(self, kafka_df):
        """
        Kafka mesajlarini JSON'dan DataFrame'e parse et.

        Args:
            kafka_df: Raw Kafka DataFrame

        Returns:
            DataFrame: Parsed events
        """
        return kafka_df \
            .selectExpr("CAST(value AS STRING) as json_str") \
            .select(F.from_json(F.col("json_str"), EVENT_SCHEMA).alias("data")) \
            .select("data.*") \
            .withColumn(
                "event_timestamp",
                F.to_timestamp(F.col("event_time"), "yyyy-MM-dd HH:mm:ss 'UTC'")
            )

    def compute_streaming_features(self, events_df):
        """
        Streaming feature engineering.

        Mevcut preprocessing.py'deki engineer_features() mantigi
        streaming windowed aggregation'a adapte edilmistir.

        Args:
            events_df: Parsed events DataFrame

        Returns:
            DataFrame: Session features with window
        """
        # Watermark ekle (late data handling)
        events_with_watermark = events_df \
            .withWatermark("event_timestamp", SPARK_CONFIG["watermark_delay"])

        # Session-based windowed aggregation
        # 5 dakikalik pencere, 30 saniyelik slide
        session_features = events_with_watermark \
            .groupBy(
                F.col("user_session"),
                F.window("event_timestamp", "5 minutes", "30 seconds")
            ) \
            .agg(
                # Event counts (mevcut batch logic ile ayni)
                F.count(F.when(F.col("event_type") == "view", 1)).alias("view_count"),
                F.count(F.when(F.col("event_type") == "cart", 1)).alias("cart_count"),
                F.count(F.when(F.col("event_type") == "purchase", 1)).alias("purchase_count"),

                # Session duration (saniye cinsinden)
                (F.max("event_timestamp").cast("long") -
                 F.min("event_timestamp").cast("long")).alias("session_duration"),

                # Price statistics
                F.avg("price").alias("avg_price"),
                F.max("price").alias("max_price"),

                # Unique items
                F.countDistinct("product_id").alias("unique_items"),

                # Total events (debugging icin)
                F.count("*").alias("total_events"),

                # Latest event time
                F.max("event_timestamp").alias("last_event_time")
            )

        return session_features

    def process_batch(self, batch_df, batch_id):
        """
        Her micro-batch icin calisir.

        1. Feature'lari topla
        2. Prediction yap
        3. Metrics'i guncelle
        4. Ground truth ile model'i update et

        Args:
            batch_df: DataFrame for this batch
            batch_id: Unique batch identifier
        """
        if batch_df.isEmpty():
            logger.debug(f"Batch {batch_id}: Empty")
            return

        self.batch_count += 1
        logger.info(f"Processing batch {batch_id} (total: {self.batch_count})")

        try:
            # Pandas'a cevir (kucuk batch'ler icin OK)
            pdf = batch_df.toPandas()

            # Null handling
            pdf = pdf.fillna({
                'view_count': 0,
                'cart_count': 0,
                'purchase_count': 0,
                'session_duration': 0,
                'avg_price': 0,
                'max_price': 0,
                'unique_items': 0
            })

            # Aggregate metrics
            total_views = int(pdf["view_count"].sum())
            total_carts = int(pdf["cart_count"].sum())
            total_purchases = int(pdf["purchase_count"].sum())
            active_sessions = len(pdf)
            total_events = int(pdf["total_events"].sum()) if "total_events" in pdf else 0

            # Conversion rate (cart -> purchase)
            conversion_rate = (total_purchases / total_carts * 100) if total_carts > 0 else 0.0

            # Her session icin prediction
            predictions = []
            for _, row in pdf.iterrows():
                features = {
                    "view_count": int(row["view_count"]),
                    "cart_count": int(row["cart_count"]),
                    "session_duration": float(row["session_duration"] or 0),
                    "avg_price": float(row["avg_price"] or 0),
                    "max_price": float(row["max_price"] or 0),
                    "unique_items": int(row["unique_items"])
                }

                # Online model prediction
                prob = self.predictor.predict_proba(features)

                has_purchased = int(row["purchase_count"]) > 0

                predictions.append({
                    "session_id": str(row["user_session"]),
                    "purchase_probability": round(prob, 4),
                    "features": features,
                    "has_purchased": has_purchased
                })

                # Ground truth varsa model'i guncelle (online learning)
                if has_purchased:
                    self.predictor.partial_fit(features, label=1)
                elif features["cart_count"] > 0:
                    # Cart ekleyip almayan - negatif ornek
                    # (sadece cart olanlar icin, view-only cok fazla)
                    self.predictor.partial_fit(features, label=0)

            # Metrics store'a yaz
            self.metrics_store.update_metrics({
                "batch_id": batch_id,
                "batch_count": self.batch_count,
                "total_views": total_views,
                "total_carts": total_carts,
                "total_purchases": total_purchases,
                "total_events": total_events,
                "active_sessions": active_sessions,
                "conversion_rate": round(conversion_rate, 2),
                "predictions": predictions[:50],  # Son 50 prediction
                "model_metrics": self.predictor.get_metrics()
            })

            logger.info(
                f"Batch {batch_id}: {active_sessions} sessions, "
                f"{total_views} views, {total_carts} carts, {total_purchases} purchases, "
                f"conversion: {conversion_rate:.1f}%"
            )

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            import traceback
            traceback.print_exc()

    def start(self):
        """Streaming pipeline'i baslat."""
        logger.info("="*60)
        logger.info("SPARK STRUCTURED STREAMING STARTING")
        logger.info("="*60)
        logger.info(f"Kafka: {KAFKA_CONFIG['bootstrap_servers']}")
        logger.info(f"Topic: {KAFKA_CONFIG['topic']}")
        logger.info(f"Trigger: {SPARK_CONFIG['trigger_interval']}")
        logger.info(f"Checkpoint: {SPARK_CONFIG['checkpoint_location']}")
        logger.info("="*60)

        try:
            # 1. Kafka'dan oku
            kafka_stream = self.read_kafka_stream()

            # 2. Parse et
            events = self.parse_events(kafka_stream)

            # 3. Streaming features
            features = self.compute_streaming_features(events)

            # 4. Micro-batch processing
            query = features \
                .writeStream \
                .outputMode("update") \
                .trigger(processingTime=SPARK_CONFIG["trigger_interval"]) \
                .foreachBatch(self.process_batch) \
                .option("checkpointLocation", SPARK_CONFIG["checkpoint_location"]) \
                .start()

            logger.info("Streaming query started. Waiting for data...")
            logger.info("Press Ctrl+C to stop")

            query.awaitTermination()

        except KeyboardInterrupt:
            logger.info("Streaming stopped by user")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Streaming'i durdur ve kaynaklari temizle."""
        logger.info("Stopping streaming processor...")

        # Model'i kaydet
        self.predictor.force_update()

        # Spark'i kapat
        if self.spark:
            self.spark.stop()

        logger.info("Streaming processor stopped")


# =============================================================================
# MOCK PROCESSOR (Kafka olmadan test icin)
# =============================================================================

class MockStreamingProcessor:
    """
    Kafka olmadan test icin mock processor.
    CSV'den okuyarak batch processing simule eder.
    """

    def __init__(self):
        self.metrics_store = MetricsStore()
        self.predictor = HeuristicPredictor()
        self.batch_count = 0
        logger.info("MockStreamingProcessor initialized")

    def process_csv_batch(self, df_batch, batch_id):
        """CSV batch'ini isle."""
        import pandas as pd

        # Basit aggregation
        session_features = df_batch.groupby('user_session').agg({
            'event_type': lambda x: (x == 'view').sum(),
            'price': ['mean', 'max'],
            'product_id': 'nunique'
        })

        # Basitlestirilmis metrics
        total_views = len(df_batch[df_batch['event_type'] == 'view'])
        total_carts = len(df_batch[df_batch['event_type'] == 'cart'])
        total_purchases = len(df_batch[df_batch['event_type'] == 'purchase'])

        self.metrics_store.update_metrics({
            "batch_id": batch_id,
            "total_views": total_views,
            "total_carts": total_carts,
            "total_purchases": total_purchases,
            "active_sessions": len(session_features),
            "conversion_rate": (total_purchases / total_carts * 100) if total_carts > 0 else 0
        })

        self.batch_count += 1
        logger.info(f"Mock batch {batch_id}: {len(session_features)} sessions")

    def start(self, csv_path, batch_size=10000):
        """CSV'den batch processing baslat."""
        import pandas as pd
        import time

        logger.info(f"Starting mock streaming from {csv_path}")

        for batch_id, chunk in enumerate(pd.read_csv(csv_path, chunksize=batch_size)):
            self.process_csv_batch(chunk, batch_id)
            time.sleep(1)  # Simulate processing time

            if batch_id >= 10:  # Demo icin 10 batch
                break

        logger.info("Mock streaming complete")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Spark Structured Streaming Processor"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock processor (no Kafka required)"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV path for mock mode"
    )

    args = parser.parse_args()

    if args.mock:
        from .config import PRODUCER_CONFIG
        csv_path = args.csv or PRODUCER_CONFIG["csv_path"]
        processor = MockStreamingProcessor()
        processor.start(csv_path)
    else:
        processor = StreamingProcessor()
        processor.start()


if __name__ == "__main__":
    main()
