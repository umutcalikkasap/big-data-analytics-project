"""
kafka_producer.py - CSV to Kafka Event Simulator

Bu modul mevcut CSV datasini okuyarak Kafka'ya
gercek zamanli event akisi simule eder.

Demo ve test amacli tasarlanmistir.

Usage:
    python -m src.streaming.kafka_producer
    python -m src.streaming.kafka_producer --limit 10000
    python -m src.streaming.kafka_producer --rate 500

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import json
import time
import argparse
import pandas as pd
from datetime import datetime
import logging
import sys

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: kafka-python not installed. Using mock mode.")

from .config import KAFKA_CONFIG, PRODUCER_CONFIG

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockKafkaProducer:
    """
    Kafka olmadan test icin mock producer.
    Eventleri stdout'a yazar.
    """
    def __init__(self):
        self.message_count = 0
        logger.info("Using MockKafkaProducer (no Kafka connection)")

    def send(self, topic, key=None, value=None):
        self.message_count += 1
        if self.message_count % 1000 == 0:
            logger.info(f"Mock sent {self.message_count} messages")
        return self

    def flush(self):
        logger.info(f"Mock flush: {self.message_count} total messages")

    def close(self):
        pass


class ECommerceEventProducer:
    """
    CSV'den Kafka'ya event ureten producer.

    Features:
        - Chunk-based CSV reading (memory efficient)
        - Rate limiting (configurable events/second)
        - JSON serialization
        - Session-based partitioning
    """

    def __init__(self, use_mock=False):
        """
        Producer'i initialize et.

        Args:
            use_mock: True ise Kafka yerine mock kullan
        """
        self.use_mock = use_mock or not KAFKA_AVAILABLE

        if self.use_mock:
            self.producer = MockKafkaProducer()
        else:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=KAFKA_CONFIG["bootstrap_servers"],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3
                )
                logger.info(f"Connected to Kafka: {KAFKA_CONFIG['bootstrap_servers']}")
            except Exception as e:
                logger.warning(f"Kafka connection failed: {e}. Using mock.")
                self.producer = MockKafkaProducer()
                self.use_mock = True

        self.topic = KAFKA_CONFIG["topic"]
        self.events_per_second = PRODUCER_CONFIG["events_per_second"]
        self.total_sent = 0

    def load_csv_iterator(self, csv_path, chunk_size=10000):
        """
        CSV'yi chunk'lar halinde oku (memory-efficient).

        Args:
            csv_path: CSV dosya yolu
            chunk_size: Her chunk'taki satir sayisi

        Yields:
            DataFrame chunks
        """
        logger.info(f"Loading CSV: {csv_path}")
        return pd.read_csv(csv_path, chunksize=chunk_size)

    def row_to_event(self, row):
        """
        DataFrame row'unu JSON event'e cevir.

        Args:
            row: pandas Series

        Returns:
            dict: Kafka message payload
        """
        return {
            "event_time": str(row["event_time"]),
            "event_type": row["event_type"],
            "product_id": int(row["product_id"]),
            "category_id": int(row["category_id"]) if pd.notna(row["category_id"]) else None,
            "category_code": row["category_code"] if pd.notna(row["category_code"]) else "unknown",
            "brand": row["brand"] if pd.notna(row["brand"]) else "unknown",
            "price": float(row["price"]) if pd.notna(row["price"]) else 0.0,
            "user_id": int(row["user_id"]),
            "user_session": row["user_session"],
            "ingestion_time": datetime.now().isoformat()
        }

    def produce_events(self, csv_path=None, limit=None, rate=None):
        """
        CSV'den eventleri Kafka'ya gonder.

        Args:
            csv_path: CSV dosya yolu (default: config'den)
            limit: Maksimum event sayisi (None = tum data)
            rate: Events per second (None = config'den)

        Returns:
            int: Gonderilen toplam event sayisi
        """
        csv_path = csv_path or PRODUCER_CONFIG["csv_path"]
        rate = rate or self.events_per_second

        logger.info("="*60)
        logger.info("KAFKA PRODUCER STARTING")
        logger.info("="*60)
        logger.info(f"CSV Path: {csv_path}")
        logger.info(f"Topic: {self.topic}")
        logger.info(f"Rate: {rate} events/second")
        logger.info(f"Limit: {limit or 'No limit'}")
        logger.info(f"Mode: {'Mock' if self.use_mock else 'Kafka'}")
        logger.info("="*60)

        event_count = 0
        start_time = time.time()
        last_log_time = start_time

        try:
            for chunk in self.load_csv_iterator(csv_path):
                for _, row in chunk.iterrows():
                    # Limit check
                    if limit and event_count >= limit:
                        break

                    # Convert to event
                    event = self.row_to_event(row)

                    # Send to Kafka (session ID as key for partitioning)
                    self.producer.send(
                        self.topic,
                        key=event["user_session"],
                        value=event
                    )

                    event_count += 1

                    # Rate limiting
                    if rate > 0 and event_count % rate == 0:
                        elapsed = time.time() - start_time
                        expected = event_count / rate
                        if elapsed < expected:
                            time.sleep(expected - elapsed)

                    # Progress logging (every 5 seconds)
                    current_time = time.time()
                    if current_time - last_log_time >= 5:
                        actual_rate = event_count / (current_time - start_time)
                        logger.info(f"Progress: {event_count:,} events sent "
                                  f"({actual_rate:.0f} events/sec)")
                        last_log_time = current_time

                # Break outer loop if limit reached
                if limit and event_count >= limit:
                    break

            # Flush remaining messages
            self.producer.flush()

        except KeyboardInterrupt:
            logger.info("Producer interrupted by user")
        except Exception as e:
            logger.error(f"Producer error: {e}")
            raise
        finally:
            elapsed = time.time() - start_time
            actual_rate = event_count / elapsed if elapsed > 0 else 0

            logger.info("="*60)
            logger.info("PRODUCER COMPLETE")
            logger.info("="*60)
            logger.info(f"Total Events: {event_count:,}")
            logger.info(f"Elapsed Time: {elapsed:.2f} seconds")
            logger.info(f"Average Rate: {actual_rate:.0f} events/second")
            logger.info("="*60)

        self.total_sent = event_count
        return event_count

    def close(self):
        """Producer'i kapat."""
        if hasattr(self.producer, 'close'):
            self.producer.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="E-Commerce CSV to Kafka Producer"
    )
    parser.add_argument(
        "--input", "-i",
        default=PRODUCER_CONFIG["csv_path"],
        help="CSV file path"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Maximum number of events to send"
    )
    parser.add_argument(
        "--rate", "-r",
        type=int,
        default=PRODUCER_CONFIG["events_per_second"],
        help="Events per second (0 = no limit)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock producer (no Kafka required)"
    )

    args = parser.parse_args()

    producer = ECommerceEventProducer(use_mock=args.mock)

    try:
        producer.produce_events(
            csv_path=args.input,
            limit=args.limit,
            rate=args.rate
        )
    finally:
        producer.close()


if __name__ == "__main__":
    main()
