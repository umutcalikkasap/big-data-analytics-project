"""
preprocessing.py - PySpark Leakage-Free Feature Engineering

This module processes e-commerce data to generate session-level features.
Critical: "Leakage Prevention" - filtering out post-purchase events.

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
import argparse
import time


def create_spark_session(app_name="ECommerce-Preprocessing"):
    """Create Spark Session."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_clean_data(spark, input_path):
    """Load data and perform basic cleaning."""
    print(f"[1/4] Loading data: {input_path}")

    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df = df.fillna({'category_code': 'unknown', 'brand': 'unknown'})

    count = df.count()
    print(f"      Total rows: {count:,}")
    return df


def apply_leakage_prevention(df):
    """
    Leakage Prevention (Time-Travel Logic)

    If a session has a purchase, delete all events occurring AFTER that moment.
    This prevents the model from learning from "future" information (look-ahead bias).
    """
    print("[2/4] Applying Leakage Prevention...")

    window_spec = Window.partitionBy("user_session")

    df_marked = df.withColumn(
        "purchase_timestamp",
        F.min(
            F.when(F.col("event_type") == "purchase", F.col("event_time"))
        ).over(window_spec)
    )

    df_filtered = df_marked.filter(
        (F.col("purchase_timestamp").isNull()) |
        (F.col("event_time") <= F.col("purchase_timestamp"))
    )

    return df_filtered


def engineer_features(df):
    """
    Session-level feature engineering.

    Features:
    - label: Did purchase occur? (1/0)
    - view_count: Number of product views
    - cart_count: Number of cart additions
    - session_duration: Session length in seconds
    - avg_price: Average price of viewed products
    - max_price: Maximum product price viewed
    - unique_items: Number of unique products viewed
    """
    print("[3/4] Performing Feature Engineering...")

    session_features = df.groupBy("user_session").agg(
        # Label
        F.max(
            F.when(F.col("purchase_timestamp").isNotNull(), 1).otherwise(0)
        ).alias("label"),

        # Event counts
        F.count(F.when(F.col("event_type") == "view", 1)).alias("view_count"),
        F.count(F.when(F.col("event_type") == "cart", 1)).alias("cart_count"),

        # Session duration
        (F.max("event_time").cast("long") - F.min("event_time").cast("long")).alias("session_duration"),

        # Price statistics
        F.avg("price").alias("avg_price"),
        F.max("price").alias("max_price"),

        # Unique products
        F.countDistinct("product_id").alias("unique_items")
    )

    session_features = session_features.fillna(0)
    return session_features


def print_statistics(df):
    """Print dataset statistics."""
    print("[4/4] Calculating statistics...")

    total = df.count()
    buyers = df.filter(F.col("label") == 1).count()

    print(f"\n{'='*40}")
    print(f"Total sessions: {total:,}")
    print(f"Buyers        : {buyers:,} ({100*buyers/total:.2f}%)")
    print(f"Non-buyers    : {total-buyers:,} ({100*(total-buyers)/total:.2f}%)")
    print(f"{'='*40}")


def save_features(df, output_path):
    """Save as Parquet format."""
    print(f"\nSaving to: {output_path}")
    df.write.mode("overwrite").parquet(output_path)
    print("Complete!")


def run_preprocessing(input_path, output_path):
    """Main preprocessing pipeline."""
    start = time.time()
    spark = create_spark_session()

    try:
        df = load_and_clean_data(spark, input_path)
        df_clean = apply_leakage_prevention(df)
        features = engineer_features(df_clean)
        print_statistics(features)
        save_features(features, output_path)

        print(f"\nTotal time: {time.time()-start:.2f} seconds")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce Preprocessing")
    parser.add_argument("--input", "-i", default="data/2019-Oct.csv")
    parser.add_argument("--output", "-o", default="data/session_features")
    args = parser.parse_args()

    run_preprocessing(args.input, args.output)
