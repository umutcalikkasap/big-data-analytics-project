"""
test_pipeline.py - Pipeline Test File

This script tests the entire pipeline with a small data sample.
Validates both Spark and Pandas implementations.

Veri dosyasi otomatik olarak tespit edilir:
  - Oncelik: 2019-Nov-first-week.csv (sample data)
  - Alternatif: 2019-Oct.csv (full data)

Usage:
    python -m tests.test_pipeline              # Auto-detect data, 1% sample
    python -m tests.test_pipeline --sample 0.05  # 5% sample
    python -m tests.test_pipeline --full       # Full Spark test

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import os
import sys
import time
import argparse

# Set project directory (tests/ klasorunun bir ust dizini)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import auto-detection from streaming config
from src.streaming.config import DEFAULT_DATA_FILE, DATA_FILES_PRIORITY, DATA_DIR


def get_available_data_file():
    """
    Mevcut veri dosyasini otomatik tespit et.

    Returns:
        tuple: (filepath, filename, is_sample)
    """
    for filename in DATA_FILES_PRIORITY:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            is_sample = "first-week" in filename.lower() or "sample" in filename.lower()
            return filepath, filename, is_sample

    # Fallback
    return DEFAULT_DATA_FILE, os.path.basename(DEFAULT_DATA_FILE), False


def test_pandas_pipeline(input_file, sample_rate=0.01):
    """Test Pandas pipeline."""
    print("\n" + "="*60)
    print("TEST: Pandas Pipeline")
    print("="*60)

    from src.pandas.pandas_baseline import run_pandas_benchmark

    try:
        metrics = run_pandas_benchmark(input_file, sample_rate)
        if metrics:
            print("\n[OK] Pandas pipeline successful!")
            print(f"     AUC: {metrics['auc']:.4f}")
            print(f"     F1 : {metrics['f1']:.4f}")
            return True
        else:
            print("\n[FAIL] Pandas pipeline failed!")
            return False
    except Exception as e:
        print(f"\n[FAIL] Pandas pipeline error: {e}")
        return False


def test_spark_pipeline(input_file):
    """Test Spark pipeline."""
    print("\n" + "="*60)
    print("TEST: Spark Pipeline (Full)")
    print("="*60)

    try:
        from pyspark.sql import SparkSession, Window
        from pyspark.sql import functions as F
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

        print("[1/5] Starting Spark Session...")
        spark = SparkSession.builder \
            .appName("Test-Pipeline") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "50") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        print("[2/5] Loading data...")
        df = spark.read.csv(input_file, header=True, inferSchema=True)
        df = df.fillna({'category_code': 'unknown', 'brand': 'unknown'})
        print(f"      Row count: {df.count():,}")

        print("[3/5] Leakage Prevention & Feature Engineering...")
        window_spec = Window.partitionBy("user_session")
        df_marked = df.withColumn(
            "purchase_timestamp",
            F.min(F.when(F.col("event_type") == "purchase", F.col("event_time"))).over(window_spec)
        )
        df_clean = df_marked.filter(
            (F.col("purchase_timestamp").isNull()) |
            (F.col("event_time") <= F.col("purchase_timestamp"))
        )

        session_features = df_clean.groupBy("user_session").agg(
            F.max(F.when(F.col("purchase_timestamp").isNotNull(), 1).otherwise(0)).alias("label"),
            F.count(F.when(F.col("event_type") == "view", 1)).alias("view_count"),
            F.count(F.when(F.col("event_type") == "cart", 1)).alias("cart_count"),
            (F.max("event_time").cast("long") - F.min("event_time").cast("long")).alias("session_duration"),
            F.avg("price").alias("avg_price"),
            F.max("price").alias("max_price"),
            F.countDistinct("product_id").alias("unique_items")
        ).fillna(0)

        total = session_features.count()
        buyers = session_features.filter(F.col("label") == 1).count()
        print(f"      Sessions: {total:,}, Buyers: {buyers:,} ({100*buyers/total:.2f}%)")

        print("[4/5] Undersampling & Model Training...")
        minority = session_features.filter(F.col("label") == 1)
        majority = session_features.filter(F.col("label") == 0)
        minority_count = minority.count()
        majority_count = majority.count()
        ratio = minority_count / majority_count
        majority_sampled = majority.sample(False, ratio, 42)
        balanced = minority.union(majority_sampled)

        assembler = VectorAssembler(
            inputCols=['view_count', 'cart_count', 'session_duration', 'avg_price', 'max_price', 'unique_items'],
            outputCol="features"
        )
        data = assembler.transform(balanced).select("label", "features")
        train, test = data.randomSplit([0.8, 0.2], 42)

        rf = RandomForestClassifier(numTrees=20, maxDepth=5, seed=42)
        model = rf.fit(train)

        print("[5/5] Evaluation...")
        predictions = model.transform(test)

        binary_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
        multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

        auc = binary_eval.evaluate(predictions)
        f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})

        spark.stop()

        print(f"\n[OK] Spark pipeline successful!")
        print(f"     AUC: {auc:.4f}")
        print(f"     F1 : {f1:.4f}")
        return True

    except Exception as e:
        print(f"\n[FAIL] Spark pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_spark(input_file):
    """Quick Spark syntax test (without loading data)."""
    print("\n" + "="*60)
    print("TEST: Spark Syntax Check")
    print("="*60)

    try:
        print("[1/2] Import check...")
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        print("      Imports OK")

        print("[2/2] Starting Spark Session...")
        spark = SparkSession.builder \
            .appName("Syntax-Test") \
            .config("spark.driver.memory", "1g") \
            .master("local[1]") \
            .getOrCreate()
        spark.stop()
        print("      Spark Session OK")

        print("\n[OK] Spark installation verified!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Spark error: {e}")
        return False


def test_streaming_imports():
    """Test streaming module imports."""
    print("\n" + "="*60)
    print("TEST: Streaming Module Imports")
    print("="*60)

    try:
        print("[1/3] Config import...")
        from src.streaming.config import KAFKA_CONFIG, FEATURE_COLS, DEFAULT_DATA_FILE
        print(f"      Default data: {os.path.basename(DEFAULT_DATA_FILE)}")

        print("[2/3] Metrics store import...")
        from src.streaming.metrics_store import MetricsStore
        store = MetricsStore()
        print("      MetricsStore OK")

        print("[3/3] Online model import...")
        from src.streaming.online_model import OnlinePredictor
        model = OnlinePredictor()
        prob = model.predict_proba({'view_count': 5, 'cart_count': 1, 'session_duration': 100,
                                     'avg_price': 50, 'max_price': 100, 'unique_items': 3})
        print(f"      OnlinePredictor OK (test prediction: {prob:.2%})")

        print("\n[OK] Streaming modules verified!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Streaming import error: {e}")
        return False


def run_all_tests(input_file, sample_rate=0.01, full_test=False):
    """Run all tests."""
    print("="*60)
    print("E-COMMERCE INTENT PREDICTION - TEST SUITE")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Sample Rate: {sample_rate if not full_test else 'FULL'}")

    results = {}
    start = time.time()

    # Test 1: Streaming Imports
    results['streaming_imports'] = test_streaming_imports()

    # Test 2: Spark Syntax
    results['spark_syntax'] = test_quick_spark(input_file)

    # Test 3: Pandas Pipeline
    results['pandas'] = test_pandas_pipeline(input_file, sample_rate)

    # Test 4: Full Spark Pipeline (optional)
    if full_test:
        results['spark_full'] = test_spark_pipeline(input_file)

    # Summary
    elapsed = time.time() - start
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print(f"\nTotal time: {elapsed:.2f} seconds")

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARNING] Some tests failed!")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Test Suite")
    parser.add_argument("--input", "-i", default=None,
                        help="CSV file path (auto-detected if not specified)")
    parser.add_argument("--sample", "-s", type=float, default=0.01,
                        help="Sample rate (0.0-1.0)")
    parser.add_argument("--full", action="store_true",
                        help="Also run full Spark test")

    args = parser.parse_args()

    # Auto-detect data file if not specified
    if args.input is None:
        input_file, filename, is_sample = get_available_data_file()
        print(f"\n[AUTO-DETECT] Using: {filename}")
        if is_sample:
            print("[INFO] Sample data detected - ideal for quick testing")
    else:
        input_file = args.input

    # File check
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        print("\nAvailable options:")
        print("  1. Place data file in data/ folder")
        print("  2. Download from: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store")
        print("  3. Use --input flag to specify custom path")
        sys.exit(1)

    success = run_all_tests(input_file, args.sample, args.full)
    sys.exit(0 if success else 1)
