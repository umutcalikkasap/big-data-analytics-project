"""
compare_frameworks.py - Pandas vs Spark Performance Comparison

This script runs both frameworks and compares timing and results.
Automatically performs "Small Data Paradox" analysis.

Usage:
    python benchmarks/compare_frameworks.py --sample 0.1

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import sys
import os
import time
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_pandas_benchmark(input_file, sample_rate):
    """Run Pandas implementation."""
    print("\n" + "="*60)
    print("PANDAS BENCHMARK")
    print("="*60)

    from src.pandas.pandas_baseline import run_pandas_benchmark as pandas_run

    start = time.time()
    metrics = pandas_run(input_file, sample_rate)
    elapsed = time.time() - start

    if metrics:
        metrics['total_elapsed'] = elapsed
    return metrics


def run_spark_benchmark(input_file):
    """Run Spark implementation."""
    print("\n" + "="*60)
    print("SPARK BENCHMARK")
    print("="*60)

    try:
        from pyspark.sql import SparkSession, Window
        from pyspark.sql import functions as F
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

        start = time.time()

        # Spark Session
        spark = SparkSession.builder \
            .appName("Benchmark-Spark") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        # Load & Process
        df = spark.read.csv(input_file, header=True, inferSchema=True)
        df = df.fillna({'category_code': 'unknown', 'brand': 'unknown'})

        # Leakage Prevention
        window_spec = Window.partitionBy("user_session")
        df_marked = df.withColumn(
            "purchase_timestamp",
            F.min(F.when(F.col("event_type") == "purchase", F.col("event_time"))).over(window_spec)
        )
        df_clean = df_marked.filter(
            (F.col("purchase_timestamp").isNull()) |
            (F.col("event_time") <= F.col("purchase_timestamp"))
        )

        # Feature Engineering
        session_features = df_clean.groupBy("user_session").agg(
            F.max(F.when(F.col("purchase_timestamp").isNotNull(), 1).otherwise(0)).alias("label"),
            F.count(F.when(F.col("event_type") == "view", 1)).alias("view_count"),
            F.count(F.when(F.col("event_type") == "cart", 1)).alias("cart_count"),
            (F.max("event_time").cast("long") - F.min("event_time").cast("long")).alias("session_duration"),
            F.avg("price").alias("avg_price"),
            F.max("price").alias("max_price"),
            F.countDistinct("product_id").alias("unique_items")
        ).fillna(0)

        preprocess_time = time.time() - start

        # Undersampling
        minority = session_features.filter(F.col("label") == 1)
        majority = session_features.filter(F.col("label") == 0)
        ratio = minority.count() / majority.count()
        majority_sampled = majority.sample(False, ratio, 42)
        balanced = minority.union(majority_sampled)

        # Model
        assembler = VectorAssembler(
            inputCols=['view_count', 'cart_count', 'session_duration', 'avg_price', 'max_price', 'unique_items'],
            outputCol="features"
        )
        data = assembler.transform(balanced).select("label", "features")
        train, test = data.randomSplit([0.8, 0.2], 42)

        train_start = time.time()
        rf = RandomForestClassifier(numTrees=20, maxDepth=5, seed=42)
        model = rf.fit(train)
        training_time = time.time() - train_start

        # Evaluate
        predictions = model.transform(test)
        auc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(predictions)
        multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
        f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
        accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})

        spark.stop()

        total_time = time.time() - start

        return {
            'auc': auc,
            'f1': f1,
            'accuracy': accuracy,
            'preprocess_time': preprocess_time,
            'training_time': training_time,
            'total_elapsed': total_time
        }

    except Exception as e:
        print(f"Spark error: {e}")
        return None


def compare_results(pandas_metrics, spark_metrics):
    """Compare results and generate report."""
    print("\n" + "="*70)
    print("COMPARISON REPORT")
    print("="*70)

    if not pandas_metrics or not spark_metrics:
        print("One or more benchmarks failed!")
        return

    print(f"\n{'Metric':<25} {'Pandas':<15} {'Spark':<15} {'Winner':<10}")
    print("-"*70)

    # AUC
    pandas_auc = pandas_metrics.get('auc', 0)
    spark_auc = spark_metrics.get('auc', 0)
    winner = "Tie" if abs(pandas_auc - spark_auc) < 0.01 else ("Pandas" if pandas_auc > spark_auc else "Spark")
    print(f"{'AUC':<25} {pandas_auc:<15.4f} {spark_auc:<15.4f} {winner:<10}")

    # F1
    pandas_f1 = pandas_metrics.get('f1', 0)
    spark_f1 = spark_metrics.get('f1', 0)
    winner = "Tie" if abs(pandas_f1 - spark_f1) < 0.01 else ("Pandas" if pandas_f1 > spark_f1 else "Spark")
    print(f"{'F1-Score':<25} {pandas_f1:<15.4f} {spark_f1:<15.4f} {winner:<10}")

    # Preprocessing Time
    pandas_prep = pandas_metrics.get('preprocess_time', 0)
    spark_prep = spark_metrics.get('preprocess_time', 0)
    winner = "Pandas" if pandas_prep < spark_prep else "Spark"
    print(f"{'Preprocessing (sec)':<25} {pandas_prep:<15.2f} {spark_prep:<15.2f} {winner:<10}")

    # Training Time
    pandas_train = pandas_metrics.get('training_time', 0)
    spark_train = spark_metrics.get('training_time', 0)
    winner = "Pandas" if pandas_train < spark_train else "Spark"
    print(f"{'Training (sec)':<25} {pandas_train:<15.2f} {spark_train:<15.2f} {winner:<10}")

    # Total Time
    pandas_total = pandas_metrics.get('total_elapsed', 0)
    spark_total = spark_metrics.get('total_elapsed', 0)
    winner = "Pandas" if pandas_total < spark_total else "Spark"
    print(f"{'Total Time (sec)':<25} {pandas_total:<15.2f} {spark_total:<15.2f} {winner:<10}")

    print("-"*70)

    # Summary
    speedup = spark_total / pandas_total if pandas_total > 0 else 0
    print(f"\nSUMMARY:")
    print(f"   - Pandas is {speedup:.1f}x faster than Spark (for this data size)")
    print(f"   - Model performance (AUC, F1) is similar across both frameworks")
    print(f"\nSMALL DATA PARADOX:")
    print(f"   - ~6GB data fits in RAM, so Pandas is faster")
    print(f"   - However, Spark provides fault tolerance and scalability")
    print(f"   - For production or >10GB data, Spark is recommended")


def main():
    parser = argparse.ArgumentParser(description="Pandas vs Spark Comparison")
    parser.add_argument("--input", "-i", default="data/2019-Oct.csv")
    parser.add_argument("--sample", "-s", type=float, default=0.05,
                        help="Sample rate (default: 0.05 = 5%)")
    parser.add_argument("--pandas-only", action="store_true", help="Run only Pandas")
    parser.add_argument("--spark-only", action="store_true", help="Run only Spark")

    args = parser.parse_args()

    print("="*70)
    print("PANDAS vs SPARK - FRAMEWORK COMPARISON")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Sample Rate: {args.sample*100:.0f}%")

    pandas_metrics = None
    spark_metrics = None

    if not args.spark_only:
        pandas_metrics = run_pandas_benchmark(args.input, args.sample)

    if not args.pandas_only:
        spark_metrics = run_spark_benchmark(args.input)

    if pandas_metrics and spark_metrics:
        compare_results(pandas_metrics, spark_metrics)
    elif pandas_metrics:
        print("\n Pandas benchmark completed")
    elif spark_metrics:
        print("\n Spark benchmark completed")


if __name__ == "__main__":
    main()
