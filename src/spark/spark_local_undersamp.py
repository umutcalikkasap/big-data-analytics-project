"""
spark_local_undersamp.py - Single-file PySpark Implementation

Complete end-to-end pipeline for purchase intent prediction using PySpark.
Includes leakage prevention and random undersampling for class balancing.

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

from pyspark.sql import SparkSession, Window, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time


def run_spark_undersampling():
    # Spark Configuration (Optimized for local execution)
    spark = SparkSession.builder \
        .appName("Spark_Undersampling_Strategy") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    print("--- [STEP 1] Data Preparation and Leakage Prevention ---")
    start_time = time.time()

    # Load Data
    df = spark.read.csv("data/2019-Oct.csv", header=True, inferSchema=True)
    df_clean = df.fillna({'category_code': 'unknown', 'brand': 'unknown'})

    # Leakage Prevention (Time-Travel Logic)
    window_spec = Window.partitionBy("user_session")
    df_marked = df_clean.withColumn(
        "purchase_timestamp",
        F.min(F.when(F.col("event_type") == "purchase", F.col("event_time"))).over(window_spec)
    )

    df_no_leakage = df_marked.filter(
        (F.col("purchase_timestamp").isNull()) |
        (F.col("event_time") <= F.col("purchase_timestamp"))
    )

    # Feature Engineering (Aggregation)
    session_features = df_no_leakage.groupBy("user_session").agg(
        F.max(F.when(F.col("purchase_timestamp").isNotNull(), 1).otherwise(0)).alias("label"),
        F.count(F.when(F.col("event_type") == "view", 1)).alias("view_count"),
        F.count(F.when(F.col("event_type") == "cart", 1)).alias("cart_count"),
        (F.max("event_time").cast("long") - F.min("event_time").cast("long")).alias("session_duration"),
        F.avg("price").alias("avg_price"),
        F.max("price").alias("max_price"),
        F.countDistinct("product_id").alias("unique_items")
    )

    # Fill nulls
    df_ml = session_features.fillna(0)

    preprocess_end = time.time()
    print(f"   Preprocessing Time: {preprocess_end - start_time:.2f} seconds")

    print("\n--- [STEP 2] Undersampling (Class Balancing) ---")

    # 1. Separate classes
    minority_df = df_ml.filter(F.col("label") == 1)
    majority_df = df_ml.filter(F.col("label") == 0)

    # 2. Get counts (this is an action so it takes some time)
    minority_count = minority_df.count()
    majority_count = majority_df.count()

    print(f"   Buyers (1) Count: {minority_count}")
    print(f"   Non-buyers (0) Count: {majority_count}")

    # 3. Calculate ratio (to downsample majority to minority count)
    ratio = minority_count / majority_count
    print(f"   Balancing Ratio (Sampling Ratio): {ratio:.4f}")

    # 4. Sample and combine
    majority_sampled = majority_df.sample(withReplacement=False, fraction=ratio, seed=42)
    df_balanced = minority_df.union(majority_sampled)

    # Shuffle (optional, MLlib already shuffles)
    # df_balanced = df_balanced.orderBy(F.rand())

    print(f"   Balanced Dataset Size (Approx): {df_balanced.count()}")

    print("\n--- [STEP 3] Model Training (Random Forest) ---")

    # Vectorization
    assembler = VectorAssembler(
        inputCols=['view_count', 'cart_count', 'session_duration', 'avg_price', 'max_price', 'unique_items'],
        outputCol="features"
    )

    final_data = assembler.transform(df_balanced).select("label", "features")

    # Split
    train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)

    # Model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20, maxDepth=5)

    train_start = time.time()
    model = rf.fit(train_data)
    train_end = time.time()

    print(f"   Training Time: {train_end - train_start:.2f} seconds")

    print("\n--- [STEP 4] Results ---")
    predictions = model.transform(test_data)

    # Metrics
    binary_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = binary_eval.evaluate(predictions)

    multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    acc = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
    f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
    weighted_recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})

    print("-" * 30)
    print("PYSPARK RESULTS (UNDERSAMPLED)")
    print("-" * 30)
    print(f"Preprocessing Time : {preprocess_end - start_time:.2f} seconds")
    print(f"Training Time      : {train_end - train_start:.2f} seconds")
    print("-" * 30)
    print(f"AUC Score    : {auc:.4f}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Recall (W)   : {weighted_recall:.4f}")
    print("-" * 30)

    spark.stop()


if __name__ == "__main__":
    run_spark_undersampling()
