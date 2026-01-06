"""
train_intent.py - Purchase Intent Prediction (PySpark)

Random Forest model for purchase prediction. Uses undersampling for class balancing.
Results: AUC=0.93, F1-Score=0.84

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import argparse
import time


# Model parameters (values from Progress Report)
NUM_TREES = 20
MAX_DEPTH = 5
FEATURE_COLS = ['view_count', 'cart_count', 'session_duration', 'avg_price', 'max_price', 'unique_items']


def create_spark_session(app_name="Intent-Training"):
    """Create Spark Session."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_features(spark, input_path):
    """Load features from Parquet."""
    print(f"[1/4] Loading features: {input_path}")
    df = spark.read.parquet(input_path)

    total = df.count()
    buyers = df.filter(F.col("label") == 1).count()

    print(f"      Total sessions: {total:,}")
    print(f"      Buyers: {buyers:,} ({100*buyers/total:.2f}%)")

    return df


def apply_undersampling(df):
    """
    Random Undersampling (1:1 ratio)

    The majority class (non-buyers) is downsampled to match the minority class (buyers).
    This strategy improved F1-Score from 0.51 to 0.84.
    """
    print("[2/4] Applying Undersampling...")

    minority_df = df.filter(F.col("label") == 1)
    majority_df = df.filter(F.col("label") == 0)

    minority_count = minority_df.count()
    majority_count = majority_df.count()

    print(f"      Buyers (1): {minority_count:,}")
    print(f"      Non-buyers (0): {majority_count:,}")

    # Calculate sampling ratio
    ratio = minority_count / majority_count
    print(f"      Sampling ratio: {ratio:.4f}")

    # Downsample majority class
    majority_sampled = majority_df.sample(withReplacement=False, fraction=ratio, seed=42)
    df_balanced = minority_df.union(majority_sampled)

    print(f"      Balanced dataset: {df_balanced.count():,}")

    return df_balanced


def train_model(df):
    """Train Random Forest model."""
    print(f"[3/4] Training model (Trees={NUM_TREES}, Depth={MAX_DEPTH})...")

    # Create feature vector
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    df_vec = assembler.transform(df).select("label", "features")

    # Train/Test split
    train_data, test_data = df_vec.randomSplit([0.8, 0.2], seed=42)

    print(f"      Train: {train_data.count():,}, Test: {test_data.count():,}")

    # Model
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=NUM_TREES,
        maxDepth=MAX_DEPTH,
        seed=42
    )

    start = time.time()
    model = rf.fit(train_data)
    train_time = time.time() - start

    print(f"      Training time: {train_time:.2f} seconds")

    return model, test_data


def evaluate_model(model, test_data):
    """Evaluate model performance."""
    print("[4/4] Evaluating model...")

    predictions = model.transform(test_data)

    # Metrics
    binary_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
    auc = binary_eval.evaluate(predictions)

    multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
    f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
    recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})

    print(f"\n{'='*40}")
    print("SPARK MODEL RESULTS (UNDERSAMPLED)")
    print(f"{'='*40}")
    print(f"AUC Score     : {auc:.4f}")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"Recall (W)    : {recall:.4f}")
    print(f"{'='*40}")

    return {"auc": auc, "accuracy": accuracy, "f1": f1, "recall": recall}


def run_training(input_path, model_output=None):
    """Main training pipeline."""
    start = time.time()
    spark = create_spark_session()

    try:
        df = load_features(spark, input_path)
        df_balanced = apply_undersampling(df)
        model, test_data = train_model(df_balanced)
        metrics = evaluate_model(model, test_data)

        if model_output:
            print(f"\nSaving model to: {model_output}")
            model.write().overwrite().save(model_output)

        print(f"\nTotal time: {time.time()-start:.2f} seconds")
        return metrics

    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent Prediction Training")
    parser.add_argument("--input", "-i", default="data/session_features")
    parser.add_argument("--model-output", "-m", default=None)
    args = parser.parse_args()

    run_training(args.input, args.model_output)
