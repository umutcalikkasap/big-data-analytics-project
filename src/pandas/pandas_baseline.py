"""
pandas_baseline.py - Pandas Benchmark (with Undersampling)

Implements the same Purchase Intent prediction logic using Pandas.
Used for comparison with Spark implementation.

"Small Data Paradox": Pandas is faster for ~6GB data but lacks
fault tolerance.

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import pandas as pd
import numpy as np
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import argparse


# Model parameters (same as Spark)
N_ESTIMATORS = 20
MAX_DEPTH = 5
FEATURE_COLS = ['view_count', 'cart_count', 'session_duration', 'avg_price', 'max_price', 'unique_items']


def run_pandas_benchmark(input_file, sample_rate=1.0):
    """Main Pandas benchmark function."""
    print(f"{'='*50}")
    print("PANDAS BENCHMARK (UNDERSAMPLING)")
    print(f"{'='*50}")

    # ---------------------------------------------------------
    # STEP 1: Data Loading
    # ---------------------------------------------------------
    start_total = time.time()
    print("\n[1/5] Loading data...")

    try:
        df = pd.read_csv(input_file)
        if sample_rate < 1.0:
            print(f"      Sample rate: {sample_rate*100:.0f}%")
            df = df.sample(frac=sample_rate, random_state=42)
        print(f"      Rows loaded: {len(df):,}")
    except MemoryError:
        print("!!! ERROR: MemoryError - Data doesn't fit in RAM.")
        return None

    # ---------------------------------------------------------
    # STEP 2: Leakage Prevention
    # ---------------------------------------------------------
    print("\n[2/5] Applying Leakage Prevention...")
    start_preprocess = time.time()

    df['event_time'] = pd.to_datetime(df['event_time'])
    df['category_code'] = df['category_code'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')

    # Find first purchase timestamp for each session
    purchases = df[df['event_type'] == 'purchase'][['user_session', 'event_time']]
    first_purchase = purchases.groupby('user_session')['event_time'].min().reset_index()
    first_purchase.columns = ['user_session', 'purchase_timestamp']

    # Merge
    df = df.merge(first_purchase, on='user_session', how='left')

    # Filter: Remove future events
    mask = (df['purchase_timestamp'].isna()) | (df['event_time'] <= df['purchase_timestamp'])
    df_clean = df[mask].copy()

    # Memory cleanup
    del df, purchases, first_purchase
    gc.collect()

    print(f"      Clean data: {len(df_clean):,} rows")

    # ---------------------------------------------------------
    # STEP 3: Feature Engineering
    # ---------------------------------------------------------
    print("\n[3/5] Performing Feature Engineering...")

    df_clean['is_view'] = (df_clean['event_type'] == 'view').astype(int)
    df_clean['is_cart'] = (df_clean['event_type'] == 'cart').astype(int)

    agg_funcs = {
        'purchase_timestamp': lambda x: 1 if x.notna().any() else 0,  # Label
        'is_view': 'sum',
        'is_cart': 'sum',
        'event_time': lambda x: (x.max() - x.min()).total_seconds(),
        'price': ['mean', 'max'],
        'product_id': 'nunique'
    }

    session_features = df_clean.groupby('user_session').agg(agg_funcs)
    session_features.columns = ['label', 'view_count', 'cart_count', 'session_duration',
                                 'avg_price', 'max_price', 'unique_items']
    session_features = session_features.fillna(0)

    end_preprocess = time.time()
    preprocess_time = end_preprocess - start_preprocess

    print(f"      Total sessions: {len(session_features):,}")
    print(f"      Preprocessing time: {preprocess_time:.2f} seconds")

    # ---------------------------------------------------------
    # STEP 4: Undersampling
    # ---------------------------------------------------------
    print("\n[4/5] Applying Undersampling...")

    minority_class = session_features[session_features['label'] == 1]
    majority_class = session_features[session_features['label'] == 0]

    print(f"      Buyers (1): {len(minority_class):,}")
    print(f"      Non-buyers (0): {len(majority_class):,}")

    # 1:1 ratio balancing
    majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)
    session_features_balanced = pd.concat([minority_class, majority_downsampled])
    session_features = session_features_balanced.sample(frac=1, random_state=42)

    print(f"      Balanced data: {len(session_features):,}")

    # ---------------------------------------------------------
    # STEP 5: Model Training
    # ---------------------------------------------------------
    print(f"\n[5/5] Training model (Trees={N_ESTIMATORS}, Depth={MAX_DEPTH})...")

    X = session_features[FEATURE_COLS]
    y = session_features['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                 n_jobs=-1, random_state=42)

    start_train = time.time()
    rf.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train

    print(f"      Training time: {training_time:.2f} seconds")

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    total_time = time.time() - start_total

    print(f"\n{'='*50}")
    print("PANDAS RESULTS (UNDERSAMPLED)")
    print(f"{'='*50}")
    print(f"Preprocessing Time : {preprocess_time:.2f} seconds")
    print(f"Training Time      : {training_time:.2f} seconds")
    print(f"Total Time         : {total_time:.2f} seconds")
    print(f"{'-'*50}")
    print(f"AUC Score  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score   : {f1_score(y_test, y_pred):.4f}")
    print(f"{'='*50}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Purchase', 'Purchase']))

    return {
        'auc': roc_auc_score(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'preprocess_time': preprocess_time,
        'training_time': training_time,
        'total_time': total_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandas Benchmark")
    parser.add_argument("--input", "-i", default="data/2019-Oct.csv")
    parser.add_argument("--sample", "-s", type=float, default=1.0,
                        help="Sample rate (0.0-1.0), 1.0 = full data")
    args = parser.parse_args()

    run_pandas_benchmark(args.input, args.sample)
