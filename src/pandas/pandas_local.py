"""
pandas_local.py - Pandas Baseline Implementation (without Undersampling)

Basic Pandas implementation for purchase intent prediction.
This version does NOT include undersampling - used as a baseline comparison.

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# SETTINGS
INPUT_FILE = "data/2019-Oct.csv"
SAMPLE_RATE = 1.0  # 1.0 = Full data (100%). If crashes, try 0.1 (10%).


def run_pandas_benchmark():
    print(f"--- PANDAS BENCHMARK STARTING (Sample Rate: {SAMPLE_RATE}) ---")

    # ---------------------------------------------------------
    # STEP 1: Data Loading
    # ---------------------------------------------------------
    start_total = time.time()
    print("1. Loading data from CSV...")

    try:
        # Reading data without type specification for fair comparison with Spark.
        # Specifying dtypes in Pandas would save RAM but we want a fair benchmark.
        # For sampling, using skiprows logic is possible but sampling after read_csv
        # is more common (when RAM is available).

        df = pd.read_csv(INPUT_FILE)

        if SAMPLE_RATE < 1.0:
            print(f"   Data too large, taking {int(SAMPLE_RATE*100)}% sample...")
            df = df.sample(frac=SAMPLE_RATE, random_state=42)

        print(f"   Rows Loaded: {len(df)}")

    except MemoryError:
        print("!!! ERROR: MemoryError. Data doesn't fit in RAM.")
        print("   Report Note: 'Pandas implementation failed during data loading due to OOM.'")
        return

    # ---------------------------------------------------------
    # STEP 2: Preprocessing & Leakage Prevention
    # ---------------------------------------------------------
    print("2. Applying Preprocessing and Leakage-Free Logic...")
    start_preprocess = time.time()

    # Convert date format (slow operation in Pandas)
    df['event_time'] = pd.to_datetime(df['event_time'])

    # Fill missing values
    df['category_code'] = df['category_code'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')

    # --- Complex Logic: Find Purchase Timestamp ---
    # Pandas equivalent of Spark's Window function:
    print("   Calculating purchase timestamps (Merge operation)...")

    # Get only purchases
    purchases = df[df['event_type'] == 'purchase'][['user_session', 'event_time']]

    # Find FIRST purchase time for each session
    first_purchase = purchases.groupby('user_session')['event_time'].min().reset_index()
    first_purchase.columns = ['user_session', 'purchase_timestamp']

    # Merge with main table (Left Join) - THIS OPERATION USES A LOT OF RAM!
    df = df.merge(first_purchase, on='user_session', how='left')

    # Filter: If no purchase (NaT) OR event_time <= purchase_timestamp
    print("   Removing future data (Filtering)...")
    mask = (df['purchase_timestamp'].isna()) | (df['event_time'] <= df['purchase_timestamp'])
    df_clean = df[mask].copy()

    # Memory cleanup (Delete old df)
    del df, purchases, first_purchase
    import gc
    gc.collect()

    # ---------------------------------------------------------
    # STEP 3: Feature Engineering (Aggregation)
    # ---------------------------------------------------------
    print("3. Feature Engineering (GroupBy & Aggregation)...")

    # Pandas GroupBy
    # Label creation logic: 1 if purchase_timestamp is filled, 0 otherwise
    # Need to derive label again after groupby.

    # Helper columns (for Spark's count(when(...)) logic)
    df_clean['is_view'] = (df_clean['event_type'] == 'view').astype(int)
    df_clean['is_cart'] = (df_clean['event_type'] == 'cart').astype(int)

    # Aggregation dictionary
    agg_funcs = {
        'purchase_timestamp': lambda x: 1 if x.notna().any() else 0,  # Label
        'is_view': 'sum',          # view_count
        'is_cart': 'sum',          # cart_count
        'event_time': lambda x: (x.max() - x.min()).total_seconds(),  # Duration
        'price': ['mean', 'max'],  # avg_price, max_price
        'product_id': 'nunique'    # unique_items
    }

    session_features = df_clean.groupby('user_session').agg(agg_funcs)

    # Fix column names
    session_features.columns = [
        'label', 'view_count', 'cart_count', 'session_duration',
        'avg_price', 'max_price', 'unique_items'
    ]

    # Fill null prices with 0
    session_features = session_features.fillna(0)

    end_preprocess = time.time()
    preprocess_time = end_preprocess - start_preprocess
    print(f"   Preprocessing Time: {preprocess_time:.2f} seconds")

    # ---------------------------------------------------------
    # STEP 4: Model Training
    # ---------------------------------------------------------
    print("4. Model Training (Scikit-Learn Random Forest)...")

    X = session_features.drop('label', axis=1)
    y = session_features['label']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    # Using n_jobs=-1 to use all cores (fair comparison with Spark)
    rf = RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=-1, random_state=42)

    start_train = time.time()
    rf.fit(X_train, y_train)
    end_train = time.time()

    training_time = end_train - start_train
    print(f"   Training Time: {training_time:.2f} seconds")

    # ---------------------------------------------------------
    # STEP 5: Results
    # ---------------------------------------------------------
    print("5. Evaluation...")
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("-" * 30)
    print(f"PANDAS RESULTS (Sample Rate: {SAMPLE_RATE})")
    print("-" * 30)
    print(f"Total Time (Load+Preprocess+Train): {time.time() - start_total:.2f} seconds")
    print(f"Preprocessing Time: {preprocess_time:.2f} seconds")
    print(f"Training Time     : {training_time:.2f} seconds")
    print("-" * 30)
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("-" * 30)


if __name__ == "__main__":
    run_pandas_benchmark()
