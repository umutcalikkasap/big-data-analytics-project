"""
pandas_undersamp.py - Pandas Implementation with Undersampling

Single-file implementation for purchase intent prediction using Pandas.
Includes leakage prevention and random undersampling for class balancing.

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
SAMPLE_RATE = 1.0


def run_pandas_benchmark():
    print(f"--- PANDAS BENCHMARK (UNDERSAMPLING) STARTING ---")

    # ---------------------------------------------------------
    # STEP 1: Data Loading
    # ---------------------------------------------------------
    start_total = time.time()
    print("1. Loading data from CSV...")

    try:
        df = pd.read_csv(INPUT_FILE)
        if SAMPLE_RATE < 1.0:
            df = df.sample(frac=SAMPLE_RATE, random_state=42)
        print(f"   Rows Loaded: {len(df)}")
    except MemoryError:
        print("!!! ERROR: MemoryError. Data doesn't fit in RAM.")
        return

    # ---------------------------------------------------------
    # STEP 2: Preprocessing & Leakage Prevention
    # ---------------------------------------------------------
    print("2. Applying Preprocessing and Leakage-Free Logic...")
    start_preprocess = time.time()

    df['event_time'] = pd.to_datetime(df['event_time'])
    df['category_code'] = df['category_code'].fillna('unknown')
    df['brand'] = df['brand'].fillna('unknown')

    # Find purchase timestamp
    purchases = df[df['event_type'] == 'purchase'][['user_session', 'event_time']]
    first_purchase = purchases.groupby('user_session')['event_time'].min().reset_index()
    first_purchase.columns = ['user_session', 'purchase_timestamp']

    # Merge
    df = df.merge(first_purchase, on='user_session', how='left')

    # Filter (Remove future events)
    mask = (df['purchase_timestamp'].isna()) | (df['event_time'] <= df['purchase_timestamp'])
    df_clean = df[mask].copy()

    # Memory cleanup
    del df, purchases, first_purchase
    import gc
    gc.collect()

    # ---------------------------------------------------------
    # STEP 3: Feature Engineering (Aggregation)
    # ---------------------------------------------------------
    print("3. Feature Engineering (GroupBy & Aggregation)...")

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

    # Fix column names
    session_features.columns = [
        'label', 'view_count', 'cart_count', 'session_duration',
        'avg_price', 'max_price', 'unique_items'
    ]
    session_features = session_features.fillna(0)

    end_preprocess = time.time()
    print(f"   Preprocessing Time: {end_preprocess - start_preprocess:.2f} seconds")

    # ---------------------------------------------------------
    # STEP 3.5: UNDERSAMPLING (NEWLY ADDED SECTION)
    # ---------------------------------------------------------
    print("3.5. Applying Undersampling...")

    # Separate classes
    minority_class = session_features[session_features['label'] == 1]
    majority_class = session_features[session_features['label'] == 0]

    print(f"   Buyers Count (Label 1): {len(minority_class)}")
    print(f"   Non-buyers Count (Label 0): {len(majority_class)}")

    # Downsample majority class to match minority class count (1:1 Ratio)
    # This is aggressive but maximizes F1 improvement.
    majority_downsampled = majority_class.sample(n=len(minority_class), random_state=42)

    # Combine
    session_features_balanced = pd.concat([minority_class, majority_downsampled])

    # Shuffle
    session_features = session_features_balanced.sample(frac=1, random_state=42)

    print(f"   Balanced Data Count: {len(session_features)} (Original: {len(minority_class) + len(majority_class)})")

    # ---------------------------------------------------------
    # STEP 4: Model Training
    # ---------------------------------------------------------
    print("4. Model Training (Scikit-Learn Random Forest)...")

    X = session_features.drop('label', axis=1)
    y = session_features['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    print(f"PANDAS RESULTS (UNDERSAMPLED)")
    print("-" * 30)
    print(f"Preprocessing Time: {end_preprocess - start_preprocess:.2f} seconds")
    print(f"Training Time     : {training_time:.2f} seconds")
    print("-" * 30)
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print("-" * 30)

    # Detailed Report (Pay attention to Recall)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    run_pandas_benchmark()
