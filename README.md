# E-Commerce User Behavior Analysis & Purchase Intent Prediction

[![PySpark](https://img.shields.io/badge/PySpark-3.x-orange)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)

A scalable Big Data analytics pipeline for predicting real-time purchase intent on e-commerce platforms. Processing **42.4 million** user interaction events to classify session-level purchase behavior.

**Course:** YZV411E Big Data Analytics - Istanbul Technical University
**Authors:** Abdulkadir Külçe (150210322), Berkay Türk (150220320), Umut Çalıkkasap (150210721)

---

## Results

### Intent Prediction Model (Random Forest)

| Metric | Value |
|--------|-------|
| **AUC (Area Under ROC)** | **0.9276** |
| **F1-Score** | **0.8366** |
| Recall (Weighted) | 0.8385 |
| Accuracy | 0.8385 |

**Model Parameters:** `n_estimators=20`, `max_depth=5`

**Feature Importance:**
- `cart_count`: 0.42 (dominant predictor)
- `session_duration`: 0.21

---

## The "Small Data Paradox"

### Why Pandas Was Faster

For this ~6GB dataset, Pandas outperformed Spark in raw latency:

| Task | Pandas (Single-Node) | Spark (Distributed) |
|------|---------------------|---------------------|
| Preprocessing | ~13 min | ~0.5 min (Cluster) |
| Training | **1.84 sec** | 485.49 sec |
| Fault Tolerance | Low (OOM Risk) | **High (Disk Spill)** |

**Why Pandas excels for small data:**
- Zero-copy memory access via optimized C/Cython routines
- No distributed overhead (JVM startup, shuffle, serialization)
- Extremely fast when data fits entirely in RAM

**Why Spark is still necessary:**
1. **Fault Tolerance:** RDD lineage enables automatic recovery of lost partitions
2. **Disk Spilling:** Gracefully handles data exceeding RAM (Pandas raises MemoryError)
3. **Horizontal Scalability:** Linear scaling by adding more nodes

| Data Size | Recommendation |
|-----------|---------------|
| < 10 GB | Pandas (prototyping), Spark (production) |
| > 10 GB | Spark (required) |

---

## Handling Class Imbalance

### The Problem

E-commerce datasets are **highly imbalanced**. In our dataset:
- **Conversion Rate:** Only 6.8% of sessions result in a purchase
- **Event Distribution:** View (96.1%), Cart (2.2%), Purchase (1.7%)

When training on imbalanced data, the model learns to always predict the majority class:
- Baseline Accuracy: **93%** (looks good!)
- Baseline Recall: **0%** (always predicts "no purchase")
- Baseline F1-Score: **0.51**

### Our Solution: Random Undersampling

We implemented **Random Undersampling** with a 1:1 ratio:

```
Before Undersampling:
├── Buyers (Label=1):     ~610,000 sessions (6.8%)
└── Non-Buyers (Label=0): ~8,400,000 sessions (93.2%)

After Undersampling:
├── Buyers (Label=1):     ~610,000 sessions (50%)
└── Non-Buyers (Label=0): ~610,000 sessions (50%)  ← Randomly sampled
```

**Implementation (Spark):**
```python
minority_df = df.filter(F.col("label") == 1)  # Keep 100% of buyers
majority_df = df.filter(F.col("label") == 0)

# Calculate sampling ratio
ratio = minority_count / majority_count  # ≈ 0.073

# Downsample majority class
majority_sampled = majority_df.sample(fraction=ratio, seed=42)

# Combine for balanced dataset
df_balanced = minority_df.union(majority_sampled)
```

### Results After Undersampling

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| F1-Score | 0.51 | **0.84** | +64% |
| Recall | 0.00 | **0.84** | ∞ |
| AUC | 0.72 | **0.93** | +29% |

### Why Undersampling Works

1. **Forces the model to learn minority class patterns** instead of defaulting to majority
2. **Aggressive but effective** for highly imbalanced data
3. **Trade-off:** We discard ~87% of non-buyer data, but gain significant recall

### Alternative Approaches (Not Used)

| Method | Pros | Cons |
|--------|------|------|
| **SMOTE** | Creates synthetic samples | Computationally expensive on 42M rows |
| **Class Weights** | No data loss | Less effective for extreme imbalance |
| **Oversampling** | Keeps all data | Risk of overfitting on duplicates |

---

## Project Structure

```
project files/
├── src/
│   ├── spark/                        # PySpark (Distributed)
│   │   ├── preprocessing.py          # Leakage-free feature engineering
│   │   ├── train_intent.py           # Random Forest training
│   │   └── spark_local_undersamp.py  # Single-file version
│   └── pandas/                       # Pandas (Single-Node)
│       ├── pandas_baseline.py        # Modular implementation
│       ├── pandas_undersamp.py       # Undersampling version
│       └── pandas_local.py           # Baseline version
├── benchmarks/
│   └── compare_frameworks.py         # Pandas vs Spark comparison
├── scripts/
│   ├── create_cluster.sh             # Dataproc cluster setup
│   ├── submit_preprocessing.sh
│   ├── submit_intent_training.sh
│   └── run_full_pipeline.sh
├── data/
│   └── 2019-Oct.csv                  # Dataset (~6GB)
├── notebooks/
│   └── visualization.ipynb           # Result visualization
├── figures/                          # Saved plots
├── test_pipeline.py
├── eda.ipynb
├── Progress_report.pdf
├── big_data_proposal.pdf
├── requirements.txt
└── README.md
```

---

## Technical Approach

### 1. Leakage Prevention (Time-Travel Logic)

A critical challenge in session prediction is **Look-Ahead Bias**. If post-purchase events leak into the training data, the model learns trivial patterns (e.g., "users who view receipts will purchase").

**Our Solution:** Filter out all events occurring AFTER the first purchase in each session.

```python
# 1. Find the first purchase timestamp per session
# 2. Filter out ALL events occurring AFTER that timestamp
# 3. Aggregate only pre-purchase events for feature engineering
```

### 2. Feature Engineering

| Feature | Description |
|---------|-------------|
| `view_count` | Number of product views in session |
| `cart_count` | Number of cart additions |
| `session_duration` | Session length in seconds |
| `avg_price` | Average price of viewed products |
| `max_price` | Maximum product price viewed |
| `unique_items` | Number of unique products viewed |

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
# https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
```

---

## Usage

### Quick Test (1% sample)

```bash
python test_pipeline.py --sample 0.01
```

### Run Pandas (Full data)

```bash
python src/pandas/pandas_undersamp.py
```

### Run Spark (Full data)

```bash
python src/spark/spark_local_undersamp.py
```

### Framework Comparison

```bash
python benchmarks/compare_frameworks.py --sample 0.05
```

### Dataproc (Cloud)

```bash
# Create cluster
./scripts/create_cluster.sh

# Run full pipeline
./scripts/run_full_pipeline.sh
```

---

## Dataset

**Source:** [Kaggle - eCommerce Behavior Data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

| Attribute | Value |
|-----------|-------|
| Total Events | 42,448,764 |
| Size | ~6 GB |
| Unique Users | 3M |
| Unique Products | 166K |
| Time Period | October 2019 |

**Event Distribution:**
- View: 96.1%
- Cart: 2.2%
- Purchase: 1.7%

**Key Insights:**
- Nighttime users (22:00-06:00) have **19.91%** purchase rate (higher than daytime)
- **53.5%** of purchases occur without explicit cart event ("Buy Now" behavior)

---

## References

1. M. Zaharia et al., "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing," USENIX NSDI, 2012.
2. L. Breiman, "Random Forests," Machine Learning, 45(1), 2001.
3. H. He and E. A. Garcia, "Learning from Imbalanced Data," IEEE TKDE, vol. 21, no. 9, 2009.
4. Y. Hu, Y. Koren, C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM, 2008.

---

**Istanbul Technical University - Department of AI and Data Engineering**
