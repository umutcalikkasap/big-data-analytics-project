# E-Commerce Purchase Intent Prediction

[![PySpark](https://img.shields.io/badge/PySpark-3.x-orange)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![Kafka](https://img.shields.io/badge/Kafka-Streaming-black)](https://kafka.apache.org/)

A scalable Big Data analytics pipeline for predicting real-time purchase intent on e-commerce platforms. Processing **42.4 million** user interaction events to classify session-level purchase behavior. Features both **batch processing** (Pandas/Spark) and **real-time streaming** (Kafka + Spark Streaming + Streamlit Dashboard).

**Course:** YZV411E Big Data Analytics - Istanbul Technical University

**Authors:** Abdulkadir Külçe (150210322), Berkay Türk (150220320), Umut Çalıkkasap (150210721)

---

## 🎯 Quick Start (5 Minutes)

### Step 1: Clone & Install

```bash
# Clone the repository
git clone https://github.com/umutcalikkasap/big-data-analytics-intent-prediction.git
cd big-data-analytics-intent-prediction

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get the Data

**Option A - Quick Test (Recommended for First Run):**

We provide a **sample dataset** (`2019-Nov-first-week.csv` - 1.5GB, 11.5M rows) in the `data/` folder. The code **automatically detects** this file, so you can start testing immediately without downloading the full 6GB dataset.

```bash
# Verify sample data exists
ls -lh data/
# Should show: 2019-Nov-first-week.csv (1.5G)
```

**Option B - Full Dataset:**

Download from one of these sources:
- **Kaggle:** [eCommerce Behavior Data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv)
- **Google Drive (Direct):** [Full Datasets (Oct + Nov)](https://drive.google.com/drive/folders/10jq8QIDrHJoZoHYhookPuwo2vRqN_207?usp=drive_link)

Place downloaded files in the `data/` folder. The code auto-detects available files with this priority:
1. `2019-Nov-first-week.csv` (sample - quick testing)
2. `2019-Nov.csv` (full November)
3. `2019-Oct.csv` (full October)

### Step 3: Run Tests

```bash
# Auto-detects sample data and runs quick tests
python -m tests.test_pipeline

# With custom sample rate
python -m tests.test_pipeline --sample 0.05
```

### Step 4: Start Streaming Dashboard (Optional)

```bash
# Install streaming dependencies
pip install kafka-python streamlit plotly

# Start dashboard (no Kafka required for demo)
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## 📈 Results

### Batch Processing Model (Random Forest)

| Metric | Value |
|--------|-------|
| **AUC (Area Under ROC)** | **0.9276** |
| **F1-Score** | **0.8366** |
| Recall (Weighted) | 0.8385 |
| Accuracy | 0.8385 |

**Model Parameters:** `n_estimators=20`, `max_depth=5`

### Streaming Model (Online SGDClassifier)

| Metric | Description |
|--------|-------------|
| Model Type | SGDClassifier (Logistic Loss) |
| Learning | Incremental (partial_fit) |
| Update Strategy | Mini-batch (every 100 samples) |
| Real-time Accuracy | ~70-80% (improves with data) |

### Feature Importance

| Feature | Importance | Description |
|---------|------------|-------------|
| `cart_count` | **0.42** | Number of cart additions (dominant predictor) |
| `session_duration` | 0.21 | Session length in seconds |
| `view_count` | 0.19 | Number of product views |
| `avg_price` | 0.08 | Average price of viewed products |
| `max_price` | 0.06 | Maximum product price viewed |
| `unique_items` | 0.04 | Number of unique products viewed |

---

## 📊 Dataset

### Source

**Kaggle:** [eCommerce Behavior Data from Multi Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv)

**Google Drive (Full Data):** [Download Link](https://drive.google.com/drive/folders/10jq8QIDrHJoZoHYhookPuwo2vRqN_207?usp=drive_link)

### Available Files

| File | Size | Rows | Description |
|------|------|------|-------------|
| `2019-Nov-first-week.csv` | 1.5 GB | 11.5M | **Sample data** (included for quick testing) |
| `2019-Oct.csv` | 5.3 GB | 42.4M | Full October 2019 data |
| `2019-Nov.csv` | 6.0 GB | 67.5M | Full November 2019 data |

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total Events | 42,448,764 |
| Unique Users | ~3M |
| Unique Products | 166K |
| Time Period | October 2019 |

### Event Distribution

| Event Type | Percentage | Count |
|------------|------------|-------|
| View | 96.1% | 40.8M |
| Cart | 2.2% | 934K |
| Purchase | 1.7% | 723K |

### Key Insights

- **Conversion Rate:** Only **6.8%** of sessions result in a purchase
- **Nighttime Behavior:** Users active between 22:00-06:00 have **19.91%** purchase rate (significantly higher than daytime)
- **"Buy Now" Pattern:** **53.5%** of purchases occur without an explicit cart event (direct purchase)
- **Session Duration:** Longer sessions correlate with higher purchase probability

---

## ⚖️ Handling Class Imbalance

### The Problem

E-commerce datasets are **highly imbalanced**. In our dataset:
- **Buyers:** Only 6.8% of sessions (minority class)
- **Non-Buyers:** 93.2% of sessions (majority class)

When training on imbalanced data, the model learns to always predict the majority class:

| Metric | Baseline (No Balancing) |
|--------|------------------------|
| Accuracy | 93% (looks good!) |
| Recall | **0%** (always predicts "no purchase") |
| F1-Score | 0.51 |

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

## ⚡ The "Small Data Paradox"

### Why Pandas Was Faster

For this ~6GB dataset, Pandas outperformed Spark in raw latency:

| Task | Pandas (Single-Node) | Spark (Distributed) |
|------|---------------------|---------------------|
| Preprocessing | ~13 min | ~0.5 min (Cluster) |
| Training | **1.84 sec** | 485.49 sec |
| Fault Tolerance | Low (OOM Risk) | **High (Disk Spill)** |

### Why Pandas Excels for Small Data

- Zero-copy memory access via optimized C/Cython routines
- No distributed overhead (JVM startup, shuffle, serialization)
- Extremely fast when data fits entirely in RAM

### Why Spark is Still Necessary

1. **Fault Tolerance:** RDD lineage enables automatic recovery of lost partitions
2. **Disk Spilling:** Gracefully handles data exceeding RAM (Pandas raises MemoryError)
3. **Horizontal Scalability:** Linear scaling by adding more nodes

### Recommendation

| Data Size | Recommendation |
|-----------|---------------|
| < 10 GB | Pandas (prototyping), Spark (production) |
| > 10 GB | Spark (required) |

---

## 🔬 Technical Approach

### 1. Leakage Prevention (Time-Travel Logic)

A critical challenge in session prediction is **Look-Ahead Bias**. If post-purchase events leak into the training data, the model learns trivial patterns (e.g., "users who view receipts will purchase").

**Our Solution:** Filter out all events occurring AFTER the first purchase in each session.

```python
# 1. Find the first purchase timestamp per session
window_spec = Window.partitionBy("user_session")
df_marked = df.withColumn(
    "purchase_timestamp",
    F.min(F.when(F.col("event_type") == "purchase", F.col("event_time"))).over(window_spec)
)

# 2. Keep only pre-purchase events
df_clean = df_marked.filter(
    (F.col("purchase_timestamp").isNull()) |
    (F.col("event_time") <= F.col("purchase_timestamp"))
)

# 3. Aggregate only pre-purchase events for feature engineering
```

### 2. Feature Engineering

| Feature | Description | Source |
|---------|-------------|--------|
| `view_count` | Number of product views in session | `COUNT(event_type='view')` |
| `cart_count` | Number of cart additions | `COUNT(event_type='cart')` |
| `session_duration` | Session length in seconds | `MAX(time) - MIN(time)` |
| `avg_price` | Average price of viewed products | `AVG(price)` |
| `max_price` | Maximum product price viewed | `MAX(price)` |
| `unique_items` | Number of unique products viewed | `COUNT(DISTINCT product_id)` |

### 3. Online Learning (Streaming)

The streaming pipeline uses **SGDClassifier** with incremental updates:

```python
# Predict purchase probability
prob = model.predict_proba(features)

# Update model with ground truth (when purchase is observed)
model.partial_fit(features, label=1)
```

---

## 📁 Project Structure

```
project files/
├── src/
│   ├── spark/                        # PySpark batch processing
│   │   ├── preprocessing.py          # Leakage-free feature engineering
│   │   ├── train_intent.py           # Random Forest training
│   │   └── spark_local_undersamp.py  # Single-file local version
│   ├── pandas/                       # Pandas batch processing
│   │   ├── pandas_baseline.py        # Modular implementation
│   │   ├── pandas_undersamp.py       # With undersampling
│   │   └── pandas_local.py           # Baseline version
│   └── streaming/                    # Real-time streaming pipeline
│       ├── config.py                 # All configurations (auto-detect data)
│       ├── kafka_producer.py         # CSV → Kafka simulator
│       ├── stream_processor.py       # Spark Structured Streaming
│       ├── online_model.py           # SGDClassifier online learning
│       └── metrics_store.py          # Shared state storage
├── dashboard/                        # Streamlit real-time dashboard
│   └── app.py                        # Main dashboard application
├── tests/                            # Test suite
│   └── test_pipeline.py              # Pipeline tests (auto-detect data)
├── benchmarks/                       # Performance comparisons
│   └── compare_frameworks.py         # Pandas vs Spark benchmark
├── scripts/                          # Automation scripts
│   ├── create_cluster.sh             # GCP Dataproc setup
│   ├── start_streaming.sh            # Start streaming pipeline
│   ├── stop_streaming.sh             # Stop streaming pipeline
│   └── run_full_pipeline.sh          # Cloud pipeline
├── docker/                           # Docker configurations
│   └── docker-compose.yml            # Kafka + Zookeeper + Redis
├── notebooks/                        # Jupyter notebooks
│   ├── eda.ipynb                     # Exploratory data analysis
│   └── visualization.ipynb           # Results visualization
├── data/                             # Data files (gitignored)
│   └── 2019-Nov-first-week.csv       # Sample data for testing
├── models/                           # Trained models
├── report/                           # Project documents
│   ├── Progress_report.pdf
│   └── big_data_proposal.pdf
├── figures/                          # Generated plots
├── requirements.txt
└── README.md
```

---

## 🚀 Usage Guide

### Batch Processing

#### Run Pandas Pipeline

```bash
# Full data (requires ~8GB RAM)
python src/pandas/pandas_undersamp.py

# Or use the modular version with sample rate
python -c "from src.pandas.pandas_baseline import run_pandas_benchmark; run_pandas_benchmark('data/2019-Nov-first-week.csv', 0.1)"
```

#### Run Spark Pipeline

```bash
# Local mode (single file)
python src/spark/spark_local_undersamp.py

# Or step by step (modular)
python src/spark/preprocessing.py --input data/2019-Nov-first-week.csv --output data/features
python src/spark/train_intent.py --input data/features
```

#### Compare Frameworks

```bash
python benchmarks/compare_frameworks.py --sample 0.05
```

### Real-Time Streaming Pipeline

#### Option 1: Quick Demo (No Docker Required)

```bash
# Terminal 1: Start dashboard
streamlit run dashboard/app.py

# Terminal 2: Generate sample streaming data
python -c "
from src.streaming.metrics_store import MetricsStore
from src.streaming.online_model import OnlinePredictor
import random, time

store = MetricsStore()
model = OnlinePredictor()

for i in range(20):
    store.update_metrics({
        'batch_id': i,
        'total_views': random.randint(100, 500) * (i+1),
        'total_carts': random.randint(10, 50) * (i+1),
        'total_purchases': random.randint(5, 20) * (i+1),
        'conversion_rate': random.uniform(15, 25),
        'active_sessions': random.randint(50, 200),
        'predictions': [],
        'model_metrics': model.get_metrics()
    })
    print(f'Batch {i} sent')
    time.sleep(2)
"
```

Open http://localhost:8501 to see live updates.

#### Option 2: Full Pipeline (With Docker)

```bash
# Start Kafka infrastructure
docker-compose -f docker/docker-compose.yml up -d

# Wait for Kafka to be ready
sleep 15

# Terminal 1: Start Kafka producer (uses auto-detected data file)
python -m src.streaming.kafka_producer --limit 100000

# Terminal 2: Start Spark Streaming processor
python -m src.streaming.stream_processor

# Terminal 3: Start dashboard
streamlit run dashboard/app.py
```

#### Option 3: All-in-One Script

```bash
# Start everything
./scripts/start_streaming.sh --demo

# Stop everything
./scripts/stop_streaming.sh
```

### Cloud Deployment (GCP Dataproc)

```bash
# Create cluster
./scripts/create_cluster.sh

# Run full pipeline
./scripts/run_full_pipeline.sh
```

---

## 🛠️ Installation

### Core Dependencies

```bash
pip install pandas numpy scikit-learn pyspark matplotlib seaborn
```

### Streaming Dependencies (Optional)

```bash
pip install kafka-python streamlit plotly watchdog
```

### Full Installation

```bash
pip install -r requirements.txt
```

---

## 📚 References

1. M. Zaharia et al., "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing," USENIX NSDI, 2012.
2. L. Breiman, "Random Forests," Machine Learning, 45(1), 2001.
3. H. He and E. A. Garcia, "Learning from Imbalanced Data," IEEE TKDE, vol. 21, no. 9, 2009.
4. Y. Hu, Y. Koren, C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets," ICDM, 2008.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

---

**Istanbul Technical University - Department of AI and Data Engineering**
