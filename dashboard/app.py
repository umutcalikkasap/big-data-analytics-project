"""
app.py - Streamlit Real-time Dashboard

E-Commerce Purchase Intent Prediction icin canli metrikler,
tahminler ve model performansi gosterir.

Usage:
    streamlit run dashboard/app.py

Authors: Abdulkadir Kulce, Berkay Turk, Umut Calikkasap
Course: YZV411E Big Data Analytics - Istanbul Technical University
"""

import streamlit as st
import pandas as pd
import time
import sys
import os

# Plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.streaming.metrics_store import get_metrics, get_history, get_predictions
from src.streaming.config import DASHBOARD_CONFIG


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="E-Commerce Streaming Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .prediction-high {
        background-color: #90EE90;
    }
    .prediction-medium {
        background-color: #FFFFE0;
    }
    .prediction-low {
        background-color: #FFB6C1;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("⚙️ Dashboard Settings")

refresh_rate = st.sidebar.slider(
    "Refresh Rate (seconds)",
    min_value=1,
    max_value=10,
    value=DASHBOARD_CONFIG["refresh_rate"]
)

show_predictions = st.sidebar.checkbox("Show Live Predictions", True)
show_charts = st.sidebar.checkbox("Show Charts", True)
auto_refresh = st.sidebar.checkbox("Auto Refresh", True)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About**

Real-time E-Commerce analytics dashboard
powered by Kafka + Spark Streaming.

**Authors:**
- Abdulkadir Kulce
- Berkay Turk
- Umut Calikkasap

**Course:** YZV411E Big Data Analytics
Istanbul Technical University
""")


# =============================================================================
# HEADER
# =============================================================================
st.title("🛒 E-Commerce Real-Time Analytics")
st.markdown("**Purchase Intent Prediction - Streaming Dashboard**")
st.markdown("---")


# =============================================================================
# DATA FETCHING
# =============================================================================
@st.cache_data(ttl=2)  # Cache for 2 seconds
def fetch_data():
    """Fetch current metrics and history."""
    return {
        "metrics": get_metrics(),
        "history": get_history(100),
        "predictions": get_predictions(50)
    }


data = fetch_data()
metrics = data["metrics"]
history = data["history"]
predictions = data["predictions"]


# =============================================================================
# MAIN METRICS ROW
# =============================================================================
st.subheader("📊 Live Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👁️ Total Views",
        value=f"{metrics.get('total_views', 0):,}"
    )

with col2:
    st.metric(
        label="🛒 Cart Additions",
        value=f"{metrics.get('total_carts', 0):,}"
    )

with col3:
    st.metric(
        label="💰 Purchases",
        value=f"{metrics.get('total_purchases', 0):,}"
    )

with col4:
    conversion = metrics.get('conversion_rate', 0)
    st.metric(
        label="📈 Conversion Rate",
        value=f"{conversion:.2f}%"
    )


# =============================================================================
# SECONDARY METRICS ROW
# =============================================================================
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📱 Active Sessions",
        value=metrics.get('active_sessions', 0)
    )

with col2:
    st.metric(
        label="📦 Total Events",
        value=f"{metrics.get('total_events', 0):,}"
    )

with col3:
    st.metric(
        label="🔢 Batch ID",
        value=metrics.get('batch_id', 'N/A')
    )

with col4:
    timestamp = metrics.get('timestamp', '')
    if timestamp:
        time_str = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp[:8]
    else:
        time_str = "Waiting..."
    st.metric(
        label="🕐 Last Update",
        value=time_str
    )


# =============================================================================
# CHARTS SECTION
# =============================================================================
if show_charts and PLOTLY_AVAILABLE and history:
    st.markdown("---")
    st.subheader("📈 Real-time Charts")

    chart_col1, chart_col2 = st.columns(2)

    # Conversion Rate Over Time
    with chart_col1:
        if len(history) > 1:
            df_history = pd.DataFrame(history)

            if 'timestamp' in df_history.columns and 'conversion_rate' in df_history.columns:
                df_history['time'] = pd.to_datetime(df_history['timestamp'])

                fig = px.line(
                    df_history.tail(50),  # Son 50 batch
                    x='time',
                    y='conversion_rate',
                    title='Conversion Rate Over Time',
                    labels={'conversion_rate': 'Conversion Rate (%)', 'time': 'Time'}
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for data to show conversion chart...")

    # Event Distribution
    with chart_col2:
        if len(history) >= 1:
            # Son 10 batch'in toplami
            recent = history[-10:] if len(history) >= 10 else history

            total_views = sum(h.get('total_views', 0) for h in recent)
            total_carts = sum(h.get('total_carts', 0) for h in recent)
            total_purchases = sum(h.get('total_purchases', 0) for h in recent)

            fig = go.Figure(data=[
                go.Bar(
                    x=['Views', 'Carts', 'Purchases'],
                    y=[total_views, total_carts, total_purchases],
                    marker_color=['#3498db', '#f39c12', '#27ae60']
                )
            ])

            fig.update_layout(
                title='Event Distribution (Last 10 Batches)',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for data to show event distribution...")


# =============================================================================
# MODEL PERFORMANCE SECTION
# =============================================================================
st.markdown("---")
st.subheader("🤖 Online Model Performance")

model_metrics = metrics.get('model_metrics', {})

model_col1, model_col2, model_col3, model_col4 = st.columns(4)

with model_col1:
    st.metric(
        label="Model Type",
        value=model_metrics.get('model_type', 'N/A')[:20]
    )

with model_col2:
    st.metric(
        label="Predictions Made",
        value=f"{model_metrics.get('predictions_made', 0):,}"
    )

with model_col3:
    accuracy = model_metrics.get('recent_accuracy', 0)
    st.metric(
        label="Recent Accuracy",
        value=f"{accuracy*100:.1f}%"
    )

with model_col4:
    is_fitted = model_metrics.get('is_fitted', False)
    st.metric(
        label="Model Status",
        value="✅ Fitted" if is_fitted else "⏳ Training"
    )


# =============================================================================
# PREDICTIONS TABLE
# =============================================================================
if show_predictions:
    st.markdown("---")
    st.subheader("🔮 Live Purchase Predictions")

    if predictions:
        # DataFrame olustur
        pred_data = []
        for pred in predictions[:20]:  # Son 20
            features = pred.get('features', {})
            prob = pred.get('purchase_probability', 0)

            # Status belirleme
            if pred.get('has_purchased'):
                status = "✅ Purchased"
            elif prob > 0.7:
                status = "🔥 High Intent"
            elif prob > 0.4:
                status = "⏳ Medium Intent"
            else:
                status = "👀 Browsing"

            pred_data.append({
                'Session': str(pred.get('session_id', ''))[:12] + '...',
                'Views': features.get('view_count', 0),
                'Carts': features.get('cart_count', 0),
                'Duration': f"{features.get('session_duration', 0):.0f}s",
                'Avg Price': f"${features.get('avg_price', 0):.2f}",
                'Purchase Prob': f"{prob*100:.1f}%",
                'Status': status
            })

        df_pred = pd.DataFrame(pred_data)

        # Styling function
        def highlight_probability(val):
            if '%' in str(val):
                prob = float(val.replace('%', ''))
                if prob > 70:
                    return 'background-color: #90EE90'  # Light green
                elif prob > 40:
                    return 'background-color: #FFFFE0'  # Light yellow
                else:
                    return 'background-color: #FFB6C1'  # Light red
            return ''

        # Apply styling
        styled_df = df_pred.style.applymap(
            highlight_probability,
            subset=['Purchase Prob']
        )

        st.dataframe(styled_df, use_container_width=True, height=400)

        # Legend
        st.markdown("""
        **Legend:** 🟢 High (>70%) | 🟡 Medium (40-70%) | 🔴 Low (<40%)
        """)

    else:
        st.info("⏳ Waiting for streaming predictions...")
        st.markdown("""
        **To see predictions:**
        1. Start Docker: `docker-compose -f docker/docker-compose.yml up -d`
        2. Start Producer: `python -m src.streaming.kafka_producer --limit 10000`
        3. Start Processor: `python -m src.streaming.stream_processor`
        """)


# =============================================================================
# RAW DATA (Expandable)
# =============================================================================
with st.expander("📋 Raw Metrics Data"):
    st.json(metrics)


# =============================================================================
# AUTO REFRESH
# =============================================================================
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    YZV411E Big Data Analytics | Istanbul Technical University | 2024
</div>
""", unsafe_allow_html=True)
