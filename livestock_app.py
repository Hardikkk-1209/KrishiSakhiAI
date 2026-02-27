"""
🐄 Livestock Biosecurity Monitoring Dashboard
================================================
Premium Streamlit dashboard for real-time livestock health monitoring.
Features: ML-powered diagnostics, gait analysis, anomaly detection,
biosecurity alerts, and herd health analytics.

Usage:
    streamlit run livestock_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livestock_biosecurity.data_generator import LivestockDataGenerator
from livestock_biosecurity.models import load_models, HEALTH_LABELS
from livestock_biosecurity.cv_module import GaitAnalyzer, BehaviorAnalyzer
from livestock_biosecurity.alert_system import BiosecurityAlertSystem

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="KrishiSakhi — Livestock Biosecurity",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.85);
    --bg-card-hover: rgba(17, 24, 39, 0.95);
    --border-subtle: rgba(16, 185, 129, 0.15);
    --border-glow: rgba(16, 185, 129, 0.4);
    --green-primary: #10b981;
    --green-secondary: #059669;
    --green-dark: #047857;
    --green-glow: rgba(16, 185, 129, 0.3);
    --blue-accent: #3b82f6;
    --red-alert: #ef4444;
    --orange-warn: #f97316;
    --yellow-caution: #eab308;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 40%, #0f172a 100%);
    color: var(--text-primary);
}

/* Animated mesh background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 15% 85%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
        radial-gradient(circle at 85% 15%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.hero-header::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; right: -100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--green-primary), transparent);
    animation: hero-sweep 4s ease-in-out infinite;
}
@keyframes hero-sweep {
    0%,100% { transform: translateX(-50%); }
    50% { transform: translateX(50%); }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #a7f3d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.3rem;
    backdrop-filter: blur(20px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: var(--border-glow);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px var(--green-glow);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--green-primary), var(--blue-accent));
    border-radius: 16px 16px 0 0;
}
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--green-primary);
    line-height: 1;
    margin: 0.3rem 0;
}
.metric-value.critical { color: var(--red-alert); }
.metric-value.warning { color: var(--orange-warn); }
.metric-value.healthy { color: var(--green-primary); }
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Alert Cards ── */
.alert-card {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
}
.alert-card:hover {
    transform: translateX(4px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.alert-critical { border-left-color: var(--red-alert); background: rgba(239, 68, 68, 0.08); }
.alert-high { border-left-color: var(--orange-warn); background: rgba(249, 115, 22, 0.08); }
.alert-medium { border-left-color: var(--yellow-caution); background: rgba(234, 179, 8, 0.08); }
.alert-low { border-left-color: var(--green-primary); background: rgba(16, 185, 129, 0.08); }

.alert-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.alert-severity {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.sev-critical { background: rgba(239,68,68,0.2); color: #fca5a5; }
.sev-high { background: rgba(249,115,22,0.2); color: #fdba74; }
.sev-medium { background: rgba(234,179,8,0.2); color: #fde047; }
.sev-low { background: rgba(16,185,129,0.2); color: #6ee7b7; }
.alert-message { font-size: 0.88rem; color: var(--text-primary); line-height: 1.5; }
.alert-meta { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.3rem; }

/* ── Section Headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Animal Card ── */
.animal-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.2rem;
    backdrop-filter: blur(20px);
    transition: all 0.3s ease;
}
.animal-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 30px var(--green-glow);
}
.animal-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
}
.animal-breed {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* ── Status Badge ── */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.status-healthy { background: rgba(16,185,129,0.2); color: #6ee7b7; }
.status-at-risk { background: rgba(249,115,22,0.2); color: #fdba74; }
.status-critical { background: rgba(239,68,68,0.2); color: #fca5a5; }

/* ── Progress Bars ── */
.progress-container {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    overflow: hidden;
    height: 8px;
    margin: 0.5rem 0;
}
.progress-bar {
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease;
}
.progress-green { background: linear-gradient(90deg, var(--green-primary), #34d399); }
.progress-orange { background: linear-gradient(90deg, var(--orange-warn), #fb923c); }
.progress-red { background: linear-gradient(90deg, var(--red-alert), #f87171); }

/* ── Gait Visualization ── */
.gait-meter {
    display: flex;
    gap: 4px;
    margin: 0.5rem 0;
}
.gait-segment {
    flex: 1;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 600;
    color: white;
    transition: all 0.3s ease;
}

/* ── Streamlit Overrides ── */
.stSelectbox > div > div, .stSlider > div > div > div > div {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    color: var(--text-primary);
}
.stButton > button {
    background: linear-gradient(135deg, var(--green-primary), var(--green-secondary));
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px var(--green-glow);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    border-right: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--text-secondary);
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--green-primary), var(--green-secondary)) !important;
    color: white !important;
}
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1rem;
    backdrop-filter: blur(20px);
}
</style>
""", unsafe_allow_html=True)


# ── Initialize Session State ────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    """Load trained ML models (cached)."""
    models = load_models('trained_models')
    return models

@st.cache_resource
def get_data_generator():
    return LivestockDataGenerator(seed=42)

def init_session_state():
    """Initialize session state variables."""
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = BiosecurityAlertSystem()
    if 'gait_analyzer' not in st.session_state:
        st.session_state.gait_analyzer = GaitAnalyzer()
    if 'behavior_analyzer' not in st.session_state:
        st.session_state.behavior_analyzer = BehaviorAnalyzer()
    if 'scan_history' not in st.session_state:
        st.session_state.scan_history = []
    if 'herd_data' not in st.session_state:
        st.session_state.herd_data = None


init_session_state()


# ── Helper Functions ────────────────────────────────────────────────────────

def render_metric_card(icon, value, label, value_class="healthy"):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value {value_class}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_alert_card(alert):
    """Render a styled alert card."""
    severity = alert['severity'].lower()
    icon = alert['severity_info']['icon']
    st.markdown(f"""
    <div class="alert-card alert-{severity}">
        <div class="alert-header">
            <span style="font-weight:700; color:var(--text-primary);">
                {icon} {alert['category_label']}
            </span>
            <span class="alert-severity sev-{severity}">{alert['severity']}</span>
        </div>
        <div class="alert-message">{alert['message']}</div>
        <div class="alert-meta">
            🐄 {alert['animal_id']} &nbsp;|&nbsp;
            🤖 {alert.get('model_source', 'System')} &nbsp;|&nbsp;
            🕐 {alert['timestamp'][:19]}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status):
    """Render a status badge."""
    if status == 0:
        return '<span class="status-badge status-healthy">✅ Healthy</span>'
    elif status == 1:
        return '<span class="status-badge status-at-risk">⚠️ At-Risk</span>'
    else:
        return '<span class="status-badge status-critical">🚨 Critical</span>'


def render_gait_meter(score):
    """Render a visual gait score meter."""
    colors = ['#10b981', '#22c55e', '#eab308', '#f97316', '#ef4444']
    labels = ['1', '2', '3', '4', '5']
    segments = ""
    for i in range(5):
        opacity = '1.0' if (i + 1) <= round(score) else '0.15'
        segments += f'<div class="gait-segment" style="background:{colors[i]};opacity:{opacity};">{labels[i]}</div>'
    return f'<div class="gait-meter">{segments}</div>'


def generate_herd_data(n_animals=25):
    """Generate herd data for dashboard display."""
    gen = get_data_generator()
    herd = []
    diseases = ['Mastitis', 'Lameness', 'BRD', 'Heat_Stress', 'Metabolic_Disorder']
    
    for i in range(n_animals):
        # 70% healthy, 20% at-risk, 10% critical
        rand = np.random.random()
        if rand < 0.70:
            reading = gen.generate_realtime_reading(f"KS-{i:04d}", health='healthy')
        elif rand < 0.90:
            disease = np.random.choice(diseases)
            reading = gen.generate_realtime_reading(f"KS-{i:04d}", health='diseased', 
                                                     disease=disease, severity='mild')
        else:
            disease = np.random.choice(diseases)
            reading = gen.generate_realtime_reading(f"KS-{i:04d}", health='diseased',
                                                     disease=disease, severity='severe')
        herd.append(reading)
    
    return herd


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:2.5rem;">🐄</div>
        <div style="font-size:1.1rem; font-weight:700; color:#10b981; margin-top:0.3rem;">
            Livestock Biosecurity
        </div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.2rem;">
            KrishiSakhi AI Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model status
    st.markdown("##### 🤖 ML Models")
    models = None
    try:
        models = load_all_models()
        model_names = {
            'health_predictor': '🏥 Health Predictor',
            'anomaly_detector': '🔍 Anomaly Detector',
            'gait_predictor': '🦿 Gait Predictor',
            'disease_forecaster': '🦠 Disease Forecaster',
        }
        for key, name in model_names.items():
            if key in models:
                st.markdown(f"✅ {name}")
            else:
                st.markdown(f"❌ {name}")
    except Exception as e:
        st.error(f"Models not loaded. Run `python train_livestock_models.py` first.")
        st.stop()
    
    if not models or len(models) < 4:
        st.error("⚠️ Not all models are available. Please train models first.")
        st.stop()

    st.divider()

    # Herd settings
    st.markdown("##### ⚙️ Simulation Settings")
    herd_size = st.slider("Herd Size", 10, 50, 25, step=5)
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)

    if st.button("🔄 Refresh Herd Data", use_container_width=True):
        st.session_state.herd_data = None
        st.session_state.alert_system.clear_all_alerts()
        st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem; color:#64748b; text-align:center; padding:0.5rem;">
        Built with ❤️ for Indian Farmers<br/>
        KrishiSakhi AI v1.0
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ────────────────────────────────────────────────────────────

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🐄 Livestock Biosecurity Monitor</div>
    <div class="hero-subtitle">
        AI-Powered Health Surveillance • Computer Vision Gait Analysis • IoT Sensor Analytics • Real-Time Biosecurity Alerts
    </div>
</div>
""", unsafe_allow_html=True)


# ── Generate / Load Herd Data ──────────────────────────────────────────────
if st.session_state.herd_data is None:
    with st.spinner("🔄 Scanning herd sensors..."):
        st.session_state.herd_data = generate_herd_data(herd_size)
        # Run all models on herd data
        alert_sys = st.session_state.alert_system
        alert_sys.clear_all_alerts()
        
        for animal_data in st.session_state.herd_data:
            aid = animal_data['animal_id']
            
            # Health prediction
            health_result = models['health_predictor'].predict(animal_data)
            animal_data['health_prediction'] = health_result
            alert_sys.process_health_prediction(aid, health_result)
            
            # Anomaly detection
            anomaly_result = models['anomaly_detector'].predict(animal_data)
            animal_data['anomaly_result'] = anomaly_result
            alert_sys.process_anomaly_detection(aid, anomaly_result)
            
            # Gait prediction
            gait_result = models['gait_predictor'].predict(animal_data)
            animal_data['gait_prediction'] = gait_result
            alert_sys.process_gait_analysis(aid, gait_result)
            
            # Disease forecast
            disease_result = models['disease_forecaster'].predict(animal_data)
            animal_data['disease_prediction'] = disease_result
            alert_sys.process_disease_forecast(aid, disease_result)
            
            # CV Gait analysis
            gait_cv = st.session_state.gait_analyzer.analyze_gait(animal_data)
            animal_data['gait_cv'] = gait_cv
            
            # Behavior analysis
            behavior = st.session_state.behavior_analyzer.analyze_behavior(animal_data)
            animal_data['behavior_analysis'] = behavior
            alert_sys.process_behavior_analysis(aid, behavior)

herd = st.session_state.herd_data
alert_system = st.session_state.alert_system
alert_summary = alert_system.get_alert_summary()


# ── Dashboard Metrics ──────────────────────────────────────────────────────
healthy_count = sum(1 for a in herd if a['health_prediction']['status'] == 0)
at_risk_count = sum(1 for a in herd if a['health_prediction']['status'] == 1)
critical_count = sum(1 for a in herd if a['health_prediction']['status'] == 2)
anomaly_count = sum(1 for a in herd if a['anomaly_result']['is_anomaly'])
avg_gait = np.mean([a['gait_prediction']['gait_score'] for a in herd])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card("🐄", str(len(herd)), "Total Animals")
with col2:
    render_metric_card("💚", str(healthy_count), "Healthy", "healthy")
with col3:
    render_metric_card("⚠️", str(at_risk_count), "At-Risk", "warning")
with col4:
    render_metric_card("🚨", str(critical_count), "Critical", "critical")
with col5:
    render_metric_card("🔍", str(anomaly_count), "Anomalies Detected", "warning" if anomaly_count > 0 else "healthy")


# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Herd Overview",
    "🚨 Biosecurity Alerts",
    "🐄 Individual Analysis",
    "🦿 Gait & Vision Analysis",
    "📊 Model Performance",
])


# ── Tab 1: Herd Overview ──────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📋 Herd Health Overview</div>', unsafe_allow_html=True)

    # Health distribution chart
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Health Status Distribution**")
        health_df = pd.DataFrame({
            'Status': ['Healthy', 'At-Risk', 'Critical'],
            'Count': [healthy_count, at_risk_count, critical_count],
        })
        st.bar_chart(health_df.set_index('Status'), color=['#10b981'])

    with col_chart2:
        st.markdown("**Disease Risk Distribution**")
        disease_types = [a['disease_prediction']['predicted_disease'] for a in herd]
        disease_counts = pd.Series(disease_types).value_counts()
        disease_df = pd.DataFrame({'Disease': disease_counts.index, 'Count': disease_counts.values})
        st.bar_chart(disease_df.set_index('Disease'), color=['#f97316'])

    # Herd data table
    st.markdown('<div class="section-header">📊 Sensor Readings Table</div>', unsafe_allow_html=True)
    
    table_data = []
    for a in herd:
        table_data.append({
            'Animal ID': a['animal_id'],
            'Breed': a['breed'],
            'Status': a['health_prediction']['status_label'],
            'Confidence': f"{a['health_prediction']['confidence']*100:.0f}%",
            'Temp (°C)': a['body_temp'],
            'Heart Rate': a['heart_rate'],
            'Resp. Rate': a['respiratory_rate'],
            'Activity': a['activity_level'],
            'Gait Score': a['gait_prediction']['gait_score'],
            'Disease Risk': a['disease_prediction']['predicted_disease'],
            'Anomaly': '⚠️' if a['anomaly_result']['is_anomaly'] else '✅',
        })
    
    df_display = pd.DataFrame(table_data)
    
    # Color-coded display
    def color_status(val):
        if val == 'Healthy':
            return 'background-color: rgba(16, 185, 129, 0.2); color: #6ee7b7'
        elif val == 'At-Risk':
            return 'background-color: rgba(249, 115, 22, 0.2); color: #fdba74'
        elif val == 'Critical':
            return 'background-color: rgba(239, 68, 68, 0.2); color: #fca5a5'
        return ''
    
    styled_df = df_display.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, height=400)


# ── Tab 2: Biosecurity Alerts ──────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🚨 Biosecurity Alert Center</div>', unsafe_allow_html=True)

    # Alert summary metrics
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        render_metric_card("🚨", str(alert_summary['by_severity']['CRITICAL']), "Critical Alerts", "critical")
    with col_a2:
        render_metric_card("⚠️", str(alert_summary['by_severity']['HIGH']), "High Alerts", "warning")
    with col_a3:
        render_metric_card("🔔", str(alert_summary['by_severity']['MEDIUM']), "Medium Alerts")
    with col_a4:
        render_metric_card("🐄", str(alert_summary['unique_animals_affected']), "Animals Affected", "warning")

    st.markdown("")

    # Filter alerts
    severity_filter = st.selectbox("Filter by Severity", ['All', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
    
    if severity_filter == 'All':
        filtered_alerts = alert_system.get_active_alerts(limit=30)
    else:
        filtered_alerts = alert_system.get_active_alerts(severity=severity_filter, limit=30)

    if filtered_alerts:
        for alert in filtered_alerts:
            render_alert_card(alert)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:var(--text-muted);">
            <div style="font-size:3rem;">✅</div>
            <div style="font-size:1.1rem; margin-top:0.5rem;">No active alerts. All systems normal.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 3: Individual Analysis ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🐄 Individual Animal Diagnostics</div>', unsafe_allow_html=True)

    animal_ids = [a['animal_id'] for a in herd]
    selected_animal_id = st.selectbox("Select Animal", animal_ids)
    
    selected_animal = next((a for a in herd if a['animal_id'] == selected_animal_id), None)
    
    if selected_animal:
        # Animal profile
        col_p1, col_p2 = st.columns([1, 2])
        
        with col_p1:
            st.markdown(f"""
            <div class="animal-card">
                <div style="text-align:center; font-size:3rem; margin-bottom:0.5rem;">🐄</div>
                <div class="animal-name" style="text-align:center;">{selected_animal['animal_id']}</div>
                <div class="animal-breed" style="text-align:center;">{selected_animal['breed']}</div>
                <div style="text-align:center; margin-top:0.8rem;">
                    {render_status_badge(selected_animal['health_prediction']['status'])}
                </div>
                <hr style="border-color: var(--border-subtle); margin: 1rem 0;">
                <div style="font-size:0.85rem; color:var(--text-secondary);">
                    <div>📏 Age: {selected_animal['age_years']} yrs</div>
                    <div>⚖️ Weight: {selected_animal['weight_kg']} kg</div>
                    <div>🍼 Lactation: #{selected_animal['lactation_number']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_p2:
            # Health prediction details
            hp = selected_animal['health_prediction']
            
            st.markdown("**🏥 Health Risk Assessment**")
            prob_df = pd.DataFrame([hp['probabilities']])
            st.bar_chart(prob_df.T, color=['#10b981'])
            
            st.markdown(f"**Confidence:** {hp['confidence']*100:.1f}% — **Status:** {hp['status_label']}")

        # Sensor readings
        st.markdown('<div class="section-header">📡 IoT Sensor Readings</div>', unsafe_allow_html=True)

        sensor_cols = st.columns(5)
        sensors = [
            ("🌡️", "Body Temp", f"{selected_animal['body_temp']}°C", "38.3–39.4°C"),
            ("❤️", "Heart Rate", f"{selected_animal['heart_rate']} bpm", "40–80 bpm"),
            ("🫁", "Resp. Rate", f"{selected_animal['respiratory_rate']}/min", "15–30/min"),
            ("🏃", "Activity", f"{selected_animal['activity_level']}/100", "50–85"),
            ("🥛", "Milk Yield", f"{selected_animal['milk_yield']} L", "12–30 L"),
        ]
        for i, (icon, label, value, normal) in enumerate(sensors):
            with sensor_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value" style="font-size:1.4rem;">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.3rem;">Normal: {normal}</div>
                </div>
                """, unsafe_allow_html=True)

        sensor_cols2 = st.columns(5)
        sensors2 = [
            ("🔄", "Rumination", f"{selected_animal['rumination_min']} min", "380–520 min"),
            ("🌾", "Feed Intake", f"{selected_animal['feed_intake']} kg", "18–28 kg"),
            ("💧", "Water Intake", f"{selected_animal['water_intake']} L", "40–80 L"),
            ("🛌", "Lying Time", f"{selected_animal['lying_time']} hrs", "10–14 hrs"),
            ("👣", "Steps", f"{int(selected_animal['steps_count'])}", "2000–5000"),
        ]
        for i, (icon, label, value, normal) in enumerate(sensors2):
            with sensor_cols2[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value" style="font-size:1.4rem;">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.3rem;">Normal: {normal}</div>
                </div>
                """, unsafe_allow_html=True)

        # Disease prediction
        st.markdown('<div class="section-header">🦠 Disease Risk Forecast</div>', unsafe_allow_html=True)
        dp = selected_animal['disease_prediction']
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            disease_label = dp['predicted_disease'].replace('_', ' ')
            is_healthy = dp['is_healthy']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{'✅' if is_healthy else '⚠️'}</div>
                <div class="metric-value" style="font-size:1.4rem; color:{'#10b981' if is_healthy else '#f97316'};">
                    {disease_label}
                </div>
                <div class="metric-label">Predicted Condition</div>
                <div style="font-size:0.85rem; color:var(--text-secondary); margin-top:0.5rem;">
                    Confidence: {dp['confidence']*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d2:
            st.markdown("**Disease Probability Distribution**")
            prob_data = dp['disease_probabilities']
            prob_df = pd.DataFrame([prob_data])
            st.bar_chart(prob_df.T, color=['#ef4444'])

        # Anomaly detection
        st.markdown('<div class="section-header">🔍 Anomaly Detection</div>', unsafe_allow_html=True)
        ar = selected_animal['anomaly_result']
        
        anom_status = "ANOMALY DETECTED" if ar['is_anomaly'] else "NORMAL"
        anom_color = "#ef4444" if ar['is_anomaly'] else "#10b981"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display:flex; align-items:center; gap:1rem;">
                <div style="font-size:2rem;">{'🚨' if ar['is_anomaly'] else '✅'}</div>
                <div>
                    <div style="font-size:1.3rem; font-weight:700; color:{anom_color};">{anom_status}</div>
                    <div style="font-size:0.85rem; color:var(--text-secondary);">
                        Anomaly Score: {ar['anomaly_score']:.4f} | Severity: {ar['severity']}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 4: Gait & Vision Analysis ─────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🦿 Computer Vision — Gait & Behavior Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="animal-card" style="margin-bottom:1rem;">
        <div style="font-size:0.9rem; color:var(--text-secondary); line-height:1.7;">
            📹 <strong>Computer Vision Pipeline:</strong> Video feeds from barn cameras are processed through pose estimation models
            to extract gait metrics, detect lameness, and identify behavioral changes. The system uses deep learning-based
            keypoint detection (simulated here) to compute locomotion scores, stride analysis, and posture classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Select animal for gait analysis
    selected_gait_animal = st.selectbox("Select Animal for CV Analysis", animal_ids, key="gait_select")
    gait_animal = next((a for a in herd if a['animal_id'] == selected_gait_animal), None)

    if gait_animal:
        gait_cv = gait_animal['gait_cv']
        behavior = gait_animal['behavior_analysis']

        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("**🦿 Locomotion Score**")
            score = gait_cv['locomotion_score']
            st.markdown(f"""
            <div class="animal-card">
                <div style="text-align:center;">
                    <div style="font-size:3.5rem; font-weight:900; color:{'#10b981' if score <= 2 else '#f97316' if score <= 3 else '#ef4444'};">
                        {score}/5
                    </div>
                    <div style="font-size:1rem; font-weight:600; color:var(--text-primary); margin:0.3rem 0;">
                        {gait_cv['label']}
                    </div>
                    <div style="font-size:0.8rem; color:var(--text-secondary);">
                        {gait_cv['description']}
                    </div>
                    {render_gait_meter(score)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # CV Metrics
            st.markdown("**📐 CV-Derived Metrics**")
            cv_m = gait_cv['cv_metrics']
            for metric, value in cv_m.items():
                label = metric.replace('_', ' ').title()
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:0.4rem 0; 
                            border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.85rem;">
                    <span style="color:var(--text-secondary);">{label}</span>
                    <span style="color:var(--text-primary); font-weight:600;">{value}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_g2:
            st.markdown("**🧠 Behavior Pattern Analysis**")
            bh = behavior
            
            pattern_colors = {
                'Normal': '#10b981',
                'Respiratory_Distress': '#ef4444',
                'Lethargy': '#f97316',
                'Isolation': '#eab308',
            }
            pattern_color = pattern_colors.get(bh['behavior_pattern'], '#64748b')
            
            st.markdown(f"""
            <div class="animal-card">
                <div style="text-align:center;">
                    <div style="font-size:2.5rem; font-weight:800; color:{pattern_color};">
                        {bh['behavior_health_score']}/100
                    </div>
                    <div style="font-size:1rem; font-weight:600; color:var(--text-primary); margin:0.3rem 0;">
                        {bh['behavior_pattern'].replace('_', ' ')}
                    </div>
                    <div style="font-size:0.8rem; color:var(--text-secondary);">
                        Behavioral Health Score
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Behavior metrics
            st.markdown("**📊 Behavioral Metrics**")
            bh_metrics = bh['metrics']
            for metric, value in bh_metrics.items():
                label = metric.replace('_', ' ').title()
                bar_width = min(100, max(0, value))
                bar_color = '#10b981' if value > 60 else '#f97316' if value > 30 else '#ef4444'
                st.markdown(f"""
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.3rem;">
                        <span style="color:var(--text-secondary);">{label}</span>
                        <span style="color:var(--text-primary); font-weight:600;">{value:.1f}</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width:{bar_width}%; background:{bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Behavior alerts
            if bh.get('alerts'):
                st.markdown("**⚠️ Behavior Alerts**")
                for ba in bh['alerts']:
                    st.warning(f"**{ba['type']}** ({ba['severity']}): {ba['message']}")

        # Gait component scores
        st.markdown('<div class="section-header">📐 Gait Component Breakdown</div>', unsafe_allow_html=True)
        comp = gait_cv['component_scores']
        comp_cols = st.columns(5)
        comp_icons = ['🏃', '👣', '🛌', '📏', '⚖️']
        for i, (metric, value) in enumerate(comp.items()):
            with comp_cols[i]:
                label = metric.replace('_', ' ').title()
                color = '#10b981' if value > 70 else '#f97316' if value > 40 else '#ef4444'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{comp_icons[i]}</div>
                    <div class="metric-value" style="font-size:1.5rem; color:{color};">{value}%</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Tab 5: Model Performance ──────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">📊 ML Model Performance Metrics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="animal-card" style="margin-bottom:1rem;">
        <div style="font-size:0.9rem; color:var(--text-secondary); line-height:1.7;">
            🤖 <strong>Model Architecture:</strong> The system uses an ensemble of specialized ML models — 
            Random Forest for health classification, Isolation Forest for anomaly detection, 
            Gradient Boosting for gait prediction and disease forecasting — all trained on synthetic IoT sensor data
            that simulates real-world livestock physiological parameters.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display model metrics
    model_info = [
        {
            'name': '🏥 Health Risk Classifier',
            'algorithm': 'Random Forest',
            'details': 'Classifies animals as Healthy, At-Risk, or Critical based on 10 IoT sensor features + 6 engineered features.',
            'model_key': 'health_predictor',
            'metrics_attrs': ['accuracy', 'f1_weighted'],
        },
        {
            'name': '🔍 Anomaly Detector',
            'algorithm': 'Isolation Forest',
            'details': 'Trained on healthy-only data to identify anomalous patterns. Uses 13 sensor + gait features.',
            'model_key': 'anomaly_detector',
            'metrics_attrs': ['precision', 'recall', 'f1'],
        },
        {
            'name': '🦿 Gait Score Predictor',
            'algorithm': 'Gradient Boosting Regressor',
            'details': 'Predicts locomotion score (1-5) from activity, steps, lying time, stride length, and symmetry.',
            'model_key': 'gait_predictor',
            'metrics_attrs': ['mae', 'r2'],
        },
        {
            'name': '🦠 Disease Forecaster',
            'algorithm': 'Gradient Boosting Classifier',
            'details': 'Multi-class prediction: Mastitis, Lameness, BRD, Heat Stress, Metabolic Disorder, or None.',
            'model_key': 'disease_forecaster',
            'metrics_attrs': ['accuracy', 'f1_weighted'],
        },
    ]

    for info in model_info:
        model = models.get(info['model_key'])
        if model and hasattr(model, 'metrics'):
            metrics = model.metrics
            
            metrics_html = ""
            for attr in info['metrics_attrs']:
                val = metrics.get(attr, 'N/A')
                if isinstance(val, float):
                    val_str = f"{val:.4f}"
                    pct = val * 100
                    color = '#10b981' if pct > 80 else '#f97316' if pct > 60 else '#ef4444'
                else:
                    val_str = str(val)
                    color = '#64748b'
                
                label = attr.replace('_', ' ').upper()
                metrics_html += f"""
                <div style="display:flex; justify-content:space-between; align-items:center; 
                            padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                    <span style="color:var(--text-secondary); font-size:0.85rem;">{label}</span>
                    <span style="color:{color}; font-weight:700; font-size:1.1rem;">{val_str}</span>
                </div>
                """

            st.markdown(f"""
            <div class="animal-card" style="margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
                    <div>
                        <div style="font-size:1.15rem; font-weight:700; color:var(--text-primary);">{info['name']}</div>
                        <div style="font-size:0.8rem; color:var(--green-primary); font-weight:500;">{info['algorithm']}</div>
                    </div>
                    <span class="status-badge status-healthy">TRAINED</span>
                </div>
                <div style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:0.8rem;">{info['details']}</div>
                {metrics_html}
            </div>
            """, unsafe_allow_html=True)

    # Feature Importance for Health Predictor
    st.markdown('<div class="section-header">🔑 Feature Importance — Health Predictor</div>', unsafe_allow_html=True)
    
    hp_model = models.get('health_predictor')
    if hp_model and hasattr(hp_model, 'metrics') and 'feature_importance' in hp_model.metrics:
        imp = hp_model.metrics['feature_importance']
        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:12]
        
        imp_df = pd.DataFrame(imp_sorted, columns=['Feature', 'Importance'])
        imp_df = imp_df.set_index('Feature')
        st.bar_chart(imp_df, color=['#10b981'])


# ── Auto-refresh ────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.session_state.herd_data = None
    st.rerun()


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:var(--text-muted); font-size:0.8rem;">
    <div style="margin-bottom:0.5rem;">
        🐄 Livestock Biosecurity Monitor — KrishiSakhi AI Platform v1.0
    </div>
    <div>
        Computer Vision • IoT Sensor Analytics • ML-Powered Diagnostics • Real-Time Biosecurity
    </div>
</div>
""", unsafe_allow_html=True)
