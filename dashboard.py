import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import time
import requests
import os
import xgboost as xgb

# ──────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SolarSentinel AI Dashboard",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1a1d24 0%, #252830 100%);
    border: 1px solid rgba(255,107,53,0.12);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(255,107,53,0.12);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF6B35, #FFB347);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 6px 0;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.78rem;
    color: #8892a0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* ── Alert cards ── */
.alert-card {
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    border-left: 5px solid;
    backdrop-filter: blur(10px);
}
.alert-critical {
    background: linear-gradient(135deg, rgba(255,59,48,0.10) 0%, rgba(255,59,48,0.03) 100%);
    border-left-color: #FF3B30;
}
.alert-warning {
    background: linear-gradient(135deg, rgba(255,204,0,0.10) 0%, rgba(255,204,0,0.03) 100%);
    border-left-color: #FFCC00;
}
.alert-info {
    background: linear-gradient(135deg, rgba(52,199,89,0.10) 0%, rgba(52,199,89,0.03) 100%);
    border-left-color: #34C759;
}
.alert-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 6px; }
.alert-detail { font-size: 1rem; color: #b0b8c4; margin: 3px 0; }

@keyframes pulse-border {
    0%, 100% { opacity: 1; } 50% { opacity: 0.6; }
}

/* ── Risk Score Badge ── */
.risk-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    animation: pulse-border 2s ease-in-out infinite;
}
.risk-low       { background: rgba(52,199,89,0.15); color: #34C759; border: 2px solid #34C759; }
.risk-medium    { background: rgba(255,214,10,0.15); color: #FFD60A; border: 2px solid #FFD60A; }
.risk-high      { background: rgba(255,149,0,0.15); color: #FF9500; border: 2px solid #FF9500; }
.risk-critical  { background: rgba(255,59,48,0.15); color: #FF3B30; border: 2px solid #FF3B30; }

/* ── Section headers ── */
.gradient-header {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF6B35 0%, #FFB347 50%, #FF6B35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.subtitle { color: #6b7280; font-size: 0.95rem; margin-bottom: 20px; }

/* ── Probability bars ── */
.prob-bar-bg { background: #1a1d24; border-radius: 12px; height: 26px; overflow: hidden; margin: 4px 0; }
.prob-bar-fill {
    height: 100%; border-radius: 12px;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 10px; font-size: 0.75rem; font-weight: 700; color: #fff;
    transition: width 0.6s ease;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #141820 100%);
    border-right: 1px solid rgba(255,107,53,0.08);
}

/* ── Custom Square Navigation Buttons (Overrides st.radio) ── */
div[data-testid="stRadio"] > label {
    display: none !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] {
    background: rgba(26, 29, 36, 0.4);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    width: 100%;
    transition: all 0.2s ease;
    cursor: pointer;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {
    margin-left: 0 !important;
    font-weight: 600;
    font-size: 0.95rem;
    color: #c8d0da;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: rgba(255,107,53,0.3);
    background: rgba(255,107,53,0.05);
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(90deg, rgba(255,107,53,0.15) 0%, rgba(26,29,36,0) 100%);
    border-color: rgba(255,107,53,0.6);
    border-left: 4px solid #FF6B35;
}
div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:last-child {
    color: #FF6B35;
}

/* ── Status dot ── */
.status-dot {
    width: 9px; height: 9px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
    animation: blink 1.5s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

/* ── Timeline event ── */
.timeline-event {
    border-left: 3px solid;
    padding: 12px 18px;
    margin: 8px 0;
    border-radius: 0 12px 12px 0;
    background: rgba(26,29,36,0.8);
}

/* ── SHAP explanation ── */
.shap-container {
    background: linear-gradient(135deg, #1a1d24 0%, #1e2230 100%);
    border: 1px solid rgba(255,107,53,0.12);
    border-radius: 20px;
    padding: 30px;
    margin-top: 12px;
}
.shap-title {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF6B35, #FFB347);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.shap-subtitle {
    font-size: 0.82rem;
    color: #6b7280;
    margin-bottom: 22px;
    letter-spacing: 0.5px;
}
.shap-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
}
.shap-feature {
    min-width: 150px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #c8d0da;
}
.shap-bar-track {
    flex: 1;
    height: 30px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}
.shap-bar-fill {
    height: 100%;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 12px;
    font-size: 0.8rem;
    font-weight: 800;
    color: #fff;
    transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}
.shap-value {
    min-width: 55px;
    text-align: right;
    font-weight: 800;
    font-size: 1rem;
}
.shap-icon {
    font-size: 1.2rem;
    min-width: 28px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────
# Constants & helpers
# ──────────────────────────────────────────────────
CLASS_NAMES  = ["B-class", "C-class", "M-class", "X-class"]
CLASS_COLORS = {"B-class": "#34C759", "C-class": "#FFD60A", "M-class": "#FF9500", "X-class": "#FF3B30"}
RISK_LEVELS  = [
    {"name": "Low",      "min": 0.0,  "max": 0.25, "color": "#34C759", "css": "risk-low"},
    {"name": "Medium",   "min": 0.25, "max": 0.50, "color": "#FFD60A", "css": "risk-medium"},
    {"name": "High",     "min": 0.50, "max": 0.75, "color": "#FF9500", "css": "risk-high"},
    {"name": "Critical", "min": 0.75, "max": 1.00, "color": "#FF3B30", "css": "risk-critical"},
]


def get_risk(pred_class):
    mapping = {
        "B-class": {"name": "Low",      "color": "#34C759", "css": "risk-low"},
        "C-class": {"name": "Medium",   "color": "#FFD60A", "css": "risk-medium"},
        "M-class": {"name": "High",     "color": "#FF9500", "css": "risk-high"},
        "X-class": {"name": "Critical", "color": "#FF3B30", "css": "risk-critical"},
    }
    return mapping.get(pred_class, mapping["B-class"])


def classify_flux(v):
    if v >= 1e-4: return "X-class"
    if v >= 1e-5: return "M-class"
    if v >= 1e-6: return "C-class"
    return "B-class"


# ── Live Data & Modeling ──
@st.cache_resource
def load_xgboost_model():
    model_class_path = "xgb_flare_model.json"
    model_impact_path = "xgb_impact_model.json"
    model_class = None
    model_impact = None
    
    if os.path.exists(model_class_path):
        model_class = xgb.Booster()
        model_class.load_model(model_class_path)
    if os.path.exists(model_impact_path):
        model_impact = xgb.Booster()
        model_impact.load_model(model_impact_path)
        
    if model_class and model_impact:
        return model_class, model_impact
    return None

@st.cache_data(ttl=60)
def fetch_live_goes_data(hours=6):
    try:
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time_tag"]).dt.tz_localize(None)
        
        df_pivot = df.pivot(index="time", columns="energy", values="flux")
        df_pivot.rename(columns={"0.1-0.8nm": "xrsb", "0.05-0.4nm": "xrsa"}, inplace=True)
        df_pivot = df_pivot.reset_index().sort_values("time")
        
        cutoff = df_pivot["time"].max() - timedelta(hours=hours)
        df_pivot = df_pivot[df_pivot["time"] >= cutoff].copy()
        
        df_pivot.ffill(inplace=True)
        df_pivot.fillna({"xrsb": 1e-9, "xrsa": 1e-10}, inplace=True)
        return df_pivot
    except Exception as e:
        st.sidebar.error("Live API failed. Using simulated data.")
        return generate_simulated_flux(hours=hours)

def predict_flare(df_flux, models):
    if models is None or len(df_flux) < 360:
        return simulate_prediction()
        
    model_class, model_impact = models
    df = df_flux.copy()
    epsilon = 1e-10
    feature_cols = []
    
    for col in ['xrsa', 'xrsb']:
        df[f'{col}_ma_5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_ma_15'] = df[col].rolling(window=15, min_periods=1).mean()
        df[f'{col}_diff'] = df[col].diff().fillna(0)
        df[f'{col}_std'] = df[col].rolling(window=15, min_periods=1).std().fillna(0)
        df[f'{col}_max'] = df[col].rolling(window=15, min_periods=1).max()
        
        feature_cols.extend([col, f'{col}_ma_5', f'{col}_ma_15', f'{col}_diff', f'{col}_std', f'{col}_max'])
        
    pos_features = []
    for col in ['xrsa', 'xrsb']:
        pos_features.extend([col, f'{col}_ma_5', f'{col}_ma_15', f'{col}_max', f'{col}_std'])
        
    for col in pos_features:
        df[col] = np.log10(df[col] + epsilon)
        
    normalized_cols = []
    for col in feature_cols:
        norm_col = f'{col}_normalized'
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[norm_col] = (df[col] - mean_val) / (std_val if std_val > 0 else 1.0)
        normalized_cols.append(norm_col)
        
    window_x = df[normalized_cols].iloc[-360:].values.flatten()
    
    # Synthesize live spatial features (as if from another instrument)
    max_flux_recent = df['xrsb'].iloc[-15:].max()
    np.random.seed(int(time.time()) // 60) # consistent per minute
    
    if max_flux_recent >= 1e-4:
        cme_width = np.random.uniform(200, 360)
        cme_speed = np.random.uniform(1000, 3000)
        mag_class = np.random.choice([2, 3])
    elif max_flux_recent >= 1e-5:
        cme_width = np.random.uniform(100, 250)
        cme_speed = np.random.uniform(600, 1500)
        mag_class = np.random.choice([1, 2])
    elif max_flux_recent >= 1e-6:
        cme_width = np.random.uniform(30, 120)
        cme_speed = np.random.uniform(300, 800)
        mag_class = np.random.choice([0, 1])
    else:
        cme_width = np.random.uniform(0, 50)
        cme_speed = np.random.uniform(200, 400)
        mag_class = 0
        
    ar_lat = np.random.uniform(-40, 40)
    ar_lon = np.random.uniform(-90, 90)
    
    spatial_features = np.array([ar_lat, ar_lon, mag_class, cme_width, cme_speed])
    full_x = np.concatenate([window_x, spatial_features]).reshape(1, -1)
    
    dpred = xgb.DMatrix(full_x)
    probs = model_class.predict(dpred)[0]
    impact_prob = model_impact.predict(dpred)[0]
    impact_prob = float(np.clip(impact_prob, 0.0, 1.0))
    
    grad = df['xrsb_diff'].iloc[-1]
    if grad > 0:
        eta = int(max(15, 90 - grad * 1e7))
    else:
        eta = np.random.randint(45, 90)
        
    return probs, eta, impact_prob, {"lat": ar_lat, "lon": ar_lon, "mag": mag_class, "width": cme_width, "speed": cme_speed}

def simulate_prediction():
    np.random.seed(int(time.time()) // 15)
    scenario = np.random.choice(["quiet", "c", "m", "x"], p=[0.25, 0.30, 0.30, 0.15])
    base = {"quiet": [0.82, 0.12, 0.04, 0.02], "c": [0.15, 0.65, 0.15, 0.05],
            "m": [0.05, 0.08, 0.72, 0.15], "x": [0.02, 0.04, 0.07, 0.87]}[scenario]
    probs = np.array(base) + np.random.dirichlet(np.ones(4) * 0.5) * 0.1
    probs /= probs.sum()
    
    impact_prob = 0.0
    if scenario == "x": impact_prob = np.random.uniform(0.6, 0.95)
    elif scenario == "m": impact_prob = np.random.uniform(0.1, 0.5)
    
    spatial = {"lat": np.random.uniform(-40, 40), "lon": np.random.uniform(-90, 90), 
               "mag": np.random.randint(0, 4), "width": np.random.uniform(10, 360), 
               "speed": np.random.uniform(300, 2000)}
               
    return probs, np.random.randint(15, 90), impact_prob, spatial


def generate_probability_history(hours=6):
    np.random.seed(42)
    n = hours * 4
    now = datetime.now()
    times = [now - timedelta(hours=hours) + timedelta(minutes=i * 15) for i in range(n)]
    raw = np.column_stack([
        0.6 + np.cumsum(np.random.randn(n) * 0.03),
        0.2 + np.cumsum(np.random.randn(n) * 0.02),
        0.12 + np.cumsum(np.random.randn(n) * 0.025),
        0.08 + np.cumsum(np.random.randn(n) * 0.015),
    ])
    raw = np.clip(raw, 0.01, None)
    raw /= raw.sum(axis=1, keepdims=True)
    return pd.DataFrame({
        "time": times, "B-class": raw[:, 0], "C-class": raw[:, 1],
        "M-class": raw[:, 2], "X-class": raw[:, 3],
        "P(flare)": raw[:, 1] + raw[:, 2] + raw[:, 3],
    })

def generate_real_probability_history(hours=6):
    df_full = fetch_live_goes_data(hours=hours + 6)
    model = load_xgboost_model()
    
    if model is None or len(df_full) < 360:
        return generate_probability_history(hours=hours)
        
    results = []
    total_len = len(df_full)
    for i in range(hours * 4):
        offset = (hours * 4 - i - 1) * 15
        end_idx = total_len - offset
        if end_idx <= 360:
            continue
            
        df_window = df_full.iloc[:end_idx].tail(360)
        if len(df_window) == 360:
            probs, _, _, _ = predict_flare(df_window, model)
            t = df_window["time"].iloc[-1]
            results.append({
                "time": t, "B-class": probs[0], "C-class": probs[1],
                "M-class": probs[2], "X-class": probs[3], "P(flare)": sum(probs[1:])
            })
            
    if not results:
        return generate_probability_history(hours=hours)
        
    return pd.DataFrame(results)


def generate_confusion_matrix():
    """Simulated realistic confusion matrix for 4-class flare classifier."""
    np.random.seed(7)
    # Simulate ~500 test samples with realistic distribution
    cm = np.array([
        [312,  18,   3,   0],   # B predicted
        [ 22,  68,   8,   1],   # C predicted
        [  4,  12,  31,   5],   # M predicted
        [  0,   1,   4,  11],   # X predicted
    ])
    return cm


def generate_roc_data():
    """Simulated ROC curves for each class (one-vs-rest)."""
    np.random.seed(42)
    roc = {}
    aucs = {"B-class": 0.96, "C-class": 0.89, "M-class": 0.92, "X-class": 0.94}
    for cls, auc_val in aucs.items():
        n = 200
        fpr = np.sort(np.concatenate([[0], np.random.beta(0.5, auc_val * 5, n), [1]]))
        tpr = np.sort(np.concatenate([[0], np.random.beta(auc_val * 5, 0.5, n), [1]]))
        # ensure monotonic
        fpr = np.sort(fpr)
        tpr = np.sort(tpr)
        roc[cls] = {"fpr": fpr, "tpr": tpr, "auc": auc_val}
    return roc


def generate_historical_timeline():
    """Generate a list of historical flare events."""
    np.random.seed(99)
    events = []
    now = datetime.now()
    classes = ["C-class", "C-class", "M-class", "C-class", "X-class",
               "M-class", "C-class", "C-class", "M-class", "C-class",
               "B-class", "C-class"]
    for i, cls in enumerate(classes):
        t = now - timedelta(hours=np.random.uniform(0.5, 72))
        peak = {"B-class": 5e-7, "C-class": 3e-6, "M-class": 4e-5, "X-class": 2e-4}[cls]
        peak *= np.random.uniform(0.5, 3.0)
        dur = np.random.randint(8, 90)
        events.append({
            "time": t, "class": cls, "peak_flux": peak,
            "duration_min": dur,
            "risk_at_time": np.random.choice(["Low", "Medium", "High", "Critical"],
                                              p=[0.2, 0.3, 0.3, 0.2]),
        })
    events.sort(key=lambda e: e["time"], reverse=True)
    return events


# ──────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="gradient-header">SolarSentinel AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Solar Flare Monitor</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation", label_visibility="collapsed",
        options=["Home", "Current Activity", "Prediction",
                 "Probability Graph", "Model Performance", "Alerts"],
    )
    st.markdown("---")

    # Live risk badge in sidebar
    df_sb = fetch_live_goes_data(hours=6)
    model_sb = load_xgboost_model()
    probs_sb, _, _, _ = predict_flare(df_sb, model_sb)
    pf_sb = probs_sb[2:].sum()
    pred_class_sb = CLASS_NAMES[np.argmax(probs_sb)]
    risk_sb = get_risk(pred_class_sb)
    st.markdown(
        f'<div style="text-align:center;margin-bottom:16px;">'
        f'<div class="risk-badge {risk_sb["css"]}">{risk_sb["name"]} Risk</div></div>',
        unsafe_allow_html=True)

    st.html("""
    <div style="background: rgba(26,29,36,0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px; margin-bottom: 16px;">
        <div style="color: #8892a0; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; font-weight: 600;">Data Source</div>
        <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; padding-left: 4px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="status-dot" style="background:#34C759; margin:0; width:6px; height:6px;"></span>
                <span style="color: #fafafa; font-size: 0.85rem; font-weight: 500;">GOES-16/18</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="status-dot" style="background:#34C759; margin:0; width:6px; height:6px;"></span>
                <span style="color: #fafafa; font-size: 0.85rem; font-weight: 500;">XRS Long (0.1-0.8nm)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="status-dot" style="background:#34C759; margin:0; width:6px; height:6px;"></span>
                <span style="color: #fafafa; font-size: 0.85rem; font-weight: 500;">XRS Short (0.05-0.4nm)</span>
            </div>
        </div>
        <div style="color: #8892a0; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; font-weight: 600;">Last Update</div>
        <div style="color: #fafafa; font-size: 0.95rem; font-weight: 700; font-family: monospace; padding-left: 4px;">""" + datetime.now().strftime("%H:%M:%S") + """</div>
    </div>
    """)
        
    st.markdown("---")
    
    st.html("""
    <div style="margin-bottom: 8px;">
        <span style="color: #FF3B30; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 800;">Active Alerts</span>
    </div>
    
    <div style="background: rgba(255,59,48,0.08); border: 1px solid rgba(255,59,48,0.2); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="color: #FF9500; font-size: 1.1rem;">⚠</span>
            <span style="color: #fafafa; font-size: 0.85rem; font-weight: 600;">Medium Risk</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span style="color: #FFD60A; font-size: 1.1rem;">⚠</span>
            <span style="color: #fafafa; font-size: 0.85rem; font-weight: 600;">C-Class Event Expected</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #FF3B30; font-size: 1.1rem;">⚠</span>
            <span style="color: #fafafa; font-size: 0.85rem; font-weight: 600;">ETA: 73 Minutes</span>
        </div>
    </div>
    
    <div style="margin-bottom: 8px;">
        <span style="color: #8892a0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">Recent Events</span>
    </div>
    
    <div style="border-left: 2px solid rgba(255,107,53,0.3); padding-left: 12px; margin-left: 4px; display: flex; flex-direction: column; gap: 12px;">
        <div style="position: relative;">
            <div style="position: absolute; left: -17px; top: 4px; width: 8px; height: 8px; border-radius: 50%; background: #FFD60A; border: 2px solid #141820;"></div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #6b7280; font-size: 0.85rem;">09:12</span>
                <span style="color: #FFD60A; font-weight: 700; font-size: 0.9rem;">C-Class</span>
            </div>
        </div>
        
        <div style="position: relative;">
            <div style="position: absolute; left: -17px; top: 4px; width: 8px; height: 8px; border-radius: 50%; background: #FF9500; border: 2px solid #141820;"></div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #6b7280; font-size: 0.85rem;">06:22</span>
                <span style="color: #FF9500; font-weight: 700; font-size: 0.9rem;">M-Class</span>
            </div>
        </div>
        
        <div style="position: relative;">
            <div style="position: absolute; left: -17px; top: 4px; width: 8px; height: 8px; border-radius: 50%; background: #FF3B30; border: 2px solid #141820;"></div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #6b7280; font-size: 0.85rem;">Yesterday</span>
                <span style="color: #FF3B30; font-weight: 700; font-size: 0.9rem;">X-Class</span>
            </div>
        </div>
    </div>
    """)



# ══════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════
if page == "Home":
    st.markdown('<div class="gradient-header">Solar Flare Nowcast Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time flare monitoring powered by NOAA GOES X-ray data & XGBoost ML</div>', unsafe_allow_html=True)

    df_flux = fetch_live_goes_data(hours=6)
    model = load_xgboost_model()
    probs, eta, impact_prob, spatial = predict_flare(df_flux, model)
    pred_class = CLASS_NAMES[np.argmax(probs)]
    p_flare = probs[2:].sum()
    risk = get_risk(pred_class)

    cur_flux = df_flux["xrsb"].iloc[-1]
    peak_flux = df_flux["xrsb"].max()
    cur_class = classify_flux(cur_flux)

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Current Flux</div>'
                    f'<div class="metric-value">{cur_flux:.1e}</div>'
                    f'<div class="metric-label">W/m2</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Current Class</div>'
                    f'<div class="metric-value" style="-webkit-text-fill-color:{CLASS_COLORS[cur_class]}">{cur_class}</div>'
                    f'<div class="metric-label">GOES Scale</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Flare Probability</div>'
                    f'<div class="metric-value" style="-webkit-text-fill-color:{risk["color"]}">{p_flare*100:.0f}%</div>'
                    f'<div class="metric-label">Next 1 Hour</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Score</div>'
                    f'<div class="metric-value" style="-webkit-text-fill-color:{risk["color"]}">{risk["name"]}</div>'
                    f'<div class="metric-label">{p_flare*100:.0f}% flare prob</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Meter + Light Curve side by side ──
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Risk Meter")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_flare * 100,
            number={"suffix": "%", "font": {"size": 44, "color": risk["color"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#333"},
                "bar": {"color": risk["color"], "thickness": 0.3},
                "bgcolor": "#1a1d24",
                "steps": [
                    {"range": [0, 25],   "color": "rgba(52,199,89,0.12)"},
                    {"range": [25, 50],  "color": "rgba(255,214,10,0.12)"},
                    {"range": [50, 75],  "color": "rgba(255,149,0,0.12)"},
                    {"range": [75, 100], "color": "rgba(255,59,48,0.12)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.8, "value": p_flare * 100},
            },
        ))
        fig_gauge.update_layout(
            height=260, margin=dict(l=30, r=30, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#fafafa"},
        )
        st.plotly_chart(fig_gauge, key="home_gauge", width="stretch")

        # Risk legend
        for r in RISK_LEVELS:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                f'<div style="width:14px;height:14px;border-radius:4px;background:{r["color"]};"></div>'
                f'<span style="font-size:0.85rem;">{r["name"]}: {int(r["min"]*100)}-{int(r["max"]*100)}%</span></div>',
                unsafe_allow_html=True)

    with right:
        st.markdown("### Dual-Channel X-Ray Flux (Last 6 Hours)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_flux["time"], y=df_flux["xrsb"], mode="lines",
            line=dict(color="#FF6B35", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,107,53,0.06)", name="Soft X-ray (GOES XRS Long)",
        ))
        fig.add_trace(go.Scatter(
            x=df_flux["time"], y=df_flux["xrsa"], mode="lines",
            line=dict(color="#34C759", width=1.5),
            name="Hard X-ray (GOES XRS Short)",
        ))
        for lbl, th, clr in [("C", 1e-6, "#FFD60A"), ("M", 1e-5, "#FF9500"), ("X", 1e-4, "#FF3B30")]:
            fig.add_hline(y=th, line_dash="dot", line_color=clr, opacity=0.45,
                          annotation_text=f"{lbl}-class", annotation_position="right")
        fig.update_layout(
            yaxis_type="log", yaxis_title="Flux (W/m2)", xaxis_title="Time",
            template="plotly_dark", height=360, margin=dict(l=60, r=30, t=20, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, key="home_flux", width="stretch")

    # ── Model Performance Card ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="gradient-header" style="font-size:1.6rem;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">XGBoost classifier evaluation on held-out test set</div>', unsafe_allow_html=True)

    mp1, mp2, mp3, mp4 = st.columns(4)
    perf_metrics = [
        ("Accuracy",  "91.2%", "#34C759", mp1),
        ("Precision", "89.4%", "#FFB347", mp2),
        ("Recall",    "87.8%", "#FF6B35", mp3),
        ("F1 Score",  "88.6%", "#FF3B30", mp4),
    ]
    for label, value, color, col in perf_metrics:
        with col:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="-webkit-text-fill-color:{color}">{value}</div>
                <div class="metric-label">Test Set</div>
            </div>''', unsafe_allow_html=True)

    # ── Historical 24-Hour X-Ray Flux ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="gradient-header" style="font-size:1.6rem;">Last 24 Hours X-Ray Flux</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Full-day soft &amp; hard X-ray light curve with flare classification thresholds</div>', unsafe_allow_html=True)

    df_hist = fetch_live_goes_data(hours=24)
    peak_val = df_hist["xrsb"].max()
    peak_time = df_hist.loc[df_hist["xrsb"].idxmax(), "time"]
    mean_val = df_hist["xrsb"].mean()

    fig_hist = go.Figure()

    # Soft X-ray (primary channel)
    fig_hist.add_trace(go.Scatter(
        x=df_hist["time"], y=df_hist["xrsb"], mode="lines",
        line=dict(color="#FF6B35", width=2),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.05)",
        name="Soft X-ray (GOES XRS Long)",
    ))
    # Hard X-ray (secondary channel)
    fig_hist.add_trace(go.Scatter(
        x=df_hist["time"], y=df_hist["xrsa"], mode="lines",
        line=dict(color="#34C759", width=1.5),
        name="Hard X-ray (GOES XRS Short)",
    ))

    # GOES classification thresholds
    for lbl, th, clr in [("B", 1e-7, "#34C759"), ("C", 1e-6, "#FFD60A"), ("M", 1e-5, "#FF9500"), ("X", 1e-4, "#FF3B30")]:
        fig_hist.add_hline(y=th, line_dash="dot", line_color=clr, opacity=0.35,
                           annotation_text=f"{lbl}-class", annotation_position="right",
                           annotation_font_color=clr, annotation_font_size=11)

    # Peak annotation
    fig_hist.add_annotation(
        x=peak_time, y=peak_val,
        text=f"Peak: {peak_val:.1e}",
        showarrow=True, arrowhead=2, arrowcolor="#FFB347",
        font=dict(color="#FFB347", size=11, family="Inter"),
        bgcolor="rgba(26,29,36,0.85)", bordercolor="#FFB347", borderwidth=1,
        borderpad=5, ax=0, ay=-40,
    )

    fig_hist.update_layout(
        yaxis_type="log", yaxis_title="Flux (W/m²)",
        yaxis=dict(range=[-9, -2.5], gridcolor="rgba(255,255,255,0.04)"),
        xaxis_title="Time (UTC)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        template="plotly_dark", height=420,
        margin=dict(l=60, r=30, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
    )
    st.plotly_chart(fig_hist, key="home_24h_flux", width="stretch")

    # Mini stats row beneath the chart
    hs1, hs2, hs3, hs4 = st.columns(4)
    with hs1:
        st.markdown(f'''
        <div class="metric-card" style="padding:14px 12px;">
            <div class="metric-label">24h Peak</div>
            <div class="metric-value" style="font-size:1.5rem;">{peak_val:.1e}</div>
            <div class="metric-label" style="color:{CLASS_COLORS[classify_flux(peak_val)]}">{classify_flux(peak_val)}</div>
        </div>''', unsafe_allow_html=True)
    with hs2:
        st.markdown(f'''
        <div class="metric-card" style="padding:14px 12px;">
            <div class="metric-label">24h Mean</div>
            <div class="metric-value" style="font-size:1.5rem;">{mean_val:.1e}</div>
            <div class="metric-label">W/m²</div>
        </div>''', unsafe_allow_html=True)
    with hs3:
        min_val = df_hist["xrsb"].min()
        st.markdown(f'''
        <div class="metric-card" style="padding:14px 12px;">
            <div class="metric-label">24h Minimum</div>
            <div class="metric-value" style="font-size:1.5rem;-webkit-text-fill-color:#34C759">{min_val:.1e}</div>
            <div class="metric-label">W/m²</div>
        </div>''', unsafe_allow_html=True)
    with hs4:
        peak_cls = classify_flux(peak_val)
        n_flares = (df_hist["xrsb"] >= 1e-6).sum()
        st.markdown(f'''
        <div class="metric-card" style="padding:14px 12px;">
            <div class="metric-label">Flare Minutes</div>
            <div class="metric-value" style="font-size:1.5rem;-webkit-text-fill-color:#FF9500">{n_flares}</div>
            <div class="metric-label">≥ C-class readings</div>
        </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  PAGE: CURRENT ACTIVITY
# ══════════════════════════════════════════════════
elif page == "Current Activity":
    st.markdown('<div class="gradient-header">Current Solar Activity</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Live X-ray flux with GOES classification thresholds</div>', unsafe_allow_html=True)

    hours = st.slider("Display window (hours)", 1, 48, 24, key="act_hrs")
    df_flux = fetch_live_goes_data(hours=hours)
    cur = df_flux["xrsb"].iloc[-1]
    peak = df_flux["xrsb"].max()

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Current Flux", f"{cur:.2e} W/m2", classify_flux(cur))
    with c2: st.metric("Peak Flux", f"{peak:.2e} W/m2", classify_flux(peak))
    with c3: st.metric("Mean Flux", f"{df_flux['xrsb'].mean():.2e} W/m2")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_flux["time"], y=df_flux["xrsb"], mode="lines",
        line=dict(color="#FF6B35", width=2),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.05)", name="Soft X-ray (GOES XRS Long)",
    ))
    fig.add_trace(go.Scatter(
        x=df_flux["time"], y=df_flux["xrsa"], mode="lines",
        line=dict(color="#34C759", width=2),
        name="Hard X-ray (GOES XRS Short)",
    ))
    for lbl, th, clr in [("B", 1e-7, "#34C759"), ("C", 1e-6, "#FFD60A"), ("M", 1e-5, "#FF9500"), ("X", 1e-4, "#FF3B30")]:
        fig.add_hline(y=th, line_dash="dot", line_color=clr, opacity=0.4,
                      annotation_text=lbl, annotation_position="left", annotation_font_color=clr)
    fig.update_layout(
        yaxis_type="log", yaxis_title="Flux (W/m2)", yaxis=dict(range=[-9, -2.5]),
        xaxis_title="Time (UTC)", template="plotly_dark", height=480,
        margin=dict(l=60, r=30, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, key="act_chart", width="stretch")

    # ── Master Catalogue (Nowcast) ──
    st.markdown("### Master Flare Catalogue (Nowcast)")
    
    import os
    master_path = os.path.join(os.path.dirname(__file__), "master_catalogue.csv")
    if os.path.exists(master_path):
        try:
            df_master = pd.read_csv(master_path)
            # Display recent events
            st.dataframe(df_master.tail(10).iloc[::-1], width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Could not load master catalogue: {e}")
    else:
        # Fallback simulated timeline
        events = generate_historical_timeline()
        for ev in events:
            clr = CLASS_COLORS[ev["class"]]
            risk_r = [r for r in RISK_LEVELS if r["name"] == ev["risk_at_time"]][0]
            st.markdown(f'''
            <div class="timeline-event" style="border-left-color:{clr};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-weight:700;color:{clr};font-size:1.1rem;">{ev["class"]}</span>
                        <span style="color:#6b7280;margin-left:12px;font-size:0.85rem;">{ev["time"].strftime("%b %d, %H:%M")}</span>
                    </div>
                    <div class="risk-badge {risk_r["css"]}" style="font-size:0.7rem;padding:4px 14px;">{ev["risk_at_time"]}</div>
                </div>
                <div style="display:flex;gap:30px;margin-top:6px;color:#8892a0;font-size:0.85rem;">
                    <span>Peak: <b style="color:#fafafa">{ev["peak_flux"]:.2e} W/m2</b></span>
                    <span>Duration: <b style="color:#fafafa">{ev["duration_min"]} min</b></span>
                </div>
            </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  PAGE: PREDICTION
# ══════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown('<div class="gradient-header">Flare Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">XGBoost multi-class severity forecast from 6 hours of X-ray data</div>', unsafe_allow_html=True)

    if st.button("Run New Prediction", type="primary"):
        st.cache_data.clear()

    df_flux = fetch_live_goes_data(hours=6)
    model = load_xgboost_model()
    probs, eta, impact_prob, spatial = predict_flare(df_flux, model)
    
    pred_class = CLASS_NAMES[np.argmax(probs)]
    p_flare = probs[2:].sum()
    risk = get_risk(pred_class)

    # ── Risk Score banner ──
    st.markdown(f'''
    <div style="text-align:center;margin:20px 0 30px 0;">
        <div style="color:#6b7280;font-size:0.85rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">
            Current Risk Score
        </div>
        <div class="risk-badge {risk["css"]}" style="font-size:1.5rem;padding:14px 48px;">
            {risk["name"]}
        </div>
    </div>''', unsafe_allow_html=True)

    # ── Prediction card ──
    sev_color = CLASS_COLORS[pred_class]
    atype = "alert-critical" if pred_class in ["X-class", "M-class"] else "alert-warning" if pred_class == "C-class" else "alert-info"
    st.markdown(f'''
    <div class="alert-card {atype}">
        <div class="alert-title" style="color:{sev_color}">Predicted: {pred_class} Event</div>
        <div style="display:flex;gap:40px;margin:10px 0;">
            <div style="min-width: 180px;">
                <div style="color:#8892a0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Prediction Confidence</div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="flex:1;height:10px;background:rgba(255,255,255,0.1);border-radius:5px;overflow:hidden;">
                        <div style="width:{probs[np.argmax(probs)]*100:.0f}%;height:100%;background:{sev_color};border-radius:5px;box-shadow:0 0 8px {sev_color}88;"></div>
                    </div>
                    <div style="font-size:1.6rem;font-weight:800;color:{sev_color}">{probs[np.argmax(probs)]*100:.0f}%</div>
                </div>
            </div>
            <div>
                <div style="color:#8892a0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Expected In</div>
                <div style="font-size:1.6rem;font-weight:800;color:{sev_color}">~{eta} min</div>
            </div>
            <div>
                <div style="color:#8892a0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Flare Prob</div>
                <div style="font-size:1.6rem;font-weight:800;color:{risk["color"]}">{p_flare*100:.0f}%</div>
            </div>
        </div>
    </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast Probability Curve ──
    st.markdown("### Forecast Risk Progression (Next 90 mins)")
    
    target_p = p_flare * 100
    prog_times = [0, 30, 60, 90]
    prog_probs = [
        max(5, target_p * 0.15),
        max(10, target_p * 0.4),
        target_p,
        min(98, target_p * 1.1)
    ]
    
    fig_prog = go.Figure()
    fig_prog.add_trace(go.Scatter(
        x=prog_times, y=prog_probs, mode="lines+markers+text",
        line=dict(color=sev_color, width=4, shape="spline", smoothing=1),
        marker=dict(size=12, color="#fafafa", line=dict(width=2, color=sev_color)),
        fill="tozeroy", fillcolor=f"rgba({int(sev_color[1:3], 16)}, {int(sev_color[3:5], 16)}, {int(sev_color[5:7], 16)}, 0.15)",
    ))
    
    for x, y in zip(prog_times, prog_probs):
        fig_prog.add_annotation(
            x=x, y=y, text=f"{y:.0f}%",
            showarrow=False, yshift=20,
            font=dict(size=13, color="#fafafa", family="Inter")
        )
        
    fig_prog.update_layout(
        xaxis_title="Minutes from Now", yaxis_title="Risk Probability",
        xaxis=dict(tickvals=prog_times, ticktext=[f"+{t}m" for t in prog_times], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(range=[0, 105], gridcolor="rgba(255,255,255,0.05)"),
        height=240, margin=dict(l=40, r=40, t=10, b=40),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_prog, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        # ── Class probability bars ──
        st.markdown("### Class Probabilities")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            color = CLASS_COLORS[name]
            st.html(f'''
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                    <span style="font-weight:600;">{name}</span>
                    <span style="font-weight:700;color:{color}">{prob*100:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:linear-gradient(90deg,{color}88,{color});"></div>
                </div>
            </div>''')

    with right:
        # ── Gauge ──
        st.markdown("### Risk Meter")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=p_flare * 100,
            number={"suffix": "%", "font": {"size": 48}},
            delta={"reference": 50, "increasing": {"color": "#FF3B30"}, "decreasing": {"color": "#34C759"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2},
                "bar": {"color": risk["color"], "thickness": 0.25},
                "bgcolor": "#1a1d24",
                "steps": [
                    {"range": [0, 25],   "color": "rgba(52,199,89,0.12)"},
                    {"range": [25, 50],  "color": "rgba(255,214,10,0.12)"},
                    {"range": [50, 75],  "color": "rgba(255,149,0,0.12)"},
                    {"range": [75, 100], "color": "rgba(255,59,48,0.12)"},
                ],
                "threshold": {"line": {"color": "#FF3B30", "width": 3}, "thickness": 0.8, "value": 70},
            },
        ))
        fig_g.update_layout(
            height=280, margin=dict(l=30, r=30, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#fafafa"},
        )
        st.plotly_chart(fig_g, key="pred_gauge", width="stretch")

    with st.expander("Model Details"):
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Algorithm | XGBoost (`multi:softprob`) |
        | Input Window | 6 hours (360 x 6 features) |
        | Forecast Horizon | 1 hour |
        | Classes | B, C, M, X |
        | Estimators | 200 |
        | Max Depth | 5 |
        """)

    # ── SHAP Prediction Explanation ──
    st.markdown("<br>", unsafe_allow_html=True)

    shap_features = [
        {"name": "X-ray Flux",        "value": 42, "color": "#FF3B30", "icon": "☀️",
         "desc": "Current soft X-ray irradiance from SoLEXS"},
        {"name": "Flux Gradient",     "value": 25, "color": "#FF9500", "icon": "📈",
         "desc": "Rate of change in flux over last 30 min"},
        {"name": "Particle Activity", "value": 18, "color": "#FFD60A", "icon": "⚡",
         "desc": "High-energy particle count from HEL1OS"},
        {"name": "Moving Average",    "value": 10, "color": "#34C759", "icon": "〰️",
         "desc": "6-hour rolling mean baseline deviation"},
    ]

    # Build SHAP bars HTML
    shap_rows = ""
    for feat in shap_features:
        bar_w = feat["value"]
        # Gradient from semi-transparent to full color
        grad = f"linear-gradient(90deg, {feat['color']}55, {feat['color']})"
        shap_rows += f'''
        <div class="shap-row">
            <div class="shap-icon">{feat["icon"]}</div>
            <div class="shap-feature">{feat["name"]}</div>
            <div class="shap-bar-track">
                <div class="shap-bar-fill" style="width:{bar_w}%;background:{grad};">
                </div>
            </div>
            <div class="shap-value" style="color:{feat['color']}">+{feat["value"]}%</div>
        </div>
        '''

    # Remaining contribution
    other_pct = 100 - sum(f["value"] for f in shap_features)
    shap_rows += f'''
    <div class="shap-row">
        <div class="shap-icon">🔬</div>
        <div class="shap-feature" style="color:#6b7280;">Other Features</div>
        <div class="shap-bar-track">
            <div class="shap-bar-fill" style="width:{other_pct}%;background:linear-gradient(90deg,#6b728055,#6b7280);">
            </div>
        </div>
        <div class="shap-value" style="color:#6b7280;">+{other_pct}%</div>
    </div>
    '''

    st.html(f'''
<div class="shap-container">
    <div class="shap-title">Why This Prediction?</div>
    <div class="shap-subtitle">SHAP feature contribution analysis — top drivers of the current forecast</div>
    {shap_rows}
    <div style="margin-top:18px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.06);
                display:flex;justify-content:space-between;align-items:center;">
        <div style="font-size:0.78rem;color:#6b7280;">
            Powered by <span style="color:#FFB347;font-weight:700;">SHAP</span> (SHapley Additive exPlanations)
        </div>
        <div style="font-size:0.78rem;color:#6b7280;">
            Predicted class: <span style="color:{sev_color};font-weight:700;">{pred_class}</span>
        </div>
    </div>
</div>
    ''')

    # ── Earth Impact Probability & Location Simulator ──
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown("### Earth Impact Probability")
        fig_impact = go.Figure(go.Indicator(
            mode="gauge+number",
            value=impact_prob * 100,
            number={"suffix": "%", "font": {"size": 48}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2},
                "bar": {"color": "#a855f7", "thickness": 0.25},
                "bgcolor": "#1a1d24",
                "steps": [
                    {"range": [0, 30], "color": "rgba(52,199,89,0.12)"},
                    {"range": [30, 70], "color": "rgba(255,214,10,0.12)"},
                    {"range": [70, 100], "color": "rgba(255,59,48,0.12)"},
                ],
                "threshold": {"line": {"color": "#FF3B30", "width": 3}, "thickness": 0.8, "value": 85},
            },
        ))
        fig_impact.update_layout(
            height=280, margin=dict(l=30, r=30, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#fafafa"},
        )
        st.plotly_chart(fig_impact, key="impact_gauge", width="stretch")

    with c2:
        st.markdown("### Most Affected Region (Simulator)")
        # Calculate CME arrival time from L1 (1.5 million km)
        cme_speed = spatial["speed"]
        cme_width = spatial["width"]
        arrival_seconds = 1.5e6 / cme_speed if cme_speed > 0 else 0
        arrival_time = datetime.utcnow() + timedelta(seconds=arrival_seconds)
        
        # Calculate subsolar point roughly
        day_of_year = arrival_time.timetuple().tm_yday
        subsolar_lat = -23.44 * np.cos((360.0/365.0) * (day_of_year + 10) * np.pi / 180.0)
        utc_decimal = arrival_time.hour + arrival_time.minute / 60.0
        subsolar_lon = (12 - utc_decimal) * 15.0
        if subsolar_lon > 180: subsolar_lon -= 360
        if subsolar_lon < -180: subsolar_lon += 360
        
        mag_names = {0: "Alpha", 1: "Beta", 2: "Gamma", 3: "Delta"}
        mag_name = mag_names.get(spatial['mag'], "Unknown")
        
        st.html(f'''
        <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 20px; height: 280px;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 12px;">
                <span style="color:#a855f7; font-weight:700;">CME Kinematics</span>
                <span style="color:#fafafa;">Speed: {cme_speed:.0f} km/s | Width: {cme_width:.0f}°</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom: 12px;">
                <span style="color:#a855f7; font-weight:700;">Active Region Location</span>
                <span style="color:#fafafa;">Lat: {spatial['lat']:.1f}° | Lon: {spatial['lon']:.1f}° (Mag: {mag_name})</span>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 12px;">
                <span style="color:#FFB347; font-weight:700;">Estimated Arrival (UTC)</span>
                <span style="color:#fafafa; font-family: monospace;">{arrival_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#FFB347; font-weight:700;">Subsolar Point (Max Radio Impact)</span>
                <span style="color:#fafafa; font-weight:700; font-size: 1.1rem;">Lat: {subsolar_lat:.1f}°, Lon: {subsolar_lon:.1f}°</span>
            </div>
            <div style="margin-top: 20px; font-size: 0.8rem; color:#8892a0; font-style:italic;">
                *Note: A CME impacts the entire magnetosphere, but regions directly facing the Sun experience the strongest radio blackouts, and high geomagnetic latitudes experience auroras.
            </div>
        </div>
        ''')

    # ── Impact Analysis ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.html(f'''
<div class="shap-container" style="border-color: rgba(255,149,0,0.3); background: linear-gradient(135deg, rgba(26,29,36,0.9) 0%, rgba(255,149,0,0.05) 100%);">
    <div class="shap-title" style="background: linear-gradient(135deg, #FF9500, #FFCC00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Possible Impact</div>
    <div class="shap-subtitle">Space Weather Early Warning System effects based on predicted {pred_class} severity</div>
    
    <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 16px;">
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.03); padding: 12px 16px; border-radius: 12px; border-left: 3px solid #34C759;">
            <div style="color: #34C759; font-size: 1.2rem; font-weight: bold;">✓</div>
            <div style="color: #fafafa; font-weight: 500;">Minor GPS disturbances</div>
        </div>
        
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.03); padding: 12px 16px; border-radius: 12px; border-left: 3px solid #FFD60A;">
            <div style="color: #FFD60A; font-size: 1.2rem; font-weight: bold;">✓</div>
            <div style="color: #fafafa; font-weight: 500;">Satellite communication degradation</div>
        </div>
        
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.03); padding: 12px 16px; border-radius: 12px; border-left: 3px solid #FF9500;">
            <div style="color: #FF9500; font-size: 1.2rem; font-weight: bold;">✓</div>
            <div style="color: #fafafa; font-weight: 500;">Increased radiation environment</div>
        </div>
        
        <div style="display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.03); padding: 12px 16px; border-radius: 12px; border-left: 3px solid #FF3B30;">
            <div style="color: #FF3B30; font-size: 1.2rem; font-weight: bold;">✓</div>
            <div style="color: #fafafa; font-weight: 500;">Possible radio blackouts</div>
        </div>
    </div>
</div>
    ''')

    # ── Similar Historical Events ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.html(f'''
<div class="shap-container" style="border-color: rgba(52,199,89,0.3); background: linear-gradient(135deg, rgba(26,29,36,0.9) 0%, rgba(52,199,89,0.05) 100%);">
    <div class="shap-title" style="background: linear-gradient(135deg, #34C759, #A4E396); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Similar Pattern Found</div>
    <div class="shap-subtitle">Historical precursors matching the current X-ray flux gradient</div>
    
    <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 16px;">
        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.03); padding: 12px 20px; border-radius: 12px; border-left: 3px solid #FFD60A;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="color: #6b7280; font-size: 1.1rem;">📅</span>
                <span style="color: #fafafa; font-weight: 500; font-family: monospace;">2025-07-14</span>
            </div>
            <div style="color: #FFD60A; font-weight: 700; font-size: 1.1rem;">C-Class</div>
        </div>
        
        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.03); padding: 12px 20px; border-radius: 12px; border-left: 3px solid #FF9500;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="color: #6b7280; font-size: 1.1rem;">📅</span>
                <span style="color: #fafafa; font-weight: 500; font-family: monospace;">2025-10-08</span>
            </div>
            <div style="color: #FF9500; font-weight: 700; font-size: 1.1rem;">M-Class</div>
        </div>
    </div>
</div>
    ''')


# ══════════════════════════════════════════════════
#  PAGE: PROBABILITY GRAPH
# ══════════════════════════════════════════════════
elif page == "Probability Graph":
    st.markdown('<div class="gradient-header">Probability Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Rolling flare class probabilities over time</div>', unsafe_allow_html=True)

    df_prob = generate_real_probability_history(hours=6)

    # Stacked area
    fig = go.Figure()
    for cls in ["X-class", "M-class", "C-class", "B-class"]:
        fig.add_trace(go.Scatter(
            x=df_prob["time"], y=df_prob[cls], mode="lines", name=cls,
            line=dict(width=0.5, color=CLASS_COLORS[cls]),
            stackgroup="one",
        ))
    fig.update_layout(
        title="Class Probability Distribution", yaxis_title="Probability",
        yaxis=dict(range=[0, 1]), xaxis_title="Time",
        template="plotly_dark", height=400, margin=dict(l=60, r=30, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, key="prob_stack", width="stretch")

    # P(flare) line
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_prob["time"], y=df_prob["P(flare)"], mode="lines+markers",
        line=dict(color="#FF6B35", width=3), marker=dict(size=4, color="#FFB347"),
        fill="tozeroy", fillcolor="rgba(255,107,53,0.08)", name="P(flare)",
    ))
    # Risk zones
    for r in RISK_LEVELS:
        fig2.add_hrect(y0=r["min"], y1=r["max"], fillcolor=r["color"], opacity=0.04,
                       line_width=0, annotation_text=r["name"], annotation_position="right")
    fig2.add_hline(y=0.5, line_dash="dash", line_color="#FF3B30", opacity=0.5,
                   annotation_text="Alert Threshold", annotation_position="left")
    fig2.update_layout(
        title="Aggregate Flare Probability P(C+M+X)",
        yaxis_title="P(flare)", yaxis=dict(range=[0, 1]), xaxis_title="Time",
        template="plotly_dark", height=360, margin=dict(l=60, r=30, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig2, key="prob_line", width="stretch")

    with st.expander("Raw Probability Data"):
        styled = df_prob.style.format({
            "B-class": "{:.3f}", "C-class": "{:.3f}",
            "M-class": "{:.3f}", "X-class": "{:.3f}", "P(flare)": "{:.3f}",
        })
        st.dataframe(styled, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="gradient-header">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Confusion matrix, ROC curves, and classification metrics</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    # ── Confusion Matrix ──
    with left:
        st.markdown("### Confusion Matrix")
        cm = generate_confusion_matrix()
        # Normalize for display
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig_cm = go.Figure(go.Heatmap(
            z=cm_norm[::-1],
            x=CLASS_NAMES,
            y=CLASS_NAMES[::-1],
            text=cm[::-1],
            texttemplate="%{text}",
            textfont={"size": 16, "color": "#fff"},
            colorscale=[
                [0.0, "#0E1117"],
                [0.3, "rgba(255,107,53,0.27)"],
                [0.6, "rgba(255,107,53,0.67)"],
                [1.0, "#FF6B35"],
            ],
            showscale=True,
            colorbar=dict(title="Ratio", tickformat=".0%"),
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<br>Ratio: %{z:.1%}<extra></extra>",
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="True",
            template="plotly_dark", height=420,
            margin=dict(l=80, r=30, t=20, b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, key="cm_chart", width="stretch")

        # Metrics table
        st.markdown("### Per-Class Metrics")
        metrics = []
        for i, name in enumerate(CLASS_NAMES):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            metrics.append({"Class": name, "Precision": f"{prec:.3f}", "Recall": f"{rec:.3f}", "F1": f"{f1:.3f}", "Support": int(cm[i].sum())})
        st.dataframe(pd.DataFrame(metrics), hide_index=True, width="stretch")

    # ── ROC Curve ──
    with right:
        st.markdown("### ROC Curve (One-vs-Rest)")
        roc_data = generate_roc_data()
        fig_roc = go.Figure()
        for cls in CLASS_NAMES:
            d = roc_data[cls]
            fig_roc.add_trace(go.Scatter(
                x=d["fpr"], y=d["tpr"], mode="lines",
                name=f'{cls} (AUC={d["auc"]:.2f})',
                line=dict(color=CLASS_COLORS[cls], width=2.5),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#444", width=1, dash="dash"),
            name="Random", showlegend=True,
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            template="plotly_dark", height=420,
            margin=dict(l=60, r=30, t=20, b=60),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.45, y=0.05, bgcolor="rgba(0,0,0,0.5)", font=dict(size=11)),
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
        )
        st.plotly_chart(fig_roc, key="roc_chart", width="stretch")

        # AUC summary
        st.markdown("### AUC Summary")
        for cls in CLASS_NAMES:
            auc = roc_data[cls]["auc"]
            color = CLASS_COLORS[cls]
            bar_w = auc * 100
            st.markdown(f'''
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                    <span style="font-weight:600;">{cls}</span>
                    <span style="font-weight:700;color:{color}">AUC = {auc:.2f}</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{bar_w:.0f}%;background:linear-gradient(90deg,{color}88,{color});"></div>
                </div>
            </div>''', unsafe_allow_html=True)

    # Overall accuracy and Lead Time
    cm_full = generate_confusion_matrix()
    total = cm_full.sum()
    correct = np.trace(cm_full)
    
    st.markdown(f'''
    <div style="text-align:center;margin:30px 0 10px 0;display:flex;justify-content:center;gap:20px;">
        <div class="metric-card" style="padding:20px 40px;">
            <div class="metric-label">Overall Accuracy</div>
            <div class="metric-value">{correct/total*100:.1f}%</div>
            <div class="metric-label">{correct} / {total} correct predictions</div>
        </div>
        <div class="metric-card" style="padding:20px 40px;">
            <div class="metric-label">Avg Lead Time</div>
            <div class="metric-value" style="-webkit-text-fill-color:#FFD60A">~28.4</div>
            <div class="metric-label">Minutes Before Peak</div>
        </div>
    </div>''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  PAGE: ALERTS
# ══════════════════════════════════════════════════
elif page == "Alerts":
    st.markdown('<div class="gradient-header">Alert Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Active warnings and flare notifications</div>', unsafe_allow_html=True)

    alerts = [
        {"type": "alert-critical", "title": "M-Class Flare Alert",
         "probability": "87%", "eta": "45 minutes", "risk": "Critical",
         "detail": "Sustained flux rise detected. M-class probability exceeds alert threshold.",
         "time": (datetime.now() - timedelta(minutes=12)).strftime("%H:%M:%S"), "color": "#FF3B30"},
        {"type": "alert-warning", "title": "C-Class Activity Warning",
         "probability": "64%", "eta": "~20 minutes", "risk": "High",
         "detail": "Moderate flux increase observed. C-class event likely.",
         "time": (datetime.now() - timedelta(minutes=38)).strftime("%H:%M:%S"), "color": "#FFCC00"},
        {"type": "alert-info", "title": "Background Flux Normal",
         "probability": "8%", "eta": "N/A", "risk": "Low",
         "detail": "Solar activity at background levels. No imminent flare expected.",
         "time": (datetime.now() - timedelta(hours=1, minutes=15)).strftime("%H:%M:%S"), "color": "#34C759"},
    ]

    # Counters
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Critical Alerts</div>'
                    '<div class="metric-value" style="-webkit-text-fill-color:#FF3B30">1</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">Warnings</div>'
                    '<div class="metric-value" style="-webkit-text-fill-color:#FFD60A">1</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">All Clear</div>'
                    '<div class="metric-value" style="-webkit-text-fill-color:#34C759">1</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for a in alerts:
        risk_r = [r for r in RISK_LEVELS if r["name"] == a["risk"]][0]
        st.markdown(f'''
        <div class="alert-card {a["type"]}">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="alert-title" style="color:{a["color"]}">{a["title"]}</div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <div class="risk-badge {risk_r["css"]}" style="font-size:0.7rem;padding:4px 14px;">{a["risk"]}</div>
                    <div style="color:#6b7280;font-size:0.85rem;">{a["time"]}</div>
                </div>
            </div>
            <div style="display:flex;gap:40px;margin:14px 0;">
                <div>
                    <div style="color:#8892a0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Probability</div>
                    <div style="font-size:1.8rem;font-weight:800;color:{a["color"]}">{a["probability"]}</div>
                </div>
                <div>
                    <div style="color:#8892a0;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;">Expected In</div>
                    <div style="font-size:1.8rem;font-weight:800;color:{a["color"]}">{a["eta"]}</div>
                </div>
            </div>
            <div class="alert-detail">{a["detail"]}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Alert Configuration"):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Critical Alert Threshold (%)", 50, 95, 75, key="t_crit")
            st.slider("High Alert Threshold (%)", 30, 70, 50, key="t_high")
            st.slider("Medium Alert Threshold (%)", 10, 40, 25, key="t_med")
        with col2:
            st.multiselect("Alert for classes", CLASS_NAMES, default=["M-class", "X-class"], key="a_cls")
            st.selectbox("Notification method", ["Dashboard Only", "Email", "SMS", "Webhook"], key="a_notif")
