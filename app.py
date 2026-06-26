# app.py — Week 7 Final: Polished Streamlit App (Member 2 UI)

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import __main__ as _main

def _relu(x):
    return np.maximum(0, x)

class SavedModel:
    def __init__(self, w, b):
        self.w = w; self.b = b
    def predict(self, X):
        return np.array(X) @ self.w + self.b

class NNModel:
    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1=W1; self.b1=b1
        self.W2=W2; self.b2=b2
        self.W3=W3; self.b3=b3
    def predict(self, X):
        X = np.array(X)
        a1 = _relu(X @ self.W1 + self.b1)
        a2 = _relu(a1 @ self.W2 + self.b2)
        return (a2 @ self.W3 + self.b3).flatten()

# Fix: joblib saved these as __main__.SavedModel / NNModel
_main.SavedModel = SavedModel
_main.NNModel    = NNModel

# Manual standardization — the saved scaler was fit on pre-normalized data
# so we use the true CA housing raw feature statistics instead
_FEAT_MEAN = np.array([-119.5697,  35.6319,  28.6394,
                        2635.763,  537.8705, 1425.4767,
                         499.5397,   3.8707,
                           5.4290,   0.2126,    3.0706])
_FEAT_STD  = np.array([  2.0035,   2.1359,  12.5856,
                        2181.6,    421.5,   1132.5,
                         382.3,     1.8994,
                           2.3704,   0.0576,  10.3862])

def manual_scale(X_raw):
    return (np.array(X_raw, dtype=float) - _FEAT_MEAN) / _FEAT_STD

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; padding-bottom: 0.3rem; line-height: 1.2;
    letter-spacing: -0.01em;
}
.sub-header { font-size: 1rem; color: #64748b; text-align: center; margin-bottom: 0.3rem; font-weight: 300; }

/* NEW: make every Streamlit subheader/header bold across all tabs */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-weight: 800 !important;
    letter-spacing: -0.01em;
}

/* NEW: animated gradient underline on every subheader for a more premium feel */
.stMarkdown h3::after {
    content: "";
    display: block;
    width: 46px;
    height: 3px;
    margin-top: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}
.sub-header { font-size: 1rem; color: #64748b; text-align: center; margin-bottom: 0.3rem; font-weight: 300; }
.team-tag { text-align: center; margin-bottom: 1rem; }
.team-badge { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.35rem 1.8rem; border-radius: 50px; font-size: 0.82rem; font-weight: 600; }
.section-label { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.28rem 1.1rem; border-radius: 50px; font-size: 0.8rem; font-weight: 600; display: inline-block; margin-bottom: 0.9rem; box-shadow: 0 2px 8px #667eea40; }
.predict-card {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 60%, #1a1a2e 100%);
    border-radius: 24px; padding: 2.2rem 2rem; text-align: center; color: white;
    margin: 1.2rem 0; box-shadow: 0 24px 64px #0f346040; border: 1px solid #ffffff10;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.predict-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 32px 80px #0f346060;
}
.predict-card-label { font-size: 0.85rem; opacity: 0.65; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 700; }
.price-big {
    font-family: 'DM Serif Display', serif; font-size: 3.8rem; font-weight: 900;
    color: #ffd700; display: block; margin: 0.3rem 0; text-shadow: 0 0 40px #ffd70050;
    animation: priceReveal 0.5s ease;
}
@keyframes priceReveal {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}
.price-range { font-size: 0.9rem; color: #cbd5e1; font-weight: 600; margin-top: 0.6rem; }
.price-divider { width: 50px; height: 2px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 0.8rem auto; border-radius: 2px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #f1f5f9; padding: 5px; border-radius: 14px; border: 1px solid #e2e8f0; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 9px 20px; font-weight: 700; font-size: 0.92rem;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: #e8ecff; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important;
    box-shadow: 0 4px 12px #667eea40 !important; transform: scale(1.03);
}
/* ===== GLOBAL FIX v2: targets ALL descendant text, not just wrappers ===== */

/* st.caption() — the * targets every nested p/span/small Streamlit generates internally */
[data-testid="stCaptionContainer"] * {
    color: #334155 !important;
    font-weight: 600 !important;
    -webkit-text-fill-color: #334155 !important;
}
[data-testid="stCaptionContainer"] {
    font-size: 0.9rem !important;
}

/* st.image(..., caption=...) renders its caption via stCaptionContainer too, covered above,
   but also has its own figcaption-style wrapper in some versions — cover both */
[data-testid="stImageCaption"] *,
figcaption * {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* st.metric() label + value + delta */
[data-testid="stMetricLabel"] * {
    color: #334155 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] * {
    color: #0f172a !important;
    font-weight: 800 !important;
}

/* widget labels: number_input, slider, selectbox titles */
[data-testid="stWidgetLabel"] * {
    color: #1e293b !important;
    font-weight: 700 !important;
}

/* tooltip / help-icon popover text */
[data-testid="stTooltipContent"] * {
    color: #1e293b !important;
}

/* expander header ("View Derived Features...") */
[data-testid="stExpander"] summary * {
    color: #1e293b !important;
    font-weight: 700 !important;
}

/* generic markdown paragraph text app-wide */
[data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
}

/* ===== st.table() — REAL html table, this is the one to use instead of st.dataframe() ===== */
[data-testid="stTable"] table thead tr th {
    color: #0f172a !important;
    font-weight: 800 !important;
    background-color: #eef1f8 !important;
    border-bottom: 2px solid #c7d2fe !important;
}
[data-testid="stTable"] table tbody tr td {
    color: #1e293b !important;
    font-weight: 600 !important;
}
[data-testid="stTable"] table tbody tr:hover {
    background-color: #f8faff !important;
}

/* ===== st.dataframe() fallback note =====
   st.dataframe() draws text on a <canvas> — CSS CANNOT change canvas pixel color.
   Every st.dataframe() call below has been switched to st.table() in Block 2,
   which is why the rule above (not a dataframe rule) is what now applies. */
            
.stButton > button { border-radius: 50px !important; font-weight: 600 !important; }
div[data-testid="metric-container"] { background: linear-gradient(135deg, #f8faff, #f3f0ff); border: 1px solid #e0e7ff; border-radius: 14px; padding: 1rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

MODEL_PATHS = {
    "Batch GD":       "models/linear_batch_gd.pkl",
    "SGD":            "models/linear_sgd.pkl",
    "Mini-Batch GD":  "models/linear_minibatch.pkl",
    "Momentum":       "models/linear_momentum.pkl",
    "RMSProp":        "models/linear_rmsprop.pkl",
    "Adam (Linear)":  "models/linear_adam.pkl",
    "Neural Network": "models/neural_net_adam.pkl",
}

LOSS_PATHS = {
    "Batch GD":      "results/batch_loss.npy",
    "SGD":           "results/sgd_loss.npy",
    "Mini-Batch GD": "results/minibatch_loss.npy",
    "Momentum":      "results/momentum_loss_beta_09.npy",
    "RMSProp":       "results/rmsprop_loss.npy",
    "Adam (Linear)": "results/adam_loss.npy",
}

MODEL_DESC = {
    "Batch GD":       "Uses all training samples per update. Stable but slow.",
    "SGD":            "Updates per sample. Fast but noisy convergence.",
    "Mini-Batch GD":  "Batch size=32. Best balance of speed and stability.",
    "Momentum":       "Adds velocity term β=0.9 to accelerate descent.",
    "RMSProp":        "Adapts learning rate per feature using squared gradients.",
    "Adam (Linear)":  "Momentum + RMSProp combined. Best all-round optimizer.",
    "Neural Network": "3-layer NN (12→64→32→1, ReLU) trained with Adam.",
}

@st.cache_resource
def load_models():
    loaded = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except:
                pass
    return loaded

@st.cache_resource
def load_scaler():
    path = "models/scaler.pkl"
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_results():
    res = {}
    for key in ["y_pred_linreg", "y_pred_nn", "y_test", "X_train", "X_test"]:
        path = f"results/{key}.npy"
        if os.path.exists(path):
            res[key] = np.load(path)
    for name, path in LOSS_PATHS.items():
        if os.path.exists(path):
            res[f"loss_{name}"] = np.load(path)
    return res

models  = load_models()
scaler  = load_scaler()
results = load_results()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1.2rem 0 0.8rem 0;'>
        <div style='font-size:2.8rem;'>🏡</div>
        <div style='font-size:1.15rem; font-weight:700; color:#1a1a2e;'>House Price ML</div>
        <div style='font-size:0.75rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;'>California Housing</div>
        <div style='width:40px; height:2px; background:linear-gradient(90deg,#667eea,#764ba2); margin:0.6rem auto 0 auto; border-radius:2px;'></div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#374151; margin-bottom:0.4rem;'>🤖 Select Model</div>", unsafe_allow_html=True)
    selected = st.selectbox("Optimizer / Model:", list(MODEL_PATHS.keys()), label_visibility="collapsed")
    st.markdown(f"""<div style='background:linear-gradient(135deg,#f0f4ff,#f8f0ff); border-left:3px solid #667eea; border-radius:0 10px 10px 0; padding:0.7rem 0.9rem; font-size:0.85rem; font-weight:600; color:#3b0764; margin:0.3rem 0 0.5rem 0; line-height:1.5;'>{MODEL_DESC.get(selected, "")}</div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#374151; margin-bottom:0.5rem;'>📦 Model Status</div>", unsafe_allow_html=True)
    loaded_count = 0
    for name in MODEL_PATHS:
        if name in models:
            st.markdown(f"""<div style='background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:4px 10px; margin:3px 0; font-size:0.79rem; color:#15803d;'>✅ <b>{name}</b></div>""", unsafe_allow_html=True)
            loaded_count += 1
        else:
            st.markdown(f"""<div style='background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:4px 10px; margin:3px 0; font-size:0.79rem; color:#dc2626;'>❌ {name}</div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""<div style='font-size:0.78rem; color:#334155; font-weight:600; line-height:2.1;'>📊 <b>{loaded_count}/{len(MODEL_PATHS)}</b> models loaded<br>🗃️ 15,967 training samples<br>🔢 11 features (9 raw + 3 engineered)<br>🛠️ NumPy · Pandas · Streamlit<br>🎓 DST CURIE AI Internship 2026</div>""", unsafe_allow_html=True)
# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 0.8rem 0;'>
    <div style='font-size:0.78rem; letter-spacing:5px; text-transform:uppercase; color:#667eea; font-weight:700; margin-bottom:0.6rem;'>🎓 DST CURIE AI Internship 2026</div>
    <div style='font-family:"DM Serif Display",serif; font-size:3rem; font-weight:700; color:#1a1a2e; line-height:1.1; margin-bottom:0.6rem;'>🏡 California House Price Predictor</div>
    <div style='font-size:0.95rem; color:#475569; font-weight:600; margin-bottom:0.8rem;'>End-to-End ML Pipeline &nbsp;·&nbsp; Gradient Descent from Scratch &nbsp;·&nbsp; NumPy Only</div>
    <div style='width:70px; height:3px; background:linear-gradient(90deg,#667eea,#764ba2); margin:0 auto 1rem auto; border-radius:3px;'></div>
    <span style='background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:0.35rem 1.6rem; border-radius:50px; font-size:0.82rem; font-weight:700; letter-spacing:0.02em;'>🤖 7 Models &nbsp;·&nbsp; Built 100% from Scratch in NumPy</span>
</div>
""", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict",
    "📉 Loss Curves",
    "📊 Model Evaluation",
    "📈 Price Distribution",
    "ℹ️ About"
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📋 Enter House Features")
    st.caption("Fill in the details below — all values match the California Housing dataset range.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)

        longitude = st.number_input(
            "Longitude",
            value=-122.23, min_value=-124.5, max_value=-114.0, step=0.01,
            help="Western US longitude. San Francisco ≈ -122.4, Los Angeles ≈ -118.2")

        latitude = st.number_input(
            "Latitude",
            value=37.88, min_value=32.5, max_value=42.0, step=0.01,
            help="Northern US latitude. San Francisco ≈ 37.7, Los Angeles ≈ 34.0")

        ocean_options = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ocean_options,
            help="Distance category from the ocean. NEAR BAY = San Francisco Bay area.")

        housing_median_age = st.slider(
            "Housing Median Age (years)", 1, 52, 25,
            help="Median age of houses in the block. Older houses are typically cheaper.")

    with col2:
        st.markdown('<div class="section-label">👥 Population & Rooms</div>',
                    unsafe_allow_html=True)

        total_rooms = st.number_input(
            "Total Rooms",
            min_value=1, max_value=40000, value=2000,
            help="Total number of rooms in the block (not per house). Typical: 500–5000.")

        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=1, max_value=7000, value=400,
            help="Total bedrooms in the block. Should be less than Total Rooms.")

        population = st.number_input(
            "Population",
            min_value=1, max_value=40000, value=1000,
            help="Total people living in the block. Typical: 200–3000.")

        households = st.number_input(
            "Households",
            min_value=1, max_value=7000, value=350,
            help="Number of households in the block. Usually less than Population.")

        st.markdown('<div class="section-label">💰 Income</div>', unsafe_allow_html=True)

        median_income = st.number_input(
            "Median Income (tens of thousands $)",
            value=4.5, min_value=0.5, max_value=15.0, step=0.1,
            help="Median income in tens of thousands. 4.5 means ~$45,000/year. Range: 0.5–15.")

    # ── Input validation ───────────────────────────────────────────
    has_error = False
    if total_bedrooms >= total_rooms:
        st.warning("⚠️ Total Bedrooms should be less than Total Rooms. Please adjust.")
        has_error = True
    if households > population:
        st.warning("⚠️ Households should not exceed Population. Please adjust.")
        has_error = True

    # ── Feature builder ────────────────────────────────────────────
    def build_features():
        rooms_per_hh = total_rooms    / max(households, 1)
        bed_per_room = total_bedrooms / max(total_rooms, 1)
        pop_per_hh   = population     / max(households, 1)
        X_raw = np.array([[longitude, latitude, housing_median_age,
                           total_rooms, total_bedrooms, population,
                           households, median_income,
                           rooms_per_hh, bed_per_room, pop_per_hh]])
        return manual_scale(X_raw)

    st.divider()

    # ── Derived stats preview ──────────────────────────────────────
    with st.expander("🔍 View Derived Features (auto-calculated)"):
        rph = round(total_rooms / max(households, 1), 2)
        bpr = round(total_bedrooms / max(total_rooms, 1), 3)
        pph = round(population / max(households, 1), 2)
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Rooms per Household", rph, help="total_rooms / households")
        dc2.metric("Bedrooms per Room",   bpr, help="total_bedrooms / total_rooms")
        dc3.metric("People per Household", pph, help="population / households")

    pcol1, pcol2 = st.columns([1, 2])
    with pcol1:
        predict_btn = st.button(
            "🔮 Predict House Price",
            use_container_width=True,
            type="primary",
            disabled=has_error)
    with pcol2:
        compare_btn = st.button(
            "⚖️ Compare All Models",
            use_container_width=True,
            disabled=has_error)

    if has_error:
        st.caption("⛔ Fix the input warnings above before predicting.")

    if predict_btn and not has_error:
        model = models.get(selected)
        if model is None:
            st.error(f"❌ Model '{selected}' not loaded. Run retrain_all.py first.")
        else:
            with st.spinner(f"Running {selected}..."):
                try:
                    X = build_features()
                    raw   = float(np.array(model.predict(X)).flatten()[0])
                    # Hardcoded CA housing y statistics (y_mean/y_std npy files may be stale)
                    _ym   = 206855.0
                    _ys   = 115395.0
                    price = raw * _ys + _ym

                    if price < 10_000 or price > 5_000_000:
                        st.warning(f"⚠️ Prediction seems unusual (${price:,.0f}). "
                                   "Try retraining models with retrain_all.py.")
                    else:
                        st.markdown(f"""
                                                                  <div class="predict-card">
                                                                              <div class="predict-card-label">🤖 {selected} predicts</div>
                                                                              <span class="price-big">${price:,.0f}</span>
                                                                              <div class="price-divider"></div>
                                                                              <div class="price-range">
                                                                                  📉 Low: ${price*0.9:,.0f} &nbsp;|&nbsp; 📈 High: ${price*1.1:,.0f} &nbsp;(±10%)
                                                                              </div>
                                                                  </div>
                                                                  """, unsafe_allow_html=True)
                        st.caption(
                            f"Model: **{selected}** | "
                            f"Features: 12 | "
                            f"Dataset: Ccalifornia Housing (15,967 samples)")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    if compare_btn and not has_error:
        with st.spinner("Running all models..."):
            X = build_features()
            _ym2 = 206855.0
            _ys2 = 115395.0
            rows = []
            for name, model in models.items():
                try:
                    raw2 = float(np.array(model.predict(X)).flatten()[0])
                    p    = raw2 * _ys2 + _ym2
                    status = "✅ Realistic" if 10_000 < p < 5_000_000 else "⚠️ Check model"
                    rows.append({"Model": name,
                                 "Predicted Price": f"${p:,.0f}",
                                 "Status": status,
                                 "Raw": p})
                except Exception as e:
                    rows.append({"Model": name,
                                 "Predicted Price": f"Error: {e}",
                                 "Status": "❌ Error",
                                 "Raw": 0})

            df = pd.DataFrame(rows)
            st.subheader("⚖️ All Model Predictions")
            st.table(df[["Model", "Predicted Price", "Status"]].set_index("Model"))
            # Only chart realistic predictions
            valid = df[(df["Raw"] > 10_000) & (df["Raw"] < 5_000_000)]
            if not valid.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ["#2196F3" if "Adam" in n or "RMS" in n
                          else "#90CAF9" for n in valid["Model"]]
                bars = ax.barh(valid["Model"], valid["Raw"],
                               color=colors, edgecolor="white", height=0.5)
                ax.set_xlabel("Predicted Price ($)")
                ax.set_title("Model Comparison — Realistic Predictions Only",
                             fontweight="bold")
                ax.bar_label(bars,
                    labels=[f"${v:,.0f}" for v in valid["Raw"]],
                    padding=4, fontsize=8)
                ax.set_xlim(0, valid["Raw"].max() * 1.2)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No realistic predictions yet. Run retrain_all.py to fix model weights.")

# ════════════════════════════════════════════════════════════════════
# TAB 2 — LOSS CURVES
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📉 Training Loss Curves")
    st.caption("MSE loss over epochs for each optimizer — trained from scratch using NumPy only.")

    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    available = {k: results[f"loss_{k}"]
                 for k in LOSS_PATHS if f"loss_{k}" in results}

    if available:
        cols = st.columns(2)
        for i, (name, loss) in enumerate(available.items()):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(5, 2.8))
                ax.plot(loss, color=colors[i % len(colors)], linewidth=1.8)
                ax.set_title(name, fontsize=12, fontweight="bold", color="#0f172a")
                ax.set_xlabel("Epoch", fontsize=10, fontweight="bold", color="#1e293b")
                ax.set_ylabel("MSE Loss", fontsize=10, fontweight="bold", color="#1e293b")
                ax.tick_params(axis='both', labelsize=9, colors="#1e293b")
                ax.grid(True, alpha=0.3)
                final = loss[-1]
                ax.annotate(f"Final: {final:,.0f}",
                            xy=(len(loss)-1, final),
                            xytext=(-60, 10),
                            textcoords="offset points",
                            fontsize=8, color=colors[i % len(colors)],
                            arrowprops=dict(arrowstyle="->", color="gray"))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.divider()
        st.subheader("All Optimizers — Combined Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (name, loss) in enumerate(available.items()):
            display = loss[:300] if len(loss) > 300 else loss
            ax.plot(display, label=name,
                    color=colors[i % len(colors)], linewidth=1.8, alpha=0.85)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold", color="#1e293b")
        ax.set_ylabel("MSE Loss", fontsize=11, fontweight="bold", color="#1e293b")
        ax.set_title("Optimizer Convergence Comparison", fontweight="bold", color="#0f172a")
        ax.tick_params(axis='both', labelsize=9, colors="#1e293b")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Summary table
        st.subheader("📋 Convergence Summary")
        summary_rows = []
        for name, loss in available.items():
            summary_rows.append({
                "Optimizer": name,
                "Start Loss": f"{loss[0]:,.0f}",
                "Final Loss": f"{loss[-1]:,.0f}",
                "Epochs": len(loss),
                "Reduction %": f"{(1 - loss[-1]/loss[0])*100:.1f}%"
            })
        st.table(pd.DataFrame(summary_rows).set_index("Optimizer"))
    else:
        st.warning("No loss .npy files found in results/. Run retrain_all.py first.")

# ════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Model Evaluation on Test Set")

    y_test    = results.get("y_test")
    y_pred_lr = results.get("y_pred_linreg")
    y_pred_nn = results.get("y_pred_nn")

    if y_test is None:
        st.warning("results/y_test.npy not found. Run retrain_all.py first.")
    else:
        def get_metrics(y_true, y_pred, name):
            mse_val = np.mean((y_true - y_pred)**2)
            rmse    = np.sqrt(mse_val)
            mae     = np.mean(np.abs(y_true - y_pred))
            r2      = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-np.mean(y_true))**2)
            return {"Model": name,
                    "RMSE ($)": f"{rmse:,.0f}",
                    "MAE ($)":  f"{mae:,.0f}",
                    "R² Score": f"{r2:.4f}",
                    "R² (%)":   f"{r2*100:.1f}%"}

        rows = []
        if y_pred_lr is not None:
            rows.append(get_metrics(y_test, y_pred_lr, "Linear — Adam"))
        if y_pred_nn is not None:
            rows.append(get_metrics(y_test, y_pred_nn, "Neural Network"))
        if rows:
            st.table(pd.DataFrame(rows).set_index("Model"))
            st.caption("R² Score: 1.0 = perfect, 0.0 = predicts mean only. "
                       "Higher is better.")

        # Predicted vs Actual
        pairs = []
        if y_pred_lr is not None:
            pairs.append(("Linear — Adam", y_pred_lr, "#1f77b4"))
        if y_pred_nn is not None:
            pairs.append(("Neural Network", y_pred_nn, "#ff7f0e"))

        if pairs:
            st.subheader("🎯 Predicted vs Actual")
            pcols = st.columns(len(pairs))
            for i, (name, y_pred, color) in enumerate(pairs):
                with pcols[i]:
                    idx = np.random.choice(len(y_test),
                          min(800, len(y_test)), replace=False)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(y_test[idx], y_pred[idx],
                               alpha=0.3, s=8, color=color)
                    mn = min(y_test.min(), y_pred.min())
                    mx = max(y_test.max(), y_pred.max())
                    ax.plot([mn,mx],[mn,mx], "r--", linewidth=1.5,
                            label="Perfect prediction")
                    ax.set_xlabel("Actual Price ($)", fontsize=10, fontweight="bold", color="#1e293b")
                    ax.set_ylabel("Predicted Price ($)", fontsize=10, fontweight="bold", color="#1e293b")
                    ax.set_title(name, fontweight="bold", color="#0f172a")
                    ax.tick_params(axis='both', labelsize=9, colors="#1e293b")
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # Residuals
        if y_pred_lr is not None:
            st.subheader("📐 Residual Distribution")
            residuals = y_test - y_pred_lr
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(residuals, bins=60, color="#1f77b4",
                    edgecolor="white", alpha=0.8)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5,
                       label="Zero residual (perfect)")
            ax.set_xlabel("Residual = Actual − Predicted ($)", fontsize=11, fontweight="bold", color="#1e293b")
            ax.set_ylabel("Frequency", fontsize=11, fontweight="bold", color="#1e293b")
            ax.tick_params(axis='both', labelsize=9, colors="#1e293b")
            ax.set_title("Residual Distribution — Adam Linear Model",
                         fontweight="bold", color="#0f172a")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("A good model has residuals centered at 0 with a roughly normal shape.")

        # Week 6 plots
        st.subheader("🖼️ Week 6 Analysis Plots")
        plot_files = {
            "Feature Importance":  "results/plots/feature_importance_lr.png",
            "NN Learning Curve":   "results/plots/learning_curve_nn.png",
            "Residual Comparison": "results/plots/residual_comparison.png",
        }
        pcols2 = st.columns(3)
        for i, (label, path) in enumerate(plot_files.items()):
            with pcols2[i]:
                if os.path.exists(path):
                    st.image(path, caption=label, use_container_width=True)
                else:
                    st.caption(f"⚠️ {label} not found")

# ════════════════════════════════════════════════════════════════════
# TAB 4 — PRICE DISTRIBUTION
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 California Housing Price Distribution")
    st.caption("Distribution of median house values across the dataset.")

    y_test_data = results.get("y_test")

    if y_test_data is None:
        st.warning("results/y_test.npy not found. Run retrain_all.py first.")
    else:
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(y_test_data, bins=50,
                    edgecolor="white", alpha=0.85)

            ax.axvline(
                y_test_data.mean(),
                linestyle="--",
                linewidth=2,
                label=f"Mean: ${y_test_data.mean():,.0f}"
            )

            ax.axvline(
                np.median(y_test_data),
                linestyle="--",
                linewidth=2,
                label=f"Median: ${np.median(y_test_data):,.0f}"
            )

            ax.set_xlabel("House Price ($)", fontsize=11, fontweight="bold", color="#1e293b")
            ax.set_ylabel("Frequency", fontsize=11, fontweight="bold", color="#1e293b")
            ax.tick_params(axis='both', labelsize=9, colors="#1e293b")
            ax.set_title("Price Distribution — Test Set", fontweight="bold", color="#0f172a")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close()

        with col_d2:
            bins = [0,100000,200000,300000,400000,500000,float("inf")]
            labels = [
                "<$100K",
                "$100K–200K",
                "$200K–300K",
                "$300K–400K",
                "$400K–500K",
                ">$500K"
            ]

            counts = pd.cut(
                y_test_data,
                bins=bins,
                labels=labels
            ).value_counts().sort_index()

            fig2, ax2 = plt.subplots(figsize=(6,4))
            colors_bar = ["#d32f2f","#f57c00","#fbc02d","#388e3c","#1976d2","#7b1fa2"]
            ax2.bar(labels, counts.values, color=colors_bar, edgecolor="white", alpha=0.9)
            for i, v in enumerate(counts.values):
                ax2.text(i, v + 2, str(v), ha="center", fontsize=8)
            ax2.tick_params(axis="x", rotation=30)

            ax2.set_xlabel("Price Range", fontsize=11, fontweight="bold", color="#1e293b")
            ax2.set_ylabel("Number of Houses", fontsize=11, fontweight="bold", color="#1e293b")
            ax2.tick_params(axis='both', labelsize=9, colors="#1e293b")
            ax2.set_title("Houses by Price Range", fontweight="bold", color="#0f172a")
            st.pyplot(fig2)
            plt.close()

        st.divider()

        s1, s2, s3, s4 = st.columns(4)

        s1.metric("Mean Price",
                  f"${y_test_data.mean():,.0f}")
        s2.metric("Median Price",
                  f"${np.median(y_test_data):,.0f}")
        s3.metric("Min Price",
                  f"${y_test_data.min():,.0f}")
        s4.metric("Max Price",
                  f"${y_test_data.max():,.0f}")
with tab5:
    st.subheader("📖 About This Project")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Objective**
Complete ML pipeline for California house price prediction,
implementing all gradient descent optimizers from scratch — no scikit-learn for training.

**Dataset**
- California Housing Prices (Kaggle)
- 20,640 samples → 15,967 after cleaning
- Target: `median_house_value`
- 9 raw features → 12 engineered features

**Tech Stack**
- Python, NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn (preprocessing only)
- Streamlit (this app)
        """)
    with col_b:
        st.markdown("""
**Week-by-Week Plan**

| Week | Task |
|------|------|
| 1 | Setup & EDA |
| 2 | Linear Regression + Batch GD |
| 3 | SGD & Mini-Batch GD |
| 4 | Momentum, RMSProp, Adam |
| 5 | Neural Network from scratch |
| 6 | Model Evaluation & Tuning |
| 7 | This Web App ← **Current** |
| 8 | Deployment & Documentation |
        """)

    st.divider()
    st.subheader("🧠 All Models — Description & Status")
    for name, desc in MODEL_DESC.items():
        status = "✅ Loaded" if name in models else "❌ Not found"
        col_i, col_j = st.columns([1, 3])
        with col_i:
            st.markdown(f"**{name}**  \n{status}")
        with col_j:
            st.markdown(f"<span style='color:#334155; font-weight:600; font-size:0.9rem;'>{desc}</span>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📐 Feature Engineering")
    fe_data = {
        "Feature": ["rooms_per_hh", "bed_per_room", "pop_per_hh"],
        "Formula": ["total_rooms / households",
                    "total_bedrooms / total_rooms",
                    "population / households"],
        "Why": ["Captures house size relative to block",
                "Captures bedroom density",
                "Captures household crowding"]
    }
    st.table(pd.DataFrame(fe_data).set_index("Feature"))