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
.main-header {
    font-size: 2.5rem; font-weight: 700;
    color: #1f77b4; text-align: center; padding-bottom: 0.3rem;
}
.sub-header {
    font-size: 1.05rem; color: #666;
    text-align: center; margin-bottom: 1.2rem;
}
.section-label {
    background-color: #f0f4ff;
    border-left: 4px solid #1f77b4;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.predict-card {
    background: linear-gradient(135deg, #e8f4fd, #f0f8e8);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    border: 1px solid #c8e6c9;
}
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
    st.title("🏠 House Price ML")
    st.markdown("---")
    st.subheader("🤖 Select Model")
    selected = st.selectbox("Optimizer / Model:", list(MODEL_PATHS.keys()))
    st.info(MODEL_DESC.get(selected, ""))
    st.markdown("---")
    st.subheader("📦 Model Status")
    loaded_count = 0
    for name in MODEL_PATHS:
        if name in models:
            st.caption(f"✅ {name}")
            loaded_count += 1
        else:
            st.caption(f"❌ {name} — not found")
    st.markdown("---")
    st.caption(f"**{loaded_count}/{len(MODEL_PATHS)} models loaded**")
    st.caption("📊 California Housing | 15,967 samples")
    st.caption("🔧 NumPy · Scikit-learn · Streamlit")

# ── Header ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏠 House Price Prediction Dashboard</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">End-to-End ML Pipeline | Gradient Descent Variants from Scratch</p>',
            unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict", "📉 Loss Curves", "📊 Model Evaluation", "ℹ️ About"
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
                        st.success("✅ Prediction complete!")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("💰 Predicted Price", f"${price:,.0f}")
                        m2.metric("📉 Lower (−10%)",    f"${price*0.9:,.0f}")
                        m3.metric("📈 Upper (+10%)",    f"${price*1.1:,.0f}")
                        st.caption(
                            f"Model: **{selected}** | "
                            f"Features: 12 | "
                            f"Dataset: California Housing (15,967 samples)")
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
            st.dataframe(df[["Model", "Predicted Price", "Status"]],
                         use_container_width=True, hide_index=True)

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
                ax.set_title(name, fontsize=11, fontweight="bold")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MSE Loss")
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
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Optimizer Convergence Comparison", fontweight="bold")
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
        st.dataframe(pd.DataFrame(summary_rows),
                     use_container_width=True, hide_index=True)
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
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)
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
                    ax.set_xlabel("Actual Price ($)")
                    ax.set_ylabel("Predicted Price ($)")
                    ax.set_title(name, fontweight="bold")
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
            ax.set_xlabel("Residual = Actual − Predicted ($)")
            ax.set_ylabel("Frequency")
            ax.set_title("Residual Distribution — Adam Linear Model",
                         fontweight="bold")
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
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════
with tab4:
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
            st.caption(desc)

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
    st.dataframe(pd.DataFrame(fe_data), use_container_width=True, hide_index=True)