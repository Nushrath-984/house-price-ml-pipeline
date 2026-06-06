# retrain_all.py — Final version with y-scaling (no overflow)
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.makedirs("models",         exist_ok=True)
os.makedirs("results",        exist_ok=True)
os.makedirs("results/plots",  exist_ok=True)

# ── Model classes (must be here so joblib can pickle/unpickle) ──────
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

class SavedModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def predict(self, X):
        return np.array(X) @ self.w + self.b

class NNModel:
    def __init__(self, W1, b1, W2, b2, W3, b3):
        self.W1 = W1; self.b1 = b1
        self.W2 = W2; self.b2 = b2
        self.W3 = W3; self.b3 = b3
    def predict(self, X):
        X  = np.array(X, dtype=float)
        a1 = relu(X  @ self.W1 + self.b1)
        a2 = relu(a1 @ self.W2 + self.b2)
        return (a2 @ self.W3 + self.b3).flatten()

# ── Load & prepare data ──────────────────────────────────────────────
print("Loading data...")
# Try common file locations
for path in ["data/final_prepared.csv", "data/housing.csv", "housing.csv"]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Loaded from: {path}  ({len(df)} rows)")
        break
else:
    raise FileNotFoundError("No CSV found. Place housing.csv in project root or data/.")

# Drop rows with missing values
df = df.dropna()

# Encode ocean_proximity if present
if "ocean_proximity" in df.columns:
    ocean_map = {"NEAR BAY": 3.0, "<1H OCEAN": 0.0,
                 "INLAND": 1.0, "NEAR OCEAN": 4.0, "ISLAND": 2.0}
    df["ocean_proximity"] = df["ocean_proximity"].map(ocean_map)
    df = df.dropna(subset=["ocean_proximity"])

# Feature engineering
df["rooms_per_hh"]  = df["total_rooms"]    / df["households"].clip(lower=1)
df["bed_per_room"]  = df["total_bedrooms"] / df["total_rooms"].clip(lower=1)
df["pop_per_hh"]    = df["population"]     / df["households"].clip(lower=1)

feature_cols = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
    "rooms_per_hh", "bed_per_room", "pop_per_hh"
]

X = df[feature_cols].values
y = df["median_house_value"].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

# Scale X
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# Scale y  ← THIS is the key fix that prevents overflow
y_mean    = y_train.mean()
y_std     = y_train.std()
y_train_s = (y_train - y_mean) / y_std
y_test_s  = (y_test  - y_mean) / y_std

# Save y stats so app.py can unscale predictions
np.save("results/y_mean.npy", np.array([y_mean]))
np.save("results/y_std.npy",  np.array([y_std]))
print(f"  y_mean={y_mean:,.0f}  y_std={y_std:,.0f}")

n, d = X_train_s.shape

# ── Helpers ──────────────────────────────────────────────────────────
def predict_linear(X, w, b):
    return X @ w + b

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_grads(X, y, w, b):
    n   = len(y)
    err = predict_linear(X, w, b) - y
    dw  = (2/n) * X.T @ err
    db  = (2/n) * err.sum()
    return dw, db

# ── 1. Batch GD ──────────────────────────────────────────────────────
print("\nTraining Batch GD...")
w = np.zeros(d); b = 0.0
lr = 0.05; epochs = 500; losses = []
for ep in range(epochs):
    dw, db = compute_grads(X_train_s, y_train_s, w, b)
    w -= lr * dw
    b -= lr * db
    losses.append(mse(y_train_s, predict_linear(X_train_s, w, b)))
np.save("results/batch_loss.npy", np.array(losses))
joblib.dump(SavedModel(w.copy(), b), "models/linear_batch_gd.pkl")
print(f"  Final loss: {losses[-1]:.4f}")

# ── 2. SGD ───────────────────────────────────────────────────────────
print("Training SGD...")
w = np.zeros(d); b = 0.0
lr = 0.001; epochs = 50; losses = []
rng = np.random.default_rng(0)
for ep in range(epochs):
    idx = rng.permutation(n)
    for i in idx:
        xi = X_train_s[i:i+1]
        yi = y_train_s[i:i+1]
        dw, db = compute_grads(xi, yi, w, b)
        w -= lr * dw
        b -= lr * db
    losses.append(mse(y_train_s, predict_linear(X_train_s, w, b)))
np.save("results/sgd_loss.npy", np.array(losses))
joblib.dump(SavedModel(w.copy(), b), "models/linear_sgd.pkl")
print(f"  Final loss: {losses[-1]:.4f}")

# ── 3. Mini-Batch GD ─────────────────────────────────────────────────
print("Training Mini-Batch GD...")
w = np.zeros(d); b = 0.0
lr = 0.05; batch_size = 32; epochs = 200; losses = []
rng = np.random.default_rng(1)
for ep in range(epochs):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start+batch_size]
        dw, db = compute_grads(X_train_s[batch], y_train_s[batch], w, b)
        w -= lr * dw
        b -= lr * db
    losses.append(mse(y_train_s, predict_linear(X_train_s, w, b)))
np.save("results/minibatch_loss.npy", np.array(losses))
joblib.dump(SavedModel(w.copy(), b), "models/linear_minibatch.pkl")
print(f"  Final loss: {losses[-1]:.4f}")

# ── 4. Momentum ──────────────────────────────────────────────────────
print("Training Momentum...")
w = np.zeros(d); b = 0.0
lr = 0.05; beta = 0.9; epochs = 500; losses = []
vw = np.zeros(d); vb = 0.0
for ep in range(epochs):
    dw, db = compute_grads(X_train_s, y_train_s, w, b)
    vw = beta * vw + (1-beta) * dw
    vb = beta * vb + (1-beta) * db
    w -= lr * vw
    b -= lr * vb
    losses.append(mse(y_train_s, predict_linear(X_train_s, w, b)))
np.save("results/momentum_loss_beta_09.npy", np.array(losses))
joblib.dump(SavedModel(w.copy(), b), "models/linear_momentum.pkl")
print(f"  Final loss: {losses[-1]:.4f}")

# ── 5. RMSProp ───────────────────────────────────────────────────────
print("Training RMSProp...")
w = np.zeros(d); b = 0.0
lr = 0.01; rho = 0.9; eps = 1e-8; epochs = 500; losses = []
sw = np.zeros(d); sb = 0.0
for ep in range(epochs):
    dw, db = compute_grads(X_train_s, y_train_s, w, b)
    sw = rho * sw + (1-rho) * dw**2
    sb = rho * sb + (1-rho) * db**2
    w -= lr * dw / (np.sqrt(sw) + eps)
    b -= lr * db / (np.sqrt(sb) + eps)
    losses.append(mse(y_train_s, predict_linear(X_train_s, w, b)))
np.save("results/rmsprop_loss.npy", np.array(losses))
joblib.dump(SavedModel(w.copy(), b), "models/linear_rmsprop.pkl")
print(f"  Final loss: {losses[-1]:.4f}")

# ── 6. Adam (Linear) ─────────────────────────────────────────────────
print("Training Adam (Linear)...")
w_a = np.zeros(d); b_a = 0.0
lr_a = 0.01; beta1 = 0.9; beta2 = 0.999; eps_a = 1e-8; epochs = 500
mw = np.zeros(d); mb = 0.0
vw = np.zeros(d); vb = 0.0
losses = []
for ep in range(1, epochs+1):
    dw, db = compute_grads(X_train_s, y_train_s, w_a, b_a)
    mw = beta1*mw + (1-beta1)*dw
    mb = beta1*mb + (1-beta1)*db
    vw = beta2*vw + (1-beta2)*dw**2
    vb = beta2*vb + (1-beta2)*db**2
    mw_c = mw / (1 - beta1**ep)
    mb_c = mb / (1 - beta1**ep)
    vw_c = vw / (1 - beta2**ep)
    vb_c = vb / (1 - beta2**ep)
    w_a -= lr_a * mw_c / (np.sqrt(vw_c) + eps_a)
    b_a -= lr_a * mb_c / (np.sqrt(vb_c) + eps_a)
    losses.append(mse(y_train_s, predict_linear(X_train_s, w_a, b_a)))
np.save("results/adam_loss.npy", np.array(losses))
joblib.dump(SavedModel(w_a.copy(), b_a), "models/linear_adam.pkl")
print(f"  Final loss: {losses[-1]:.4f}")

# ── 7. Neural Network (Adam) ─────────────────────────────────────────
print("Training Neural Network (this takes ~30 seconds)...")
np.random.seed(42)
input_dim = d; h1 = 64; h2 = 32

W1_n = np.random.randn(input_dim, h1) * np.sqrt(2/input_dim)
b1_n = np.zeros(h1)
W2_n = np.random.randn(h1, h2)       * np.sqrt(2/h1)
b2_n = np.zeros(h2)
W3_n = np.random.randn(h2, 1)        * np.sqrt(2/h2)
b3_n = np.zeros(1)

lr_nn = 0.001; beta1 = 0.9; beta2 = 0.999; eps_nn = 1e-8
epochs_nn = 200; batch_size = 64

params  = [W1_n, b1_n, W2_n, b2_n, W3_n, b3_n]
m_p     = [np.zeros_like(p) for p in params]
v_p     = [np.zeros_like(p) for p in params]
t       = 0
losses_nn = []
rng = np.random.default_rng(42)

for ep in range(epochs_nn):
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch = idx[start:start+batch_size]
        Xb    = X_train_s[batch]
        yb    = y_train_s[batch]

        # Forward
        z1 = Xb @ W1_n + b1_n;  a1 = relu(z1)
        z2 = a1 @ W2_n + b2_n;  a2 = relu(z2)
        y_hat = (a2 @ W3_n + b3_n).flatten()

        # Backward
        err   = y_hat - yb
        dW3   = a2.T   @ err[:,None] / len(yb)
        db3   = err.mean(keepdims=True)
        da2   = err[:,None] @ W3_n.T * relu_grad(z2)
        dW2   = a1.T   @ da2  / len(yb)
        db2   = da2.mean(axis=0)
        da1   = da2    @ W2_n.T * relu_grad(z1)
        dW1   = Xb.T   @ da1  / len(yb)
        db1   = da1.mean(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m_p[i] = beta1*m_p[i] + (1-beta1)*g
            v_p[i] = beta2*v_p[i] + (1-beta2)*g**2
            mc = m_p[i] / (1 - beta1**t)
            vc = v_p[i] / (1 - beta2**t)
            params[i] = p - lr_nn * mc / (np.sqrt(vc) + eps_nn)

        W1_n,b1_n,W2_n,b2_n,W3_n,b3_n = params

    z1 = X_train_s @ W1_n + b1_n;  a1 = relu(z1)
    z2 = a1        @ W2_n + b2_n;  a2 = relu(z2)
    y_hat = (a2 @ W3_n + b3_n).flatten()
    ep_loss = mse(y_train_s, y_hat)
    losses_nn.append(ep_loss)
    if ep % 20 == 0:
        print(f"  Epoch {ep:3d}: loss={ep_loss:.4f}")

joblib.dump(NNModel(W1_n,b1_n,W2_n,b2_n,W3_n,b3_n), "models/neural_net_adam.pkl")
print(f"  Final loss: {losses_nn[-1]:.4f}")

# ── Save test predictions (unscaled back to dollars) ─────────────────
print("\nSaving test predictions...")
adam_model = joblib.load("models/linear_adam.pkl")
nn_model   = joblib.load("models/neural_net_adam.pkl")

y_pred_lr = adam_model.predict(X_test_s) * y_std + y_mean
y_pred_nn = nn_model.predict(X_test_s)   * y_std + y_mean

np.save("results/y_test.npy",        y_test)
np.save("results/y_pred_linreg.npy", y_pred_lr)
np.save("results/y_pred_nn.npy",     y_pred_nn)

print("\n✅ All models retrained and saved!")
print("   Run: streamlit run app.py")