# retrain_all.py — Retrain all models with consistent scaler

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── Load & preprocess ──────────────────────────────────────────────
df = pd.read_csv("data/housing.csv")
df.dropna(inplace=True)

ocean_map = {"NEAR BAY": 3.0, "<1H OCEAN": 0.0,
             "INLAND": 1.0, "NEAR OCEAN": 4.0, "ISLAND": 2.0}
df["ocean_proximity"] = df["ocean_proximity"].map(ocean_map)

df["rooms_per_hh"]  = df["total_rooms"]    / df["households"]
df["bed_per_room"]  = df["total_bedrooms"] / df["total_rooms"]
df["pop_per_hh"]    = df["population"]     / df["households"]

feature_cols = ["longitude","latitude","housing_median_age",
                "total_rooms","total_bedrooms","population",
                "households","median_income",
                "rooms_per_hh","bed_per_room","pop_per_hh",
                "ocean_proximity"]

X = df[feature_cols].values
y = df["median_house_value"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Save scaler and test arrays
joblib.dump(scaler, "models/scaler.pkl")
np.save("results/y_test.npy", y_test)
np.save("results/X_train.npy", X_train_s)
np.save("results/X_test.npy",  X_test_s)

print(f"Data: {X_train_s.shape[0]} train, {X_test_s.shape[0]} test samples")

# ── Optimizer implementations ──────────────────────────────────────
n, d = X_train_s.shape

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def predict_linear(X, w, b):
    return X @ w + b

def compute_grads(X, y, w, b):
    m = len(y)
    y_pred = predict_linear(X, w, b)
    err = y_pred - y
    dw = (2/m) * X.T @ err
    db = (2/m) * np.sum(err)
    return dw, db

# ── 1. Batch GD ────────────────────────────────────────────────────
print("Training Batch GD...")
w = np.zeros(d); b = 0.0
lr = 0.01; epochs = 500
losses = []
for ep in range(epochs):
    dw, db = compute_grads(X_train_s, y_train, w, b)
    w -= lr * dw
    b -= lr * db
    losses.append(mse(y_train, predict_linear(X_train_s, w, b)))

np.save("results/batch_loss.npy", np.array(losses))

class SavedModel:
    def __init__(self, w, b):
        self.w = w; self.b = b
    def predict(self, X):
        return np.array(X) @ self.w + self.b

joblib.dump(SavedModel(w, b), "models/linear_batch_gd.pkl")
print(f"  Final loss: {losses[-1]:,.0f}")

# ── 2. SGD ─────────────────────────────────────────────────────────
print("Training SGD...")
w = np.zeros(d); b = 0.0
lr = 0.001; epochs = 100
losses = []
idx = np.arange(n)
for ep in range(epochs):
    np.random.shuffle(idx)
    ep_loss = 0
    for i in idx:
        xi = X_train_s[i:i+1]
        yi = y_train[i:i+1]
        dw, db = compute_grads(xi, yi, w, b)
        w -= lr * dw
        b -= lr * db
        ep_loss += mse(yi, predict_linear(xi, w, b))
    losses.append(ep_loss / n)

np.save("results/sgd_loss.npy", np.array(losses))
joblib.dump(SavedModel(w, b), "models/linear_sgd.pkl")
print(f"  Final loss: {losses[-1]:,.0f}")

# ── 3. Mini-Batch GD ───────────────────────────────────────────────
print("Training Mini-Batch GD...")
w = np.zeros(d); b = 0.0
lr = 0.01; epochs = 300; batch_size = 32
losses = []
for ep in range(epochs):
    perm = np.random.permutation(n)
    ep_loss = 0; count = 0
    for start in range(0, n, batch_size):
        batch = perm[start:start+batch_size]
        xb, yb = X_train_s[batch], y_train[batch]
        dw, db = compute_grads(xb, yb, w, b)
        w -= lr * dw
        b -= lr * db
        ep_loss += mse(yb, predict_linear(xb, w, b))
        count += 1
    losses.append(ep_loss / count)

np.save("results/minibatch_loss.npy", np.array(losses))
joblib.dump(SavedModel(w, b), "models/linear_minibatch.pkl")
print(f"  Final loss: {losses[-1]:,.0f}")

# ── 4. Momentum ────────────────────────────────────────────────────
print("Training Momentum...")
w = np.zeros(d); b = 0.0
lr = 0.01; beta = 0.9; epochs = 500
vw = np.zeros(d); vb = 0.0
losses = []
for ep in range(epochs):
    dw, db = compute_grads(X_train_s, y_train, w, b)
    vw = beta * vw + (1 - beta) * dw
    vb = beta * vb + (1 - beta) * db
    w -= lr * vw
    b -= lr * vb
    losses.append(mse(y_train, predict_linear(X_train_s, w, b)))

np.save("results/momentum_loss_beta_09.npy", np.array(losses))
joblib.dump(SavedModel(w, b), "models/linear_momentum.pkl")
print(f"  Final loss: {losses[-1]:,.0f}")

# ── 5. RMSProp ─────────────────────────────────────────────────────
print("Training RMSProp...")
w = np.zeros(d); b = 0.0
lr = 0.01; rho = 0.9; eps = 1e-8; epochs = 500
sw = np.zeros(d); sb = 0.0
losses = []
for ep in range(epochs):
    dw, db = compute_grads(X_train_s, y_train, w, b)
    sw = rho * sw + (1 - rho) * dw**2
    sb = rho * sb + (1 - rho) * db**2
    w -= lr / (np.sqrt(sw) + eps) * dw
    b -= lr / (np.sqrt(sb) + eps) * db
    losses.append(mse(y_train, predict_linear(X_train_s, w, b)))

np.save("results/rmsprop_loss.npy", np.array(losses))
joblib.dump(SavedModel(w, b), "models/linear_rmsprop.pkl")
print(f"  Final loss: {losses[-1]:,.0f}")

# ── 6. Adam (Linear) ───────────────────────────────────────────────
print("Training Adam (Linear)...")
w = np.zeros(d); b = 0.0
lr = 0.01; b1 = 0.9; b2 = 0.999; eps = 1e-8; epochs = 500
mw = np.zeros(d); mb_ = 0.0
vw = np.zeros(d); vb_ = 0.0
losses = []
for ep in range(1, epochs+1):
    dw, db = compute_grads(X_train_s, y_train, w, b)
    mw = b1*mw + (1-b1)*dw;  mb_ = b1*mb_ + (1-b1)*db
    vw = b2*vw + (1-b2)*dw**2; vb_ = b2*vb_ + (1-b2)*db**2
    mw_c = mw/(1-b1**ep);  mb_c = mb_/(1-b1**ep)
    vw_c = vw/(1-b2**ep);  vb_c = vb_/(1-b2**ep)
    w -= lr * mw_c / (np.sqrt(vw_c) + eps)
    b -= lr * mb_c / (np.sqrt(vb_c) + eps)
    losses.append(mse(y_train, predict_linear(X_train_s, w, b)))

np.save("results/adam_loss.npy", np.array(losses))
joblib.dump(SavedModel(w, b), "models/linear_adam.pkl")
print(f"  Final loss: {losses[-1]:,.0f}")

# ── 7. Neural Network (Adam) ───────────────────────────────────────
print("Training Neural Network...")

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

np.random.seed(42)
W1 = np.random.randn(d, 64) * 0.01;  b1_n = np.zeros(64)
W2 = np.random.randn(64, 32) * 0.01; b2_n = np.zeros(32)
W3 = np.random.randn(32, 1)  * 0.01; b3_n = np.zeros(1)

lr = 0.001; b1a = 0.9; b2a = 0.999; eps = 1e-8
epochs = 200; batch_size = 64

params = [W1, b1_n, W2, b2_n, W3, b3_n]
m_params = [np.zeros_like(p) for p in params]
v_params = [np.zeros_like(p) for p in params]
t = 0; losses = []

for ep in range(epochs):
    perm = np.random.permutation(n)
    ep_loss = 0; count = 0
    for start in range(0, n, batch_size):
        batch = perm[start:start+batch_size]
        xb = X_train_s[batch]
        yb = y_train[batch].reshape(-1, 1)
        t += 1

        # Forward
        z1 = xb @ W1 + b1_n
        a1 = relu(z1)
        z2 = a1 @ W2 + b2_n
        a2 = relu(z2)
        z3 = a2 @ W3 + b3_n
        y_pred = z3

        loss = np.mean((y_pred - yb)**2)
        ep_loss += loss; count += 1

        # Backward
        m = len(yb)
        dz3 = 2*(y_pred - yb)/m
        dW3 = a2.T @ dz3;         db3 = dz3.sum(axis=0)
        da2 = dz3 @ W3.T
        dz2 = da2 * relu_grad(z2)
        dW2 = a1.T @ dz2;         db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = xb.T @ dz1;         db1 = dz1.sum(axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m_params[i] = b1a*m_params[i] + (1-b1a)*g
            v_params[i] = b2a*v_params[i] + (1-b2a)*g**2
            mc = m_params[i]/(1-b1a**t)
            vc = v_params[i]/(1-b2a**t)
            new_params.append(p - lr * mc/(np.sqrt(vc)+eps))
        params = new_params
        W1,b1_n,W2,b2_n,W3,b3_n = params
        m_params_updated = m_params; v_params_updated = v_params

    losses.append(ep_loss/count)
    if ep % 20 == 0:
        print(f"  Epoch {ep}: loss={ep_loss/count:,.0f}")

np.save("results/nn_loss.npy", np.array(losses))

class NNModel:
    def __init__(self, W1,b1,W2,b2,W3,b3):
        self.W1=W1; self.b1=b1
        self.W2=W2; self.b2=b2
        self.W3=W3; self.b3=b3
    def predict(self, X):
        X = np.array(X)
        a1 = relu(X  @ self.W1 + self.b1)
        a2 = relu(a1 @ self.W2 + self.b2)
        return (a2 @ self.W3 + self.b3).flatten()

joblib.dump(NNModel(W1,b1_n,W2,b2_n,W3,b3_n), "models/neural_net_adam.pkl")
print(f"  Final loss: {losses[-1]:,.0f}")

# ── Save test predictions ──────────────────────────────────────────
print("\nSaving test predictions...")
best_lin = joblib.load("models/linear_adam.pkl")
nn_model = joblib.load("models/neural_net_adam.pkl")

y_pred_lr = best_lin.predict(X_test_s)
y_pred_nn = nn_model.predict(X_test_s)

np.save("results/y_pred_linreg.npy", y_pred_lr)
np.save("results/y_pred_nn.npy",     y_pred_nn)

print("\n✅ All models retrained and saved!")
print("Run: streamlit run app.py")