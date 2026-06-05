# save_models.py — Week 7 Member 1
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, '.')
from src.linear_regression import LinearRegressionScratch
from src.neural_net import NeuralNetworkScratch
from src.optimizers import OptimizerEngine
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

print("="*55)
print("Loading data/final_prepared.csv ...")
print("="*55)

df = pd.read_csv("data/final_prepared.csv")
X = df.drop(columns=["median_house_value"]).values
y = df["median_house_value"].values
print(f"X shape: {X.shape}, y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

def mse(yt, yp): return np.mean((yt - yp.flatten())**2)

n = X_train.shape[1]

# Batch GD
print("\nTraining Batch GD...")
W, b = np.zeros(n), 0.0
for _ in range(1000):
    yp = X_train @ W + b
    dW = (2/len(y_train)) * X_train.T @ (yp - y_train)
    db = (2/len(y_train)) * np.sum(yp - y_train)
    W -= 0.01 * dW
    b -= 0.01 * db
m1 = LinearRegressionScratch(learning_rate=0.01, epochs=1000)
m1.W = W; m1.b = b
joblib.dump(m1, "models/linear_batch_gd.pkl")
print(f"  [OK] linear_batch_gd.pkl  | MSE: {mse(y_test, X_test@W+b):.0f}")

# SGD
print("\nTraining SGD...")
W, b = np.zeros(n), 0.0
for _ in range(50):
    for i in np.random.permutation(len(y_train)):
        xi = X_train[i:i+1]
        yi = y_train[i]
        yp_i = (xi @ W + b)[0]
        W -= 0.001 * 2 * xi.flatten() * (yp_i - yi)
        b -= 0.001 * 2 * (yp_i - yi)
m2 = LinearRegressionScratch(learning_rate=0.001, epochs=50)
m2.W = W; m2.b = b
joblib.dump(m2, "models/linear_sgd.pkl")
print(f"  [OK] linear_sgd.pkl       | MSE: {mse(y_test, X_test@W+b):.0f}")

# Mini-Batch GD
print("\nTraining Mini-Batch GD...")
W, b = np.zeros(n), 0.0
for _ in range(500):
    for s in range(0, len(y_train), 32):
        idx = np.random.permutation(len(y_train))[s:s+32]
        Xb, yb = X_train[idx], y_train[idx]
        yp = Xb @ W + b
        dW = (2/len(yb)) * Xb.T @ (yp - yb)
        db = (2/len(yb)) * np.sum(yp - yb)
        W -= 0.01 * dW
        b -= 0.01 * db
m3 = LinearRegressionScratch(learning_rate=0.01, epochs=500)
m3.W = W; m3.b = b
joblib.dump(m3, "models/linear_minibatch.pkl")
print(f"  [OK] linear_minibatch.pkl | MSE: {mse(y_test, X_test@W+b):.0f}")

# Momentum
print("\nTraining Momentum...")
W, b, v = np.zeros(n), 0.0, np.zeros(n)
for _ in range(1000):
    yp = X_train @ W + b
    dW = (2/len(y_train)) * X_train.T @ (yp - y_train)
    db = (2/len(y_train)) * np.sum(yp - y_train)
    v = 0.9*v + 0.1*dW
    W -= 0.01 * v
    b -= 0.01 * db
m4 = LinearRegressionScratch(learning_rate=0.01, epochs=1000)
m4.W = W; m4.b = b
joblib.dump(m4, "models/linear_momentum.pkl")
print(f"  [OK] linear_momentum.pkl  | MSE: {mse(y_test, X_test@W+b):.0f}")

# RMSProp
print("\nTraining RMSProp...")
W, b, s = np.zeros(n), 0.0, np.zeros(n)
for _ in range(1000):
    yp = X_train @ W + b
    dW = (2/len(y_train)) * X_train.T @ (yp - y_train)
    db = (2/len(y_train)) * np.sum(yp - y_train)
    s = 0.9*s + 0.1*dW**2
    W -= 0.01 * dW / (np.sqrt(s) + 1e-8)
    b -= 0.01 * db
m5 = LinearRegressionScratch(learning_rate=0.01, epochs=1000)
m5.W = W; m5.b = b
joblib.dump(m5, "models/linear_rmsprop.pkl")
print(f"  [OK] linear_rmsprop.pkl   | MSE: {mse(y_test, X_test@W+b):.0f}")

# Adam Linear
print("\nTraining Adam (Linear)...")
W, b = np.zeros(n), 0.0
m_w, v_w = np.zeros(n), np.zeros(n)
for t in range(1, 1001):
    yp = X_train @ W + b
    dW = (2/len(y_train)) * X_train.T @ (yp - y_train)
    db = (2/len(y_train)) * np.sum(yp - y_train)
    m_w = 0.9*m_w + 0.1*dW
    v_w = 0.999*v_w + 0.001*dW**2
    W -= 0.01 * (m_w/(1-0.9**t)) / (np.sqrt(v_w/(1-0.999**t)) + 1e-8)
    b -= 0.01 * db
m6 = LinearRegressionScratch(learning_rate=0.01, epochs=1000)
m6.W = W; m6.b = b
joblib.dump(m6, "models/linear_adam.pkl")
print(f"  [OK] linear_adam.pkl      | MSE: {mse(y_test, X_test@W+b):.0f}")

# Neural Network (Adam)
print("\nTraining Neural Network (Adam)...")
nn = NeuralNetworkScratch(input_size=12, learning_rate=0.001)

opt_w1 = OptimizerEngine(optimizer_type="adam", learning_rate=0.001)
opt_b1 = OptimizerEngine(optimizer_type="adam", learning_rate=0.001)
opt_w2 = OptimizerEngine(optimizer_type="adam", learning_rate=0.001)
opt_b2 = OptimizerEngine(optimizer_type="adam", learning_rate=0.001)
opt_w3 = OptimizerEngine(optimizer_type="adam", learning_rate=0.001)
opt_b3 = OptimizerEngine(optimizer_type="adam", learning_rate=0.001)

y_train_nn = y_train.reshape(-1, 1)

for epoch in range(300):
    y_pred_nn, cache = nn.forward(X_train)
    grads = nn.backward(cache, y_train_nn)
    params = nn.get_params()
    nn.set_params({
        'W1': opt_w1.update_params(params['W1'], grads['dW1'], epoch),
        'b1': opt_b1.update_params(params['b1'], grads['db1'], epoch),
        'W2': opt_w2.update_params(params['W2'], grads['dW2'], epoch),
        'b2': opt_b2.update_params(params['b2'], grads['db2'], epoch),
        'W3': opt_w3.update_params(params['W3'], grads['dW3'], epoch),
        'b3': opt_b3.update_params(params['b3'], grads['db3'], epoch),
    })
    if (epoch+1) % 100 == 0:
        loss = np.mean((nn.forward(X_train)[0] - y_train_nn)**2)
        print(f"  Epoch {epoch+1}/300 | Loss: {loss:.0f}")

joblib.dump(nn, "models/neural_net_adam.pkl")
nn_mse = mse(y_test, nn.predict(X_test).flatten())
print(f"  [OK] neural_net_adam.pkl  | MSE: {nn_mse:.0f}")

# Final Verification
print("\n" + "="*55)
print("FINAL VERIFICATION")
print("="*55)
files = [
    "scaler.pkl", "linear_batch_gd.pkl", "linear_sgd.pkl",
    "linear_minibatch.pkl", "linear_momentum.pkl",
    "linear_rmsprop.pkl", "linear_adam.pkl", "neural_net_adam.pkl"
]
all_ok = True
for f in files:
    exists = os.path.exists(f"models/{f}")
    print(f"  [{'OK' if exists else 'MISSING'}] models/{f}")
    if not exists: all_ok = False
print("\n✓ Member 1 complete!" if all_ok else "\n✗ Some files missing.")