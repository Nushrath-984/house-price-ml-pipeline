import joblib
import os

MODEL_PATHS = {
    "Batch GD":              "models/linear_batch_gd.pkl",
    "SGD":                   "models/linear_sgd.pkl",
    "Mini-Batch GD":         "models/linear_minibatch.pkl",
    "Momentum":              "models/linear_momentum.pkl",
    "RMSProp":               "models/linear_rmsprop.pkl",
    "Adam (Linear)":         "models/linear_adam.pkl",
    "Neural Network (Adam)": "models/neural_net_adam.pkl",
}

def load_model(name):
    path = MODEL_PATHS.get(name)
    if not path:
        raise ValueError(f"Unknown model: {name}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    return joblib.load(path)

def load_scaler():
    return joblib.load("models/scaler.pkl")

def get_model_names():
    return list(MODEL_PATHS.keys())
