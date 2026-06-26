"""
test_model.py — quick sanity check that linear_adam.pkl loads and predicts correctly.
Run from the project root: python test_model.py
"""
import joblib
import numpy as np
import __main__ as _main

# Must define these BEFORE loading, exactly like app.py does —
# joblib saved them as __main__.SavedModel, so we re-inject the class here too.
class SavedModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b
    def predict(self, X):
        return np.array(X) @ self.w + self.b

_main.SavedModel = SavedModel

model = joblib.load("models/linear_adam.pkl")

print("Expected feature count:", model.w.shape[0])   # should print 11

# 11 zeros, not 12 — the model was trained on 11 features
X = np.zeros((1, 11))
prediction_scaled = model.predict(X)
print("Raw scaled prediction:", prediction_scaled)

# unscale to real dollars (same constants used in app.py)
Y_MEAN, Y_STD = 206855.0, 115395.0
price = prediction_scaled[0] * Y_STD + Y_MEAN
print(f"Predicted price: ${price:,.0f}")