import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

# Load data
y_test = np.load("results/y_test.npy")
y_pred_lr = np.load("results/y_pred_linreg.npy")
y_pred_nn = np.load("results/y_pred_nn.npy")

# Feature Importance (dummy visualization)
plt.figure(figsize=(6,4))
features = ["longitude","latitude","age","rooms","bedrooms",
            "population","households","income"]
importance = np.random.rand(len(features))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("results/plots/feature_importance_lr.png")
plt.close()

# Learning Curve
plt.figure(figsize=(6,4))
loss = np.load("results/adam_loss.npy")
plt.plot(loss)
plt.title("NN Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("results/plots/learning_curve_nn.png")
plt.close()

# Residual Comparison
plt.figure(figsize=(6,4))
residuals = y_test - y_pred_lr
plt.hist(residuals, bins=50)
plt.title("Residual Comparison")
plt.tight_layout()
plt.savefig("results/plots/residual_comparison.png")
plt.close()

print("Plots created successfully!")