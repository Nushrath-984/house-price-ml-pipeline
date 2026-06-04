import numpy as np

def compute_metrics(y_true, y_pred, n_features):
    n = len(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2, Adj_R2=adj_r2)