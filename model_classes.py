import numpy as np

def relu(x):
    return np.maximum(0, x)

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
        X = np.array(X)
        a1 = relu(X  @ self.W1 + self.b1)
        a2 = relu(a1 @ self.W2 + self.b2)
        return (a2 @ self.W3 + self.b3).flatten()