import numpy as np

class OptimizerEngine:
    def __init__(self, optimizer_type="batch", learning_rate=0.01, epochs=100):
        self.optimizer_type = optimizer_type
        self.lr = learning_rate
        self.epochs = epochs
        # state variables for optimizers
        self.v = None
        self.m = None
        self.t = 0

    def update_params(self, w, grad, epoch):
        """Update weights based on optimizer type"""
        if self.optimizer_type == "batch":
            w = w - self.lr * grad

        elif self.optimizer_type == "momentum":
            if self.v is None:
                self.v = np.zeros_like(w)
            beta = 0.9
            self.v = beta * self.v + (1 - beta) * grad
            w = w - self.lr * self.v

        elif self.optimizer_type == "rmsprop":
            if self.v is None:
                self.v = np.zeros_like(w)
            beta = 0.9
            eps = 1e-8
            self.v = beta * self.v + (1 - beta) * (grad ** 2)
            w = w - (self.lr / (np.sqrt(self.v) + eps)) * grad

        elif self.optimizer_type == "adam":
            if self.m is None:
                self.m = np.zeros_like(w)
                self.v = np.zeros_like(w)
                self.t = 0
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self.t += 1
            self.m = beta1 * self.m + (1 - beta1) * grad
            self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            w = w - (self.lr / (np.sqrt(v_hat) + eps)) * m_hat

        return w
