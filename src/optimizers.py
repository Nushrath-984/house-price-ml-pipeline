import numpy as np

class OptimizerEngine:
    def __init__(self, optimizer_type="batch", learning_rate=0.01, epochs=100):
        self.optimizer_type = optimizer_type
        self.lr = learning_rate
        self.epochs = epochs
        self.velocity = None   # for Momentum
        self.s = None          # for RMSProp
        self.m = None          # for Adam
        self.v = None          # for Adam
        self.t = 0             # timestep for Adam

    def update_params(self, w, grad, epoch):
        # Member 1 – Batch GD
        if self.optimizer_type == "batch":
            w = w - self.lr * grad

        # Member 2 – Momentum
        elif self.optimizer_type == "momentum":
            if self.velocity is None:
                self.velocity = np.zeros_like(w)
            beta = 0.9   # or 0.95 for second run
            self.velocity = beta * self.velocity + self.lr * grad
            w = w - self.velocity

        # Member 3 – RMSProp
        elif self.optimizer_type == "rmsprop":
            if self.s is None:
                self.s = np.zeros_like(w)
            beta = 0.9
            eps = 1e-8
            self.s = beta * self.s + (1 - beta) * (grad ** 2)
            w = w - (self.lr / (np.sqrt(self.s) + eps)) * grad

        # Member 4 – Adam
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
