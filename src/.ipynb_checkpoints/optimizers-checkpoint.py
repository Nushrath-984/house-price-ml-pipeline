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
    # Batch Gradient Descent
    if self.optimizer_type == "batch":
        w = w - self.lr * grad

    # Momentum
    elif self.optimizer_type == "momentum":
        if self.v is None:
            self.v = np.zeros_like(w)
        beta = 0.9
        self.v = beta * self.v + (1 - beta) * grad
        w = w - self.lr * self.v

    # RMSProp
    elif self.optimizer_type == "rmsprop":
        if self.v is None:
            self.v = np.zeros_like(w)
        beta = 0.9
        eps = 1e-8
        self.v = beta * self.v + (1 - beta) * (grad ** 2)
        w = w - (self.lr / (np.sqrt(self.v) + eps)) * grad
        # Adam placeholder
        elif self.optimizer_type == "adam":
            pass

        return w