import numpy as np

class OptimizerEngine:
    def __init__(self, optimizer_type="batch", learning_rate=0.01, epochs=100):
        self.optimizer_type = optimizer_type
        self.lr = learning_rate
        self.epochs = epochs
        self.velocity = None
        self.m = None
        self.v = None
        self.t = None

    def update_params(self, w, grad, epoch):
        # Batch Gradient Descent
        if self.optimizer_type == "batch":
            w = w - self.lr * grad

        # Momentum Optimizer
        elif self.optimizer_type == "momentum":
            if self.velocity is None:
                self.velocity = np.zeros_like(w)
            beta = 0.95   # momentum factor (try 0.95 also)
            self.velocity = beta * self.velocity + self.lr * grad
            w = w - self.velocity

        # RMSProp placeholder
        elif self.optimizer_type == "rmsprop":
            pass

        # Adam placeholder
        elif self.optimizer_type == "adam":
            pass

        return w