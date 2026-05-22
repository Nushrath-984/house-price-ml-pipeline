import numpy as np

class OptimizerEngine:
    def __init__(self, optimizer_type="batch", learning_rate=0.01, epochs=100):
        self.optimizer_type = optimizer_type
        self.lr = learning_rate
        self.epochs = epochs
        # placeholders for other optimizers
        self.velocity = None   # Momentum
        self.s = None          # RMSProp
        self.m = None          # Adam
        self.v = None          # Adam

    def update_params(self, w, grad, epoch):
        if self.optimizer_type == "batch":
            # baseline Batch GD
            w = w - self.lr * grad
        elif self.optimizer_type == "momentum":
            pass
        elif self.optimizer_type == "rmsprop":
            pass
        elif self.optimizer_type == "adam":
            pass
        return w