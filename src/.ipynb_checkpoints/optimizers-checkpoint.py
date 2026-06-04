import numpy as np

class OptimizerEngine:
<<<<<<< HEAD
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
=======
    def __init__(self, optimizer_type="batch", learning_rate=0.01, epochs=100, batch_size=None):
        self.optimizer_type = optimizer_type
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # State variables for advanced optimizers
        self.velocity = None   # Momentum
        self.s = None          # RMSProp
        self.m = None          # Adam
        self.v = None          # Adam
        self.t = 0             # timestep for Adam

    def update_params(self, w, grad, epoch):
        # --- Batch Gradient Descent ---
        if self.optimizer_type == "batch":
            w = w - self.lr * grad

        # --- Stochastic Gradient Descent (SGD) ---
        elif self.optimizer_type == "sgd":
            # SGD is just batch update with one sample at a time
            w = w - self.lr * grad

        # --- Mini-Batch Gradient Descent ---
        elif self.optimizer_type == "mini-batch":
            # Same as batch, but grads are computed on a subset of data
            w = w - self.lr * grad

        # --- Momentum ---
        elif self.optimizer_type == "momentum":
            beta = 0.9
            if self.velocity is None or self.velocity.shape != w.shape:
                self.velocity = np.zeros_like(w)
            self.velocity = beta * self.velocity + self.lr * grad
            w = w - self.velocity

        # --- RMSProp ---
        elif self.optimizer_type == "rmsprop":
            beta, eps = 0.9, 1e-8
            if self.s is None or self.s.shape != w.shape:
                self.s = np.zeros_like(w)
            self.s = beta * self.s + (1 - beta) * (grad ** 2)
            w = w - (self.lr / (np.sqrt(self.s) + eps)) * grad

        # --- Adam ---
        elif self.optimizer_type == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            if self.m is None or self.m.shape != w.shape:
                self.m = np.zeros_like(w)
                self.v = np.zeros_like(w)
                self.t = 0
            self.t += 1
            self.m = beta1 * self.m + (1 - beta1) * grad
            self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            w = w - (self.lr / (np.sqrt(v_hat) + eps)) * m_hat

        return w
      
>>>>>>> 362ec0da8f09ced70adf0f702a9b16c922ecc7fc
