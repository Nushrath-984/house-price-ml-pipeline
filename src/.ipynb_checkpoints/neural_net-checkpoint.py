import numpy as np


class NeuralNetworkScratch:
    """
    Feed-Forward Neural Network for regression — built with NumPy only.
    Architecture: input → 64 (ReLU) → 32 (ReLU) → 1 (linear output)

    Designed for Week 5. Compatible with:
      - Week 1: final_prepared.csv (preprocessed data)
      - Week 2-4: optimizers.py (via get_params / set_params)
      - Member 2: cache from forward() used in backpropagation
      - Member 3: training loop uses get_params() and set_params()
    """

    def __init__(self, input_size, hidden1=64, hidden2=32, output_size=1):
        np.random.seed(42)  # ensures reproducibility across all members

        # Layer 1: input → 64
        self.W1 = np.random.randn(input_size, hidden1) * 0.01
        self.b1 = np.zeros((1, hidden1))

        # Layer 2: 64 → 32
        self.W2 = np.random.randn(hidden1, hidden2) * 0.01
        self.b2 = np.zeros((1, hidden2))

        # Output layer: 32 → 1
        self.W3 = np.random.randn(hidden2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------

    def relu(self, Z):
        """ReLU activation: f(Z) = max(0, Z)"""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """
        Derivative of ReLU — required by Member 2 for backpropagation.
        Returns 1 where Z > 0, else 0.
        """
        return (Z > 0).astype(float)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X):
        """
        Forward pass through all layers.

        Args:
            X : numpy array of shape (n_samples, input_size)
                Use X from final_prepared.csv after dropping target column.

        Returns:
            output : predictions, shape (n_samples, 1)
            cache  : tuple (X, Z1, A1, Z2, A2, Z3)
                     Member 2 uses this cache for backpropagation.
        """
        # Layer 1: linear transformation + ReLU
        Z1 = np.dot(X, self.W1) + self.b1   # shape: (n, 64)
        A1 = self.relu(Z1)                   # shape: (n, 64)

        # Layer 2: linear transformation + ReLU
        Z2 = np.dot(A1, self.W2) + self.b2  # shape: (n, 32)
        A2 = self.relu(Z2)                   # shape: (n, 32)

        # Output layer: linear only — no activation for regression
        Z3 = np.dot(A2, self.W3) + self.b3  # shape: (n, 1)

        # Cache stores all intermediate values for backpropagation
        cache = (X, Z1, A1, Z2, A2, Z3)

        return Z3, cache

    # ------------------------------------------------------------------
    # Parameter interface — used by Member 3's training loop
    # ------------------------------------------------------------------

    def get_params(self):
        """
        Returns all weights and biases as a dictionary.
        Member 3's optimizer reads these, updates them, then calls set_params().
        """
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }

    def set_params(self, params):
        """
        Replaces weights and biases with updated values from optimizer.
        Member 3 calls this after every gradient update step.
        """
        self.W1 = params['W1']
        self.b1 = params['b1']
        self.W2 = params['W2']
        self.b2 = params['b2']
        self.W3 = params['W3']
        self.b3 = params['b3']
    def backward(self, cache, y):
        """
        Computes gradients for all weights and biases.
        Uses chain rule going backwards: output -> Layer2 -> Layer1
        """
        X, Z1, A1, Z2, A2, Z3 = cache
        n = X.shape[0]

        # Step 1: Output layer gradient
        dZ3 = (2 / n) * (Z3 - y)
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        dA2 = dZ3 @ self.W3.T

        # Step 2: Layer 2 gradient
        dZ2 = dA2 * self.relu_derivative(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T

        # Step 3: Layer 1 gradient
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3
        }
        return grads
