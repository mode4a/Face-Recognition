import numpy as np


class Autoencoder:
    """
    Two-hidden-layer autoencoder:

        Encoder:  input_dim  ->  hidden_dim  ->  latent_dim   (ReLU)
        Decoder:  latent_dim ->  hidden_dim  ->  input_dim    (ReLU -> Sigmoid)

    The intermediate hidden layer gives the network enough capacity to learn
    useful face representations even with a small latent dimension and only
    ~200 training samples.

    Parameters
    ----------
    input_dim  : number of raw features (e.g. 10304 for 112×92 ORL faces)
    latent_dim : bottleneck size  (set this to the PCA k you found)
    hidden_dim : size of the intermediate encoder/decoder layer
                 defaults to max(256, 2 * latent_dim) if None
    lr         : initial learning rate for SGD
    lr_decay   : multiplicative decay applied every epoch  (1.0 = no decay)
    seed       : RNG seed for full reproducibility
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim=None,
        lr=1e-3,
        lr_decay=1.0,
        seed=42,
    ):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.lr         = lr
        self.lr_decay   = lr_decay

        # default hidden size: large enough to avoid a collapsed bottleneck
        if hidden_dim is None:
            hidden_dim = max(256, 2 * latent_dim)
        self.hidden_dim = hidden_dim

        self.rng = np.random.default_rng(seed)

        def he(fan_in, fan_out):
            """He / Kaiming normal init — correct for ReLU layers."""
            return self.rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

        # =====================================================================
        # Encoder  input_dim -> hidden_dim -> latent_dim
        # =====================================================================
        self.W1 = he(input_dim,  hidden_dim)   # (D, H)
        self.b1 = np.zeros((1,   hidden_dim))

        self.W2 = he(hidden_dim, latent_dim)   # (H, L)
        self.b2 = np.zeros((1,   latent_dim))

        # =====================================================================
        # Decoder  latent_dim -> hidden_dim -> input_dim
        # =====================================================================
        self.W3 = he(latent_dim, hidden_dim)   # (L, H)
        self.b3 = np.zeros((1,   hidden_dim))

        self.W4 = he(hidden_dim, input_dim)    # (H, D)
        self.b4 = np.zeros((1,   input_dim))

    # =========================================================================
    # ACTIVATIONS
    # =========================================================================
    def relu(self, x):
        return np.maximum(0.0, x)

    def relu_derivative(self, x):
        # preserve dtype so backprop stays in float64
        return (x > 0).astype(x.dtype)

    def sigmoid(self, x):
        # clamp to avoid exp overflow
        x = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1.0 - s)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================
    def forward(self, X):

        # ---------- Encoder ----------
        self.z1 = X  @ self.W1 + self.b1   # (N, H)
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2  # (N, L)
        self.a2 = self.relu(self.z2)            # latent representation

        # ---------- Decoder ----------
        self.z3 = self.a2 @ self.W3 + self.b3  # (N, H)
        self.a3 = self.relu(self.z3)

        self.z4   = self.a3 @ self.W4 + self.b4  # (N, D)
        self.X_hat = self.sigmoid(self.z4)

        return self.X_hat

    # =========================================================================
    # LOSS  (MSE)
    # =========================================================================
    def compute_loss(self, X, X_hat):
        return np.mean((X - X_hat) ** 2)

    # =========================================================================
    # BACKPROP
    # =========================================================================
    def backward(self, X, X_hat):

        m = X.shape[0]

        # --- output layer ---
        dL      = 2.0 * (X_hat - X) / m            # dMSE/dX_hat
        dZ4     = dL * self.sigmoid_derivative(self.z4)

        dW4     = self.a3.T @ dZ4
        db4     = np.sum(dZ4, axis=0, keepdims=True)

        # --- decoder hidden ---
        dA3     = dZ4 @ self.W4.T
        dZ3     = dA3 * self.relu_derivative(self.z3)

        dW3     = self.a2.T @ dZ3
        db3     = np.sum(dZ3, axis=0, keepdims=True)

        # --- latent layer ---
        dA2     = dZ3 @ self.W3.T
        dZ2     = dA2 * self.relu_derivative(self.z2)

        dW2     = self.a1.T @ dZ2
        db2     = np.sum(dZ2, axis=0, keepdims=True)

        # --- encoder hidden ---
        dA1     = dZ2 @ self.W2.T
        dZ1     = dA1 * self.relu_derivative(self.z1)

        dW1     = X.T @ dZ1
        db1     = np.sum(dZ1, axis=0, keepdims=True)

        # --- SGD update ---
        self.W4 -= self.lr * dW4;  self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1

    # =========================================================================
    # TRAINING
    # =========================================================================
    def fit(self, X, epochs=100, batch_size=32, verbose=True):

        if X.ndim != 2:
            raise ValueError("X must be 2D (samples, features)")

        n = X.shape[0]

        for epoch in range(epochs):

            # reproducible shuffle via seeded rng
            idx       = self.rng.permutation(n)
            X_shuffle = X[idx]

            total_loss = 0.0

            for i in range(0, n, batch_size):
                X_batch  = X_shuffle[i : i + batch_size]
                X_hat    = self.forward(X_batch)
                loss     = self.compute_loss(X_batch, X_hat)
                total_loss += loss * len(X_batch)
                self.backward(X_batch, X_hat)

            total_loss /= n

            # learning-rate decay
            self.lr *= self.lr_decay

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | Loss = {total_loss:.6f} | lr = {self.lr:.2e}")

    # =========================================================================
    # ENCODE  —  forward through encoder only, returns latent vectors
    # =========================================================================
    def encode(self, X):
        z1 = X  @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return self.relu(z2)            # shape: (N, latent_dim)

    # =========================================================================
    # DECODE  —  forward through decoder only, returns reconstructions
    # =========================================================================
    def decode(self, Z):
        z3 = Z  @ self.W3 + self.b3
        a3 = self.relu(z3)
        z4 = a3 @ self.W4 + self.b4
        return self.sigmoid(z4)         # shape: (N, input_dim)

    # =========================================================================
    # PREDICT  —  full forward pass (same as Keras .predict)
    # =========================================================================
    def predict(self, X):
        return self.forward(X)