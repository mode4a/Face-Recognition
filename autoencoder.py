import numpy as np


class Autoencoder:

    def __init__(self, input_dim, latent_dim, lr=1e-3, seed=42):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.lr         = lr
        self.rng        = np.random.default_rng(seed)

        # Glorot uniform init
        def glorot(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self.rng.uniform(-limit, limit, (fan_in, fan_out))

        self.W1 = glorot(input_dim, latent_dim)
        self.b1 = np.zeros((1, latent_dim))

        self.W2 = glorot(latent_dim, input_dim)
        self.b2 = np.zeros((1, input_dim))

    # ---------------- Activations ---------------- #

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(x.dtype)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_grad(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)

    # ---------------- Forward ---------------- #

    def forward(self, X):
        self.z1  = X @ self.W1 + self.b1
        self.a1  = self._relu(self.z1)

        self.z2  = self.a1 @ self.W2 + self.b2
        self.out = self._sigmoid(self.z2)

        return self.out

    # ---------------- Backward (SGD) ---------------- #

    def backward(self, X, out):
        n = X.shape[0]

        # Loss gradient (MSE)
        dout = 2.0 * (out - X) / n

        # Decoder gradients
        dz2 = dout * self._sigmoid_grad(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        # Encoder gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_grad(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # ---------------- SGD UPDATE ---------------- #
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ---------------- Training ---------------- #

    def fit(self, X, epochs=100, batch_size=32, verbose=True):
        n = X.shape[0]

        for epoch in range(epochs):
            idx = self.rng.permutation(n)
            X = X[idx]

            total_loss = 0.0

            for i in range(0, n, batch_size):
                Xb = X[i:i + batch_size]

                out = self.forward(Xb)
                total_loss += np.mean((Xb - out) ** 2) * len(Xb)

                self.backward(Xb, out)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | Loss = {total_loss / n:.6f}")

    # ---------------- Utilities ---------------- #

    def encode(self, X):
        z1 = X @ self.W1 + self.b1
        return self._relu(z1)

    def decode(self, Z):
        z2 = Z @ self.W2 + self.b2
        return self._sigmoid(z2)

    def predict(self, X):
        return self.forward(X)