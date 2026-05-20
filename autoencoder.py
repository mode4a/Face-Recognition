import numpy as np

class Autoencoder:

    def __init__(self, input_dim, latent_dim, lr=1e-3, seed=42):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.lr         = lr
        self.rng        = np.random.default_rng(seed)

        def glorot(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self.rng.uniform(-limit, limit, (fan_in, fan_out))

        self.W1 = glorot(input_dim, latent_dim)
        self.b1 = np.zeros((1, latent_dim))

        self.W2 = glorot(latent_dim, input_dim)
        self.b2 = np.zeros((1, input_dim))


    def forward(self, X):
        self.z1  = X @ self.W1 + self.b1
        self.z2  = self.z1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, out):
        n = X.shape[0]

        # MSE loss gradient
        dout = 2.0 * (out - X) / n

        # Decoder gradients
        dW2 = self.z1.T @ dout
        db2 = np.sum(dout, axis=0, keepdims=True)

        # Encoder gradients
        dz1 = dout @ self.W2.T
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # SGD updates
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


    def fit(self, X, epochs=100, batch_size=32, verbose=True):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

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

    # ---------------- Encode / Decode ---------------- #

    def encode(self, X):
        X = X - self.mean
        return X @ self.W1 + self.b1

    def decode(self, Z):
        return Z @ self.W2 + self.b2 + self.mean

    def predict(self, X):
        return self.forward(X)