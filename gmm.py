import numpy as np
from kmeans import KMeans

def logsumexp(a, axis=None, keepdims=False):
    """
    Numerically stable calculation of log(sum(exp(a))) via the log-sum-exp trick.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    if not keepdims:
        a_max_squeezed = np.squeeze(a_max, axis=axis)
    else:
        a_max_squeezed = a_max
    
    if axis is None:
        a_max_squeezed = np.max(a)
        a_max = a_max_squeezed
        
    out = a_max_squeezed + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    return out

class GMM:
    """
    Gaussian Mixture Model (GMM) Clustering.
    
    A from-scratch, highly optimized vectorized Expectation-Maximization (EM) 
    implementation supporting 'diagonal' and 'tied' covariance types.
    Initialized using your teammate's custom KMeans class.
    
    Parameters
    ----------
    n_components : int
        Number of mixture components / clusters.
    covariance_type : str
        String describing the type of covariance parameters to be used.
        Options: 'tied' (recommended for high accuracy and beating KMeans) or 'diagonal'.
    max_iters : int
        Maximum number of EM iterations to run.
    tol : float
        Convergence threshold. EM iterations will stop when the lower bound
        average gain is below this threshold.
    reg_covar : float
        Non-negative regularization added to the diagonal of covariance
        to ensure positive-definiteness and numerical stability.
    random_state : int
        Seed for reproducibility of initialization.
    """
    def __init__(
        self,
        n_components: int = 40,
        covariance_type: str = 'tied',
        max_iters: int = 150,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        
        # Parameters to fit (custom GMM)
        self.weights = None
        self.means = None
        self.covariances = None  # shape (K, D) for diagonal, or (D,) for tied shared diagonal

    def fit(self, X: np.ndarray):
        """
        Fit GMM on data X.
        """
        if self.covariance_type not in ['diagonal', 'tied']:
            raise ValueError("Custom GMM only supports 'diagonal' or 'tied' covariance types.")
        self._fit_custom(X)
        return self

    def _fit_custom(self, X: np.ndarray):
        """
        Custom EM implementation supporting diagonal and tied (shared diagonal) covariances.
        """
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # 1. Initialize parameters using your teammate's custom KMeans!
        print(f"[GMM Custom] Initializing means using teammate's KMeans (K={self.n_components}) ...")
        kmeans = KMeans(k=self.n_components, random_state=self.random_state)
        kmeans.fit(X)
        
        self.means = kmeans.centroids  # shape (K, D)
        train_clusters = kmeans.labels  # shape (N,)
        
        # Initialize weights
        _, counts = np.unique(train_clusters, return_counts=True)
        if len(counts) < self.n_components:
            padded_counts = np.zeros(self.n_components)
            for idx, c in enumerate(np.unique(train_clusters)):
                padded_counts[c] = counts[idx]
            counts = padded_counts + 1e-5
            
        self.weights = counts / np.sum(counts)  # shape (K,)
        
        # Initialize covariances
        if self.covariance_type == 'diagonal':
            self.covariances = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                cluster_points = X[train_clusters == k]
                if len(cluster_points) > 1:
                    self.covariances[k] = np.var(cluster_points, axis=0) + self.reg_covar
                else:
                    self.covariances[k] = np.var(X, axis=0) + self.reg_covar
        elif self.covariance_type == 'tied':
            # Shared diagonal covariance vector of the entire dataset
            self.covariances = np.var(X, axis=0) + self.reg_covar  # shape (D,)
                
        # EM Loop
        prev_log_likelihood = -float('inf')
        
        for iteration in range(self.max_iters):
            # --- E-Step ---
            responsibilities, log_likelihood = self._e_step(X)
            
            # Check convergence
            log_lik_change = log_likelihood - prev_log_likelihood
            print(f"[GMM Custom] Iteration {iteration + 1:3d} | Log-Likelihood = {log_likelihood:.6f} | Change = {log_lik_change:.6f}")
            
            if iteration > 0 and 0 <= log_lik_change < self.tol:
                print("[GMM Custom] Converged.")
                break
                
            prev_log_likelihood = log_likelihood
            
            # --- M-Step ---
            self._m_step(X, responsibilities)

    def _e_step(self, X: np.ndarray):
        """
        E-step: Compute responsibilities and average log-likelihood.
        """
        n_samples = X.shape[0]
        diff = X[:, np.newaxis, :] - self.means[np.newaxis, :, :]  # shape (N, K, D)
        
        if self.covariance_type == 'diagonal':
            exponent_term = -0.5 * np.sum((diff ** 2) / self.covariances[np.newaxis, :, :], axis=2)  # shape (N, K)
            determinant_term = -0.5 * np.sum(np.log(2.0 * np.pi * self.covariances), axis=1)  # shape (K,)
            log_pdfs = exponent_term + determinant_term[np.newaxis, :]  # shape (N, K)
        elif self.covariance_type == 'tied':
            # self.covariances has shape (D,)
            exponent_term = -0.5 * np.sum((diff ** 2) / self.covariances[np.newaxis, np.newaxis, :], axis=2)  # shape (N, K)
            determinant_term = -0.5 * np.sum(np.log(2.0 * np.pi * self.covariances))  # scalar
            log_pdfs = exponent_term + determinant_term  # shape (N, K)
            
        weighted_log_pdfs = log_pdfs + np.log(self.weights + 1e-15)[np.newaxis, :]  # shape (N, K)
        log_likelihoods = logsumexp(weighted_log_pdfs, axis=1, keepdims=True)  # shape (N, 1)
        responsibilities = np.exp(weighted_log_pdfs - log_likelihoods)  # shape (N, K)
        avg_log_likelihood = np.mean(log_likelihoods)
        
        return responsibilities, avg_log_likelihood

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        M-step: Update weights, means, and covariances.
        """
        n_samples = X.shape[0]
        Nk = np.sum(responsibilities, axis=0) + 1e-15  # shape (K,)
        
        # Update weights and means
        self.weights = Nk / n_samples
        self.means = (responsibilities.T @ X) / Nk[:, np.newaxis]  # shape (K, D)
        
        # Update covariances
        diff = X[:, np.newaxis, :] - self.means[np.newaxis, :, :]  # shape (N, K, D)
        
        if self.covariance_type == 'diagonal':
            self.covariances = np.sum(responsibilities[:, :, np.newaxis] * (diff ** 2), axis=0) / Nk[:, np.newaxis] + self.reg_covar
        elif self.covariance_type == 'tied':
            # Calculate pooled shared diagonal covariance vector across all clusters
            pooled_var = np.sum(responsibilities[:, :, np.newaxis] * (diff ** 2), axis=(0, 1)) / n_samples
            self.covariances = pooled_var + self.reg_covar

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster index for each sample in X.
        """
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model on X and return cluster indices.
        """
        self.fit(X)
        return self.predict(X)
