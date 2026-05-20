import numpy as np
class KMeans:
    def __init__(
        self,
        k: int = 40,
        max_iters: int = 100,
        tol: float = 1e-10,
        random_state: int = 42,
    ):

        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _initialize_centroids(self, X: np.ndarray):

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X.shape[0]

        centroids = []

        first_index = np.random.randint(n_samples)
        centroids.append(X[first_index])

        for _ in range(1, self.k):
            distances = []
            for x in X:
                min_dist = float("inf")

                for c in centroids:
                    dist = np.linalg.norm(x - c) ** 2
                    min_dist = min(min_dist, dist)

                distances.append(min_dist)

            distances = np.array(distances)

            total = np.sum(distances)
            if total == 0:
                probabilities = np.ones(n_samples) / n_samples
            else:
                probabilities = distances / total

            next_index = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_index])

        return np.array(centroids)

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray):

        n_samples = X.shape[0]
        k = centroids.shape[0]

        distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                diff = X[i] - centroids[j]
                distances[i, j] = np.linalg.norm(diff)

        return distances

    def _assign_clusters(self, distances: np.ndarray):

        n_samples = distances.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):

            closest_centroid = np.argmin(distances[i])
            labels[i] = closest_centroid

        return labels

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ):

        n_features = X.shape[1]
        new_centroids = np.zeros((self.k, n_features))
        for cluster_id in range(self.k):

            cluster_points = X[labels == cluster_id]

            if len(cluster_points) == 0:
                random_idx = np.random.randint(0, X.shape[0])
                new_centroids[cluster_id] = X[random_idx]

            else:
                new_centroids[cluster_id] = cluster_points.mean(axis=0)

        return new_centroids

    def _compute_inertia(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray
    ):
        """
        Compute Within-Cluster Sum of Squares (WCSS).

        Inertia =
            sum of squared distances between points
            and their assigned centroid.
        """

        total = 0.0

        for cluster_id in range(self.k):

            cluster_points = X[labels == cluster_id]

            if len(cluster_points) == 0:
                continue

            distances = np.linalg.norm(
                cluster_points - centroids[cluster_id],
                axis=1
            )

            total += np.sum(distances ** 2)

        return total

    def fit(self, X: np.ndarray):

        centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iters):

            distances = self._compute_distances(X,centroids)

            labels = self._assign_clusters(distances)

            new_centroids = self._update_centroids(X,labels)

            centroid_shift = np.linalg.norm(new_centroids - centroids)

            centroids = new_centroids

            print(
                f"[KMeans] Iteration "
                f"{iteration + 1:3d} | "
                f"Centroid shift = {centroid_shift:.6f}"
            )

            if centroid_shift < self.tol:
                print("[KMeans] Converged.")
                break

        self.centroids = centroids
        self.labels = labels

        self.inertia = self._compute_inertia(
            X,
            labels,
            centroids
        )

        return self

    def predict(self, X: np.ndarray):
        distances = self._compute_distances(
            X,
            self.centroids
        )

        return self._assign_clusters(distances)

    def fit_predict(self, X: np.ndarray):

        self.fit(X)
        return self.labels