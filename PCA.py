import os
import numpy as np
import matplotlib.pyplot as plt

ALPHAS_DEFAULT = [0.80, 0.85, 0.90, 0.95]
CACHE_DIR_DEFAULT = "./pca_cache"

def run_pca(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    alphas:  list = None,
    cache_dir: str = CACHE_DIR_DEFAULT,
    n_faces_to_visualise: int = 5,
    show_plots: bool = True,
):
    if alphas is None:
        alphas = ALPHAS_DEFAULT

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = get_eigen(X_train, cache_dir)

    # Compute the mean face to center the data
    # The mean is computed from the TRAINING set only.
    # It is also saved to cache so downstream files (k_means.py, gmm.py) can centre test data in the same way.
    mean_face = get_mean_face(X_train, cache_dir)

    # Centre the data by subtracting the mean face from every row
    X_train_centred = X_train - mean_face   # shape (n_train, n_features)
    X_test_centred  = X_test  - mean_face   # shape (n_test,  n_features)

    #For each alpha, select k components and project
    results = {}

    print("\n" + "="*60)
    print("PCA — Variance Retention Summary")
    print("="*60)
    print(f"{'Alpha':>8}  {'k (components)':>15}  {'Variance retained':>18}")
    print("-"*60)

    for alpha in alphas:
        k, variance_retained = select_components(eigenvalues, alpha)

        # Keep only the top-k eigenvectors (columns of `eigenvectors`)
        # eigenvectors is shape (n_features, n_features) with columns
        # sorted by descending eigenvalue.
        components = eigenvectors[:, :k] # shape (10304, k)

        # Project: multiply centred data by the component matrix. according to the formula:
        # z = Wᵀ x_c  where W is the matrix of top-k
        # Each row is projected from 10304-D space to k-D PCA space.
        X_train_pca = X_train_centred @ components   # (n_train, k)
        X_test_pca  = X_test_centred  @ components   # (n_test,  k)

        print(f"{alpha:>8.2f}  {k:>15d}  {variance_retained:>17.4f}%")

        results[alpha] = {
            "k"           : k,
            "X_train_pca" : X_train_pca,
            "X_test_pca"  : X_test_pca,
            "mean_face"   : mean_face,
            "components"  : components,
            "eigenvalues" : eigenvalues,
        }

    print("="*60)

    if show_plots or True:   # always save; show only when requested
        visualise_eigenfaces(
            eigenvectors, cache_dir, n_show=8, show=show_plots
        )
        visualise_reconstructions(
            X_train, X_train_centred, mean_face, results,
            n_show=n_faces_to_visualise, cache_dir=cache_dir, show=show_plots
        )

    return results


def get_eigen(X_train: np.ndarray, cache_dir: str):
    """
    Return the eigenvalues and eigenvectors of the covariance matrix
    computed from X_train.

    """
    os.makedirs(cache_dir, exist_ok=True)

    eigenvalues_path  = os.path.join(cache_dir, "eigenvalues.npy")
    eigenvectors_path = os.path.join(cache_dir, "eigenvectors.npy")

    if os.path.isfile(eigenvalues_path) and os.path.isfile(eigenvectors_path):
        print(f"[pca] Cache found in '{cache_dir}'. "
              "Loading eigenvalues and eigenvectors from disk …")
        eigenvalues  = np.load(eigenvalues_path)
        eigenvectors = np.load(eigenvectors_path)
        print(f"[pca] Loaded eigenvalues  shape : {eigenvalues.shape}")
        print(f"[pca] Loaded eigenvectors shape : {eigenvectors.shape}")
        return eigenvalues, eigenvectors

    #compute the eigen-decomposition and cache the results
    print("[pca] No cache found. Computing eigen-decomposition …")
    print("      (This may take a moment — it only happens once.)")

    eigenvalues, eigenvectors = compute_eigenvectors(X_train)

    # Save to disk
    np.save(eigenvalues_path,  eigenvalues)
    np.save(eigenvectors_path, eigenvectors)
    print(f"[pca] Eigenvalues  saved → {eigenvalues_path}")
    print(f"[pca] Eigenvectors saved → {eigenvectors_path}")

    return eigenvalues, eigenvectors


def compute_eigenvectors(X_train: np.ndarray):
    """
    Compute eigenvectors of the covariance matrix using the
    **compact / dual covariance trick**.

    Why the compact trick?
--
    The covariance matrix of X_train is (n_features × n_features),
    i.e. 10304 × 10304 for ORL faces.  Computing all eigenvectors of
    such a large matrix is very slow.

    However when n_train ≪ n_features (here 200 ≪ 10304), the
    covariance matrix has at most n_train non-zero eigenvalues.

    Compact trick steps
---
    Let A = centred X_train (shape n_train × n_features).

    The full covariance matrix is:
        C_full = (1/n) * Aᵀ A   (shape n_features × n_features)

    The compact covariance matrix is:
        C_compact = (1/n) * A Aᵀ   (shape n_train × n_train)

    If v is an eigenvector of C_compact with eigenvalue λ, then
        u = Aᵀ v / ‖Aᵀ v‖
    is an eigenvector of C_full with the SAME eigenvalue λ.

    This reduces the eigen-decomposition from an (n_features × n_features)
    problem to an (n_train × n_train) problem — 10304² → 200².

    Parameters
--
    X_train : np.ndarray, shape (n_train, n_features)
        Raw (un-centred) training data.

    Returns
---
    eigenvalues  : np.ndarray, shape (n_features,)  — sorted descending
    eigenvectors : np.ndarray, shape (n_features, n_features)
                   Columns are principal directions, sorted by descending
                   eigenvalue.  Columns beyond index n_train-1 correspond
                   to zero eigenvalues (null space directions).
    """

    n_train, n_features = X_train.shape

    # Centre the training data
    mean_face         = X_train.mean(axis=0)          # shape (n_features,)
    A                 = X_train - mean_face            # shape (n_train, n_features)

    # Compact covariance matrix (n_train × n_train)
    C_compact = (A @ A.T) / n_train                   # shape (n_train, n_train)

    # Eigen-decomposition of the small matrix
    # np.linalg.eigh assumes a symmetric matrix and returns real values;
    # eigenvalues are in ASCENDING order.
    eigenvalues_small, eigenvectors_small = np.linalg.eigh(C_compact)

    # Reverse to DESCENDING order (largest variance first)
    eigenvalues_small  = eigenvalues_small[::-1]
    eigenvectors_small = eigenvectors_small[:, ::-1]

    # Clip tiny negative eigenvalues caused by floating-point arithmetic
    eigenvalues_small = np.maximum(eigenvalues_small, 0)

    # Map compact eigenvectors back to the original feature space
    # u_i = Aᵀ v_i  (un-normalised),  shape (n_features,)
    eigenvectors_full = A.T @ eigenvectors_small       # (n_features, n_train)

    # Normalise each column to unit length
    norms = np.linalg.norm(eigenvectors_full, axis=0, keepdims=True)
    # Avoid division by zero for zero-norm vectors (null-space)
    norms = np.where(norms == 0, 1, norms)
    eigenvectors_full = eigenvectors_full / norms       # (n_features, n_train)

    # Pad eigenvalues and eigenvectors to length n_features
    # (the remaining n_features - n_train eigenvalues are 0)
    n_pad = n_features - n_train
    eigenvalues_padded = np.concatenate(
        [eigenvalues_small, np.zeros(n_pad)]
    )                                                   # shape (n_features,)

    # Random orthogonal padding for eigenvectors (they are unused in
    # practice because they correspond to zero eigenvalues; we add them
    # only to keep the shape consistent with (n_features, n_features)).
    padding = np.zeros((n_features, n_pad))
    eigenvectors_padded = np.concatenate(
        [eigenvectors_full, padding], axis=1
    )                                                   # shape (n_features, n_features)

    print(f"[pca] Eigen-decomposition complete.")
    print(f"      Training samples : {n_train}")
    print(f"      Feature space dim: {n_features}")
    print(f"      Non-zero eigenvalues: {np.sum(eigenvalues_padded > 1e-10)}")

    return eigenvalues_padded, eigenvectors_padded

def get_mean_face(X_train: np.ndarray, cache_dir: str) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    mean_face_path = os.path.join(cache_dir, "mean_face.npy")

    if os.path.isfile(mean_face_path):
        # Load from cache (consistent with the eigenvalue cache)
        mean_face = np.load(mean_face_path)
        print(f"[pca] Mean face loaded from cache: {mean_face_path}")
    else:
        mean_face = X_train.mean(axis=0)   # shape (n_features,)
        np.save(mean_face_path, mean_face)
        print(f"[pca] Mean face saved to cache: {mean_face_path}")

    return mean_face

def select_components(eigenvalues: np.ndarray, alpha: float):
    """
    Find the smallest k such that the top-k eigenvalues retain at least α * 100 % of the total variance.

    Intuition
-
    Each eigenvalue λ_i represents the variance explained by the i-th
    principal component.  The cumulative explained variance ratio after
    k components is:

        Σ(from i=1 to k) λ_i  /  Σ(from i=1 to n) λ_i  ≥  α

    """
    total_variance = eigenvalues.sum()

    if total_variance == 0:
        raise ValueError("All eigenvalues are zero — cannot perform PCA.")

    # Cumulative explained variance ratio for k = 1, 2, …, n
    cumulative_ratio = np.cumsum(eigenvalues) / total_variance

    # Find the first k where the cumulative ratio reaches alpha
    # np.searchsorted returns the index at which `alpha` would be inserted
    # to keep `cumulative_ratio` sorted → that is exactly the first index
    # where cumulative_ratio[index] >= alpha.
    k = int(np.searchsorted(cumulative_ratio, alpha)) + 1

    # Guard against over-shooting the array length
    k = min(k, len(eigenvalues))

    variance_retained_pct = cumulative_ratio[k - 1] * 100

    return k, variance_retained_pct

def visualise_eigenfaces(
    eigenvectors: np.ndarray,
    cache_dir:    str,
    n_show:       int  = 8,
    img_shape:    tuple = (112, 92),   # ORL image is H × W
    show:         bool  = True,
):
    """
    Display the first `n_show` eigenvectors reshaped as face images
    (often called "eigenfaces").

    Eigenfaces are the principal directions in pixel space.  Visualising
    them helps verify that the PCA decomposition makes intuitive sense
    — the top eigenfaces should capture coarse face structure (lighting,
    overall shape) while later ones capture finer details.

"""

    n_show = min(n_show, eigenvectors.shape[1])

    fig, axes = plt.subplots(1, n_show, figsize=(2 * n_show, 3))
    fig.suptitle("Top Eigenfaces (Principal Directions)", fontsize=13)

    for i, ax in enumerate(axes):
        # Extract i-th eigenvector (a column) and reshape to image
        eigenface = eigenvectors[:, i].reshape(img_shape)

        # Normalise to [0, 1] for display
        emin, emax = eigenface.min(), eigenface.max()
        if emax > emin:
            eigenface = (eigenface - emin) / (emax - emin)

        ax.imshow(eigenface, cmap="gray")
        ax.set_title(f"PC {i+1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(cache_dir, "eigenfaces.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[pca] Eigenfaces plot saved → {save_path}")
    if show:
        plt.show()
    plt.close()


def visualise_reconstructions(
    X_train:          np.ndarray,
    X_train_centred:  np.ndarray,
    mean_face:        np.ndarray,
    results:          dict,
    n_show:           int   = 5,
    img_shape:        tuple = (112, 92),
    cache_dir:        str   = CACHE_DIR_DEFAULT,
    show:             bool  = True,
):
    """
    For each alpha, reconstruct `n_show` training faces from the
    PCA subspace and display them alongside the originals.
    Original formula is : X(reduced) = Ut @ U @ (X-Mean)
    Reconstruction formula
--
    Given a centred image vector x_c = x - μ, its PCA reconstruction is:

        X = U @ Ut @ X(reduced)      where U = eigenvectors[:, :k]

    Adding the mean face back:

        X(original) = X + μ

    Comparing originals to reconstructions lets us see how much detail
    is lost / retained at each variance threshold.

"""

    n_show = min(n_show, X_train.shape[0])
    alphas = sorted(results.keys())

    # Layout: rows = original + one row per alpha, columns = n_show faces
    n_rows = 1 + len(alphas)
    fig, axes = plt.subplots(
        n_rows, n_show,
        figsize=(2.5 * n_show, 2.5 * n_rows)
    )
    fig.suptitle(
        "Original vs PCA Reconstructions at Different α Values",
        fontsize=12, y=1.01
    )

    # Row 0 — original faces
    for col in range(n_show):
        ax = axes[0, col] if n_rows > 1 else axes[col]
        ax.imshow(X_train[col].reshape(img_shape), cmap="gray")
        ax.set_title(f"Original {col+1}", fontsize=8)
        ax.axis("off")

    # One row per alpha — reconstructed faces
    for row_idx, alpha in enumerate(alphas, start=1):
        k          = results[alpha]["k"]
        components = results[alpha]["components"]   # (n_features, k)

        for col in range(n_show):
            x_c = X_train_centred[col]              # (n_features,)

            # Project into PCA subspace then reconstruct back
            # projection: z  = W_kᵀ x_c   →  shape (k,)
            # reconstruction: x̂_c = W_k z  →  shape (n_features,)
            z     = components.T @ x_c              # (k,)
            x_hat = components @ z + mean_face       # (n_features,)

            ax = axes[row_idx, col] if n_rows > 1 else axes[col]
            ax.imshow(x_hat.reshape(img_shape), cmap="gray")
            ax.set_title(f"α={alpha}\nk={k}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(cache_dir, "reconstructions.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[pca] Reconstruction plot saved → {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_explained_variance(eigenvalues: np.ndarray, cache_dir: str = CACHE_DIR_DEFAULT, show: bool = True):
    """
    Plot the cumulative explained variance against the number of
    principal components.  Draws horizontal lines at α ∈ {0.80, 0.85,
    0.90, 0.95} to make the chosen k values easy to read off.

    """

    total = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar * 100,
            color="steelblue", linewidth=1.5, label="Cumulative variance")

    for alpha in ALPHAS_DEFAULT:
        k, _ = select_components(eigenvalues, alpha)
        ax.axhline(alpha * 100, linestyle="--", linewidth=0.9,
                   label=f"α={alpha:.0%}  →  k={k}")
        ax.axvline(k, linestyle=":", linewidth=0.9, color="grey")

    ax.set_xlabel("Number of principal components (k)", fontsize=11)
    ax.set_ylabel("Cumulative explained variance (%)", fontsize=11)
    ax.set_title("PCA — Explained Variance vs Number of Components", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 250)   # zoom in on the interesting region

    plt.tight_layout()
    save_path = os.path.join(cache_dir, "explained_variance.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[pca] Explained-variance plot saved → {save_path}")
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    import sys
    import utilities
    sys.path.insert(0, ".")
    from utilities.vectorization  import load_dataset
    from utilities.splitData import split_dataset

    if len(sys.argv) < 2:
        print("Usage: python pca.py <path_to_dataset_root>")
        sys.exit(1)

    dataset_root = sys.argv[1]
    
    D, y                               = load_dataset(dataset_root)
    X_train, X_test, y_train, y_test   = split_dataset(D, y)

    # Run PCA
    results = run_pca(X_train, X_test, show_plots=True)

    # Plot explained variance curve
    eigenvalues = results[0.95]["eigenvalues"]
    plot_explained_variance(eigenvalues)

    print("\nFinal PCA Summary")
    print("-" * 40)
    for alpha, info in sorted(results.items()):
        print(f"  α = {alpha:.2f}  →  k = {info['k']:4d} components  |  "
              f"X_train_pca: {info['X_train_pca'].shape}  |  "
              f"X_test_pca: {info['X_test_pca'].shape}")