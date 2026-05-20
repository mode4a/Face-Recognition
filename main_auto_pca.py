import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

from utilities.vectorization import load_dataset
from utilities.splitData import split_dataset

from PCA import run_pca
from kmeans import KMeans
from autoencoder import Autoencoder


# ============================================================
# Cluster -> Label mapping
# ============================================================
def map_clusters_to_labels(cluster_ids, true_labels, k):

    mapping = {}

    for cluster in range(k):

        mask = cluster_ids == cluster

        if np.sum(mask) == 0:
            continue

        majority = Counter(true_labels[mask]).most_common(1)[0][0]

        mapping[cluster] = majority

    return mapping


# ============================================================
# Accuracy
# ============================================================
def compute_accuracy(y_true, y_pred):

    return np.mean(y_true == y_pred)


# ============================================================
# Reconstruction visualization
# ============================================================
def show_reconstructions(
        ae,
        X_test,
        img_shape=(112, 92),
        n=8,
        save_path="reconstructions.png"
):

    indices = np.random.choice(
        len(X_test),
        size=n,
        replace=False
    )

    X_sample = X_test[indices]

    Z = ae.encode(X_sample)

    X_rec = ae.decode(Z)

    X_rec = np.clip(X_rec, 0, 1)

    fig, axes = plt.subplots(
        2,
        n,
        figsize=(2 * n, 5)
    )

    fig.suptitle("Original vs Reconstructed")

    for i in range(n):

        axes[0, i].imshow(
            X_sample[i].reshape(img_shape),
            cmap="gray"
        )

        axes[0, i].axis("off")

        axes[1, i].imshow(
            X_rec[i].reshape(img_shape),
            cmap="gray"
        )

        axes[1, i].axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[AE] reconstruction saved -> {save_path}")


# ============================================================
# MAIN
# ============================================================
def main(dataset_root):

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    D, y = load_dataset(dataset_root)

    X_train, X_test, y_train, y_test = split_dataset(D, y)

    # ========================================================
    # NORMALIZATION
    # ========================================================
    X_train = X_train.astype(np.float32) / 255.0

    X_test = X_test.astype(np.float32) / 255.0

    # ========================================================
    # PCA + KMEANS
    # ========================================================
    print("\n" + "=" * 60)
    print("PCA + KMEANS")
    print("=" * 60)

    pca_results = run_pca(
        X_train,
        X_test,
        show_plots=False
    )

    best_pca_acc = 0
    best_alpha = None

    # ORL dataset -> 40 people
    k = 40

    for alpha in [0.80, 0.85, 0.90, 0.95]:

        Xtr = pca_results[alpha]["X_train_pca"]

        Xte = pca_results[alpha]["X_test_pca"]

        # ====================================================
        # Standardization
        # ====================================================
        mu = Xtr.mean(axis=0)

        std = Xtr.std(axis=0) + 1e-8

        Xtr = (Xtr - mu) / std

        Xte = (Xte - mu) / std

        # ====================================================
        # KMeans
        # ====================================================
        kmeans = KMeans(k=k)

        kmeans.fit(Xtr)

        train_clusters = kmeans.predict(Xtr)

        test_clusters = kmeans.predict(Xte)

        mapping = map_clusters_to_labels(
            train_clusters,
            y_train,
            k
        )

        y_pred = np.array([
            mapping[c]
            for c in test_clusters
        ])

        acc = compute_accuracy(
            y_test,
            y_pred
        )

        print(
            f"[PCA] alpha={alpha} | "
            f"dim={Xtr.shape[1]} | "
            f"acc={acc:.4f}"
        )

        if acc > best_pca_acc:

            best_pca_acc = acc

            best_alpha = alpha

    print(
        f"\n[PCA] Best accuracy = "
        f"{best_pca_acc:.4f} "
        f"(alpha={best_alpha})"
    )

    # ========================================================
    # AUTOENCODER + KMEANS
    # ========================================================
    print("\n" + "=" * 60)
    print("AUTOENCODER + KMEANS")
    print("=" * 60)

    best_dim = pca_results[best_alpha]["X_train_pca"].shape[1]

    ae = Autoencoder(
        input_dim  = X_train.shape[1],   # 10304
        latent_dim = best_dim,            # 36
        hidden_dim = 512,
        lr         = 5e-3,    # FIX: 5x higher — needed to push gradients
                               #      through 4 layers back to the bottleneck
        lr_decay   = 0.999,   # FIX: much slower decay (was 0.995)
                               #      0.995^500 = 0.08  ← kills gradients
                               #      0.999^500 = 0.61  ← stays healthy
        seed       = 42,
    )

    ae.fit(
        X_train,
        epochs     = 100,
        batch_size = 32,
        verbose    = True,
    )

    # ========================================================
    # LATENT REPRESENTATIONS
    # ========================================================
    Z_train = ae.encode(X_train)
    Z_test  = ae.encode(X_test)

    mu  = Z_train.mean(axis=0)
    std = Z_train.std(axis=0)

    # drop dead latent dimensions (std≈0 → dividing by them = pure noise)
    active = std > 0.01
    print(f"\n[AE] Active latent dims : {active.sum()} / {len(active)}")

    if active.sum() == 0:
        print("[AE] WARNING: all latent dims are dead.")
        print("[AE] Falling back to raw standardized latent vectors.")
        # last resort: use raw (non-standardized) latent vectors
        Z_train_km = Z_train
        Z_test_km  = Z_test
    else:
        Z_train_km = (Z_train[:, active] - mu[active]) / std[active]
        Z_test_km  = (Z_test[:,  active] - mu[active]) / std[active]

    # ========================================================
    # KMEANS ON LATENT SPACE
    # ========================================================
    kmeans_ae = KMeans(k=k)

    kmeans_ae.fit(Z_train_km)

    train_clusters = kmeans_ae.predict(Z_train_km)

    test_clusters = kmeans_ae.predict(Z_test_km)

    mapping = map_clusters_to_labels(
        train_clusters,
        y_train,
        k
    )

    y_pred = np.array([
        mapping[c]
        for c in test_clusters
    ])

    ae_acc = compute_accuracy(
        y_test,
        y_pred
    )

    print(f"\n[AE] accuracy = {ae_acc:.4f}")

    # ========================================================
    # RECONSTRUCTIONS
    # ========================================================
    show_reconstructions(
        ae,
        X_test
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"PCA + KMeans : {best_pca_acc:.4f}")

    print(f"AE  + KMeans : {ae_acc:.4f}")

    if ae_acc > best_pca_acc:

        print("\nAutoencoder wins ✔")

    else:

        print("\nPCA wins ✔")


# ============================================================
if __name__ == "__main__":

    dataset_root = "./archive"

    main(dataset_root)