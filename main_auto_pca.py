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
from gmm import GMM


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

    indices = np.random.choice(len(X_test), size=n, replace=False)
    X_sample = X_test[indices]
    X_rec    = np.clip(ae.decode(ae.encode(X_sample)), 0, 1)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    fig.suptitle("Original vs Reconstructed")

    for i in range(n):
        axes[0, i].imshow(X_sample[i].reshape(img_shape), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(X_rec[i].reshape(img_shape), cmap="gray")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[AE] reconstruction saved -> {save_path}")

def run_kmeans(Z_train, Z_test, y_train, y_test, k=40):

    mu = Z_train.mean(axis=0)
    std = Z_train.std(axis=0) + 1e-8
    Z_tr = (Z_train - mu) / std
    Z_te = (Z_test - mu) / std

    km = KMeans(k=k)
    km.fit(Z_tr)

    mapping = map_clusters_to_labels(km.predict(Z_tr), y_train, k)
    y_pred  = np.array([mapping[c] for c in km.predict(Z_te)])

    return compute_accuracy(y_test, y_pred)

def run_gmm(Z_train, Z_test, y_train, y_test, k=40):
    mu = Z_train.mean(axis=0)
    std = Z_train.std(axis=0)

    # Filter out dead latent dimensions
    active = std > 0.001
    if np.sum(active) == 0:
        Z_tr = Z_train
        Z_te = Z_test
    else:
        Z_tr = (Z_train[:, active] - mu[active]) / (std[active] + 1e-8)
        Z_te = (Z_test[:, active] - mu[active]) / (std[active] + 1e-8)

    gmm = GMM(n_components=k, covariance_type='tied', random_state=0)
    gmm.fit(Z_tr)

    mapping = map_clusters_to_labels(gmm.predict(Z_tr), y_train, k)
    y_pred  = np.array([mapping[c] for c in gmm.predict(Z_te)])

    return compute_accuracy(y_test, y_pred)


# ============================================================
# MAIN
# ============================================================
def main(dataset_root):

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    D, y = load_dataset(dataset_root)
    X_train, X_test, y_train, y_test = split_dataset(D, y)

    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0

    k = 40

    # ========================================================
    # PCA
    # ========================================================
    print("\n" + "=" * 60)
    print("PCA + KMEANS & GMM")
    print("=" * 60)

    pca_results = run_pca(X_train, X_test, show_plots=False)

    alphas = [0.80, 0.85, 0.90, 0.95]
    pca_dims = {}
    pca_accs_km = {}
    pca_accs_gmm = {}

    for alpha in alphas:
        Xtr = pca_results[alpha]["X_train_pca"]
        Xte = pca_results[alpha]["X_test_pca"]
        dim = Xtr.shape[1]

        acc_km = run_kmeans(Xtr, Xte, y_train, y_test, k)
        acc_gmm = run_gmm(Xtr, Xte, y_train, y_test, k)

        pca_dims[alpha] = dim
        pca_accs_km[alpha] = acc_km
        pca_accs_gmm[alpha] = acc_gmm

        print(f"[PCA] alpha={alpha} | dim={dim:3d} | KMeans acc={acc_km:.4f} | GMM acc={acc_gmm:.4f}")

    # ========================================================
    # AUTOENCODER — one per alpha (latent_dim = PCA dim)
    # ========================================================
    print("\n" + "=" * 60)
    print("AUTOENCODER + KMEANS & GMM (tuned over all PCA dims)")
    print("=" * 60)

    ae_accs_km = {}
    ae_accs_gmm = {}

    for alpha in alphas:
        dim = pca_dims[alpha]

        print(f"\n--- latent_dim = {dim} (alpha={alpha}) ---")

        ae = Autoencoder(
            input_dim=X_train.shape[1],
            latent_dim=dim,
            lr=0.005
        )

        ae.fit(X_train, epochs=100, batch_size=32, verbose=True)

        Z_train = ae.encode(X_train)
        Z_test  = ae.encode(X_test)

        acc_km = run_kmeans(Z_train, Z_test, y_train, y_test, k)
        acc_gmm = run_gmm(Z_train, Z_test, y_train, y_test, k)

        ae_accs_km[alpha] = acc_km
        ae_accs_gmm[alpha] = acc_gmm

        print(f"[AE ] alpha={alpha} | dim={dim:3d} | KMeans acc={acc_km:.4f} | GMM acc={acc_gmm:.4f}")

    # ========================================================
    # FINAL COMPARISON FOR K-MEANS
    # ========================================================
    print("\n" + "=" * 60)
    print("FINAL K-MEANS COMPARISON")
    print("=" * 60)
    print(f"{'Alpha':<8} {'Dim':<6} {'PCA acc':<12} {'AE acc':<12} {'Winner'}")
    print("-" * 52)

    best_pca_km_acc = 0
    best_ae_km_acc  = 0
    best_pca_km_alpha = None
    best_ae_km_alpha  = None

    for alpha in alphas:
        dim     = pca_dims[alpha]
        pca_acc = pca_accs_km[alpha]
        ae_acc  = ae_accs_km[alpha]
        winner  = "AE ✔" if ae_acc > pca_acc else "PCA ✔"

        print(f"{alpha:<8} {dim:<6} {pca_acc:<12.4f} {ae_acc:<12.4f} {winner}")

        if pca_acc > best_pca_km_acc:
            best_pca_km_acc   = pca_acc
            best_pca_km_alpha = alpha

        if ae_acc > best_ae_km_acc:
            best_ae_km_acc   = ae_acc
            best_ae_km_alpha = alpha

    print("-" * 52)
    print(f"Best PCA KMeans: acc={best_pca_km_acc:.4f}  (alpha={best_pca_km_alpha}, dim={pca_dims[best_pca_km_alpha]})")
    print(f"Best AE KMeans : acc={best_ae_km_acc:.4f}  (alpha={best_ae_km_alpha},  dim={pca_dims[best_ae_km_alpha]})")

    # ========================================================
    # FINAL COMPARISON FOR GMM
    # ========================================================
    print("\n" + "=" * 60)
    print("FINAL GMM COMPARISON")
    print("=" * 60)
    print(f"{'Alpha':<8} {'Dim':<6} {'PCA acc':<12} {'AE acc':<12} {'Winner'}")
    print("-" * 52)

    best_pca_gmm_acc = 0
    best_ae_gmm_acc  = 0
    best_pca_gmm_alpha = None
    best_ae_gmm_alpha  = None

    for alpha in alphas:
        dim     = pca_dims[alpha]
        pca_acc = pca_accs_gmm[alpha]
        ae_acc  = ae_accs_gmm[alpha]
        winner  = "AE ✔" if ae_acc > pca_acc else "PCA ✔"

        print(f"{alpha:<8} {dim:<6} {pca_acc:<12.4f} {ae_acc:<12.4f} {winner}")

        if pca_acc > best_pca_gmm_acc:
            best_pca_gmm_acc   = pca_acc
            best_pca_gmm_alpha = alpha

        if ae_acc > best_ae_gmm_acc:
            best_ae_gmm_acc   = ae_acc
            best_ae_gmm_alpha = alpha

    print("-" * 52)
    print(f"Best PCA GMM: acc={best_pca_gmm_acc:.4f}  (alpha={best_pca_gmm_alpha}, dim={pca_dims[best_pca_gmm_alpha]})")
    print(f"Best AE GMM : acc={best_ae_gmm_acc:.4f}  (alpha={best_ae_gmm_alpha},  dim={pca_dims[best_ae_gmm_alpha]})")

    # ========================================================
    # OVERALL ANALYSIS
    # ========================================================
    print("\n" + "=" * 60)
    print("OVERALL BEST PERFORMER ANALYSIS")
    print("=" * 60)
    methods = {
        "PCA + KMeans": best_pca_km_acc,
        "AE + KMeans": best_ae_km_acc,
        "PCA + GMM": best_pca_gmm_acc,
        "AE + GMM": best_ae_gmm_acc
    }
    overall_winner = max(methods, key=methods.get)
    print(f"Overall Winner : {overall_winner} ✔ (accuracy = {methods[overall_winner]:.4f})")
    print("=" * 60)

    # ========================================================
    # SAVE RECONSTRUCTIONS for the best AE dim
    # ========================================================
    best_ae_alpha = best_ae_km_alpha if best_ae_km_acc > best_ae_gmm_acc else best_ae_gmm_alpha
    best_dim = pca_dims[best_ae_alpha]

    ae_best = Autoencoder(input_dim=X_train.shape[1], latent_dim=best_dim, lr=0.005)
    ae_best.fit(X_train, epochs=100, batch_size=32, verbose=False)
    show_reconstructions(ae_best, X_test)

    # ========================================================
    # ACCURACY BAR CHART - K-MEANS
    # ========================================================
    x      = np.arange(len(alphas))
    width  = 0.35
    labels = [f"α={a}\ndim={pca_dims[a]}" for a in alphas]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [pca_accs_km[a] for a in alphas], width, label="PCA + KMeans")
    ax.bar(x + width / 2, [ae_accs_km[a]  for a in alphas], width, label="AE  + KMeans")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("PCA vs Autoencoder accuracy per latent dimension (KMeans)")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("comparison_kmeans.png", dpi=120, bbox_inches="tight")
    plt.savefig("comparison.png", dpi=120, bbox_inches="tight")  # backward compatibility
    plt.close()
    print("\n[plot] K-Means comparison saved -> comparison_kmeans.png")

    # ========================================================
    # ACCURACY BAR CHART - GMM
    # ========================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [pca_accs_gmm[a] for a in alphas], width, label="PCA + GMM", color="orange")
    ax.bar(x + width / 2, [ae_accs_gmm[a]  for a in alphas], width, label="AE  + GMM", color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("PCA vs Autoencoder accuracy per latent dimension (GMM)")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("comparison_gmm.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[plot] GMM comparison saved -> comparison_gmm.png")


# ============================================================
if __name__ == "__main__":

    dataset_root = "./archive"

    main(dataset_root)