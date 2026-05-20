import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from utilities.vectorization import load_dataset
from utilities.splitData import split_dataset
from PCA import run_pca
from kmeans import KMeans

from utilities.evaluate import (
    map_clusters_to_labels,
    predict_labels,
    evaluate_clustering
)


def main(dataset_root):

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    D, y = load_dataset(dataset_root)

    print("\n" + "=" * 70)
    print("SPLITTING DATASET")
    print("=" * 70)

    X_train, X_test, y_train, y_test = split_dataset(D, y)

    print("\n" + "=" * 70)
    print("RUNNING PCA")
    print("=" * 70)

    pca_results = run_pca(
        X_train,
        X_test,
        show_plots=False
    )

    alphas = [0.80, 0.85, 0.90, 0.95]
    K_values = [20, 40, 60]

    results = []

    print("\n" + "=" * 70)
    print("RUNNING K-MEANS EXPERIMENTS")
    print("=" * 70)

    for alpha in alphas:

        print(f"\n[PCA alpha = {alpha}]")

        X_train_pca = pca_results[alpha]["X_train_pca"]
        X_test_pca = pca_results[alpha]["X_test_pca"]

        for K in K_values:

            print("\n" + "-" * 60)
            print(f"Running K-Means with K = {K}")
            print("-" * 60)

            kmeans = KMeans(k=K)
            kmeans.fit(X_train_pca)

            train_clusters = kmeans.predict(X_train_pca)
            test_clusters = kmeans.predict(X_test_pca)

            mapping = map_clusters_to_labels(
                train_clusters,
                y_train,
                K
            )

            y_pred = predict_labels(test_clusters, mapping)

            metrics = evaluate_clustering(y_test, y_pred)

            accuracy = metrics["accuracy"]
            f1 = metrics["f1_score"]

            results.append({
                "alpha": alpha,
                "K": K,
                "accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": metrics["confusion_matrix"]
            })

            print(f"Accuracy : {accuracy:.4f}")
            print(f"F1 Score : {f1:.4f}")

    # ============================================================
    # FINAL TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"{'Alpha':>10} {'K':>10} {'Accuracy':>15} {'F1 Score':>15}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['alpha']:>10.2f} "
            f"{r['K']:>10d} "
            f"{r['accuracy']:>15.4f} "
            f"{r['f1_score']:>15.4f}"
        )

    # ============================================================
    # BEST MODEL
    # ============================================================
    best_result = max(results, key=lambda x: x["accuracy"])

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Alpha    : {best_result['alpha']}\n"
        f"K        : {best_result['K']}\n"
        f"Accuracy : {best_result['accuracy']:.4f}\n"
        f"F1 Score : {best_result['f1_score']:.4f}"
    )

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================
    cm = best_result["confusion_matrix"]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')

    plt.title(
        f"Best KMeans Confusion Matrix\n"
        f"Alpha={best_result['alpha']}, K={best_result['K']}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()
    plt.tight_layout()

    plt.savefig("best_kmeans_confusion_matrix.png", dpi=120)
    plt.show()

    # ============================================================
    # ===================== PLOTS ===============================
    # ============================================================

    # ------------------------------------------------------------
    # 1. Accuracy vs K (for each alpha)
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        ks = [r["K"] for r in results if r["alpha"] == alpha]
        accs = [r["accuracy"] for r in results if r["alpha"] == alpha]

        plt.plot(ks, accs, marker="o", label=f"alpha={alpha}")

    plt.title("Accuracy vs K for different PCA alphas")
    plt.xlabel("K (clusters)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.savefig("accuracy_vs_k.png", dpi=120)
    plt.show()

    # ------------------------------------------------------------
    # 2. Accuracy vs alpha (for each K)
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for K in K_values:
        al = [r["alpha"] for r in results if r["K"] == K]
        accs = [r["accuracy"] for r in results if r["K"] == K]

        plt.plot(al, accs, marker="o", label=f"K={K}")

    plt.title("Accuracy vs PCA alpha for different K values")
    plt.xlabel("PCA alpha")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.savefig("accuracy_vs_alpha.png", dpi=120)
    plt.show()

    # ------------------------------------------------------------
    # 3. Heatmap (alpha × K)
    # ------------------------------------------------------------
    alphas_sorted = sorted(set(r["alpha"] for r in results))
    Ks_sorted = sorted(set(r["K"] for r in results))

    heatmap = np.zeros((len(alphas_sorted), len(Ks_sorted)))

    for r in results:
        i = alphas_sorted.index(r["alpha"])
        j = Ks_sorted.index(r["K"])
        heatmap[i, j] = r["accuracy"]

    plt.figure(figsize=(8, 6))

    plt.imshow(heatmap, cmap="viridis", aspect="auto")

    plt.xticks(range(len(Ks_sorted)), Ks_sorted)
    plt.yticks(range(len(alphas_sorted)), alphas_sorted)

    plt.xlabel("K (clusters)")
    plt.ylabel("PCA alpha")
    plt.title("Accuracy Heatmap (PCA alpha vs K)")

    plt.colorbar(label="Accuracy")

    plt.savefig("accuracy_heatmap.png", dpi=120)
    plt.show()

    # ============================================================
    print("\nAll experiments completed successfully.")


if __name__ == "__main__":

    dataset_root = "./archive"
    main(dataset_root)