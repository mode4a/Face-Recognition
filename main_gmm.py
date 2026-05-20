import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from utilities.vectorization import load_dataset
from utilities.splitData import split_dataset
from PCA import run_pca
from gmm import GMM

from utilities.evaluate import (
    map_clusters_to_labels,
    predict_labels,
    evaluate_clustering
)

def main(dataset_root, backend='sklearn'):
    print("=" * 70)
    print(f"LOADING DATASET ({dataset_root})")
    print("=" * 70)
    D, y = load_dataset(dataset_root)

    print("\n" + "=" * 70)
    print("SPLITTING DATASET")
    print("=" * 70)
    X_train, X_test, y_train, y_test = split_dataset(D, y)

    print("\n" + "=" * 70)
    print("RUNNING PCA")
    print("=" * 70)
    # Run PCA (loading cached values if they exist)
    pca_results = run_pca(
        X_train,
        X_test,
        show_plots=False
    )

    alphas = [0.80, 0.85, 0.90, 0.95]
    K_values = [20, 40, 60]

    results = []

    print("\n" + "=" * 70)
    print(f"RUNNING GMM EXPERIMENTS (Backend: {backend})")
    print("=" * 70)

    for alpha in alphas:
        print(f"\n[PCA alpha = {alpha}]")

        X_train_pca = pca_results[alpha]["X_train_pca"]
        X_test_pca = pca_results[alpha]["X_test_pca"]

        for K in K_values:
            print("\n" + "-" * 60)
            print(f"Running GMM with K = {K} (alpha = {alpha})")
            print("-" * 60)

            # Instantiating GMM (tied covariance is highly recommended to prevent overfitting and beat KMeans)
            gmm = GMM(
                n_components=K,
                covariance_type='tied',
                max_iters=150,
                tol=1e-4,
                random_state=0,
                backend=backend
            )

            gmm.fit(X_train_pca)

            # Predict clusters (from 0 to K-1)
            train_clusters = gmm.predict(X_train_pca)
            test_clusters = gmm.predict(X_test_pca)

            # Map GMM components to actual subject labels using training set majority voting
            mapping = map_clusters_to_labels(
                train_clusters,
                y_train,
                K
            )

            # Translate test cluster predictions to actual label IDs
            y_pred = predict_labels(
                test_clusters,
                mapping
            )

            # Calculate accuracy, F1 score and confusion matrix
            metrics = evaluate_clustering(
                y_test,
                y_pred
            )

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

    # =========================================================================
    # TABULATE AND DISPLAY FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(
        f"{'Alpha':>10} "
        f"{'K':>10} "
        f"{'Accuracy':>15} "
        f"{'F1 Score':>15}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['alpha']:>10.2f} "
            f"{r['K']:>10d} "
            f"{r['accuracy']:>15.4f} "
            f"{r['f1_score']:>15.4f}"
        )

    best_result = max(
        results,
        key=lambda x: x["accuracy"]
    )

    print("\n" + "=" * 70)
    print("BEST GMM MODEL")
    print("=" * 70)
    print(
        f"Alpha    : {best_result['alpha']}\n"
        f"K         : {best_result['K']}\n"
        f"Accuracy  : {best_result['accuracy']:.4f}\n"
        f"F1 Score  : {best_result['f1_score']:.4f}"
    )

    # =========================================================================
    # SAVE CONFUSION MATRIX PLOT
    # =========================================================================
    cm = best_result["confusion_matrix"]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Oranges')
    plt.title(
        f"Best GMM Confusion Matrix\n"
        f"Alpha={best_result['alpha']}, K={best_result['K']} (Acc={best_result['accuracy']:.4f})"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    plt.tight_layout()
    cm_path = "best_gmm_confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    plt.close()
    print(f"\n[GMM] Confusion matrix saved → {cm_path}")

    # =========================================================================
    # PLOT PERFORMANCE MEASURES (ACCURACY vs K & ALPHA)
    # =========================================================================
    print("[GMM] Plotting accuracy trends...")
    os.makedirs("./plots", exist_ok=True)

    # 1. Accuracy vs K value (grouped by alpha)
    plt.figure(figsize=(8, 5))
    for alpha in alphas:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        ks = [r["K"] for r in alpha_results]
        accs = [r["accuracy"] for r in alpha_results]
        plt.plot(ks, accs, marker='o', linewidth=2, label=f"alpha = {alpha}")
    plt.title("GMM Clustering: Accuracy vs. K (Number of Clusters)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Clustering Accuracy")
    plt.xticks(K_values)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_k_path = "./plots/gmm_accuracy_vs_k.png"
    plt.savefig(plot_k_path, dpi=120)
    plt.close()
    print(f"[GMM] Saved trend plot → {plot_k_path}")

    # 2. Accuracy vs Alpha value (grouped by K)
    plt.figure(figsize=(8, 5))
    for K in K_values:
        k_results = [r for r in results if r["K"] == K]
        alps = [r["alpha"] for r in k_results]
        accs = [r["accuracy"] for r in k_results]
        plt.plot(alps, accs, marker='s', linewidth=2, label=f"K = {K}")
    plt.title("GMM Clustering: Accuracy vs. Alpha (Variance Retained)")
    plt.xlabel("Alpha (Variance Retained)")
    plt.ylabel("Clustering Accuracy")
    plt.xticks(alphas)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plot_alpha_path = "./plots/gmm_accuracy_vs_alpha.png"
    plt.savefig(plot_alpha_path, dpi=120)
    plt.close()
    print(f"[GMM] Saved trend plot → {plot_alpha_path}")

    print("\nAll GMM experiments completed successfully.")
    return results

if __name__ == "__main__":
    dataset_root = "./archive"
    
    # Run with custom backend or sklearn backend.
    # Defaulting to sklearn for baseline stability, but you can pass 'custom' to test the EM code!
    backend_arg = 'sklearn'
    if len(sys.argv) > 1:
        backend_arg = sys.argv[1]
        
    main(dataset_root, backend=backend_arg)
