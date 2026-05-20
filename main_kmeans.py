import sys
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

            y_pred = predict_labels(
                test_clusters,
                mapping
            )

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


    print("\n" + "=" * 70)
    print("FINAL RESULTS")
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
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Alpha    : {best_result['alpha']}\n"
        f"K         : {best_result['K']}\n"
        f"Accuracy  : {best_result['accuracy']:.4f}\n"
        f"F1 Score  : {best_result['f1_score']:.4f}"
    )


    cm = best_result["confusion_matrix"]

    plt.figure(figsize=(10, 8))

    plt.imshow(cm, cmap='Blues')

    plt.title(
        f"Best KMeans Confusion Matrix\n"
        f"Alpha={best_result['alpha']}, "
        f"K={best_result['K']}"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        "best_kmeans_confusion_matrix.png",
        dpi=120
    )

    plt.show()

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":

    dataset_root = "./archive"

    main(dataset_root)