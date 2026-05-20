import numpy as np

from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# Cluster -> Label Mapping
# ============================================================

def map_clusters_to_labels(
    cluster_ids,
    true_labels,
    k
):

    mapping = {}

    for cluster in range(k):

        mask = cluster_ids == cluster

        if np.sum(mask) == 0:
            continue

        majority_label = Counter(
            true_labels[mask]
        ).most_common(1)[0][0]

        mapping[cluster] = majority_label

    return mapping


# ============================================================
# Convert Clusters -> Predicted Labels
# ============================================================

def predict_labels(
    cluster_ids,
    mapping
):

    return np.array([
        mapping.get(cluster_id, -1)
        for cluster_id in cluster_ids
    ])


# ============================================================
# Full Evaluation
# ============================================================

def evaluate_clustering(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average='macro'
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm
    }