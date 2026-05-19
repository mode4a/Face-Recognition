import numpy as np
def split_dataset(D: np.ndarray, y: np.ndarray):
    validate_inputs(D, y)
    N = D.shape[0]   # total number of samples [400]
    train_indices = np.arange(0, N, 2)   # 0-based even → training
    test_indices  = np.arange(1, N, 2)   # 0-based odd  → testing

    X_train = D[train_indices]   # shape (200, 10304)
    X_test  = D[test_indices]    # shape (200, 10304)
    y_train = y[train_indices]   # shape (200,)
    y_test  = y[test_indices]    # shape (200,)

    print_split_summary(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test

def validate_inputs(D: np.ndarray, y: np.ndarray) -> None:
    # D must be a 2-D array.
    if D.ndim != 2: 
        raise ValueError(
            f"D must be 2-D (n_samples × n_features), got shape {D.shape}."
        )

    # y must be a 1-D array.
    if y.ndim != 1: 
        raise ValueError(
            f"y must be 1-D (n_samples,), got shape {y.shape}."
        )
    
    # D and y must have the same number of rows / elements.
    if D.shape[0] != y.shape[0]:
        raise ValueError(
            f"D and y must have the same number of rows. "
            f"Got D.shape[0]={D.shape[0]} and y.shape[0]={y.shape[0]}."
        )

    # The number of rows must be even (so the split is perfectly balanced).
    if D.shape[0] % 2 != 0:
        raise ValueError(
            f"Number of samples ({D.shape[0]}) must be even for the "
            "odd/even split to produce equally-sized subsets."
        )


def print_split_summary(X_train, X_test, y_train, y_test) -> None:
    print("[data_splitter] Dataset split complete.")
    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")

    # Count samples per subject in the training set
    unique_labels, train_counts = np.unique(y_train, return_counts=True)
    _, test_counts              = np.unique(y_test,  return_counts=True)

    # Check the expected 5-per-subject balance
    expected_per_subject = X_train.shape[0] // len(unique_labels)
    train_ok = np.all(train_counts == expected_per_subject)
    test_ok  = np.all(test_counts  == expected_per_subject)

    if train_ok and test_ok:
        print(f"  ✓ Each subject has exactly {expected_per_subject} images "
              "in both the training and test sets.")
    else:
        print("  ⚠ Imbalanced split detected — check dataset integrity.")
        print(f"    Train counts per subject: {train_counts}")
        print(f"    Test  counts per subject: {test_counts}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from vectorization import load_dataset

    if len(sys.argv) < 2:
        print("Usage: python data_splitter.py <path_to_dataset_root>")
        sys.exit(1)

    D, y = load_dataset(sys.argv[1])
    X_train, X_test, y_train, y_test = split_dataset(D, y)

    assert np.array_equal(X_train[0], D[0]), \
        "Row 0 of D (1-based odd) should be the first training row."

    # Row 1 of D (1-based row 2 → even → test) must be in X_test
    assert np.array_equal(X_test[0], D[1]), \
        "Row 1 of D (1-based even) should be the first test row."

    print("\n[data_splitter] All sanity checks passed!")