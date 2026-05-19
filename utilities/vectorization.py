import os
import numpy as np
from PIL import Image         
def load_dataset(dataset_root: str):
    dataset_root = os.path.abspath(dataset_root)

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Please download the ORL dataset from "
            "https://www.kaggle.com/kasikrit/att-database-of-faces/"
        )

    # Collect (subject_id, image_path) pairs in a reproducible order
    # so that rows of D are always in the same sequence.
    records = collect_image_paths(dataset_root)

    if len(records) == 0:
        raise FileNotFoundError(
            f"No images found under: {dataset_root}\n"
            "Make sure the folder contains sub-folders named s1, s2, …, s40."
        )

    # Pre-allocate the Data Matrix.
    # IMAGE_SIZE = 92 × 112 = 10 304  
    IMAGE_SIZE = 92 * 112   # 10 304
    n_samples   = len(records)

    D = np.zeros((n_samples, IMAGE_SIZE), dtype=np.float64)
    y = np.zeros(n_samples, dtype=int)

    for idx, (subject_id, img_path) in enumerate(records):
        img_vector = image_to_vector(img_path, IMAGE_SIZE)
        D[idx] = img_vector   # store the flat vector as a row of D
        y[idx] = subject_id   # store the integer label (1 … 40)

    print(f"[data_loader] Loaded {n_samples} images.")
    print(f"[data_loader] Data Matrix D shape : {D.shape}")
    print(f"[data_loader] Label vector y shape : {y.shape}")
    print(f"[data_loader] Subjects found       : {np.unique(y).tolist()}")

    return D, y

def collect_image_paths(dataset_root: str):
    """
    Walk the dataset directory and return a sorted list of
    (subject_id, absolute_image_path) tuples.

    Sorting guarantees that:
      * subjects are processed in order s1, s2, …, s40
      * images within each subject are processed in order 1, 2, …, 10

    This reproducible ordering is critical so that the odd / even split
    in `data_splitter.py` always assigns the same images to training /
    testing regardless of file-system ordering.
    """

    records = []

    # List sub-folders that match the pattern "sN"
    subject_dirs = sorted(
        [d for d in os.listdir(dataset_root)
         if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("s")],
        key=lambda d: int(d[1:])   # sort numerically: s1 < s2 < … < s40
    )

    for subject_dir in subject_dirs:
        subject_id  = int(subject_dir[1:])   # "s3" → 3
        subject_path = os.path.join(dataset_root, subject_dir)

        # List image files inside the subject folder, sorted numerically
        image_files = sorted(
            [f for f in os.listdir(subject_path)
             if f.lower().endswith((".pgm", ".png", ".jpg", ".jpeg"))],
            key=lambda f: int(os.path.splitext(f)[0])  # "7.pgm" → 7
        )

        for img_file in image_files:
            img_path = os.path.join(subject_path, img_file)
            records.append((subject_id, img_path))

    return records


def image_to_vector(img_path: str, expected_size: int) -> np.ndarray:
    """
    Open a grayscale image, flatten it to a 1-D float64 array, and
    verify that the number of pixels matches `expected_size`.

    Parameters
    ----------
    img_path : str
        Absolute path to the image file.
    expected_size : int
        Expected number of pixels (10 304 for ORL images).

    Returns
    -------
    vector : np.ndarray, shape (expected_size,), dtype float64
    """

    img = Image.open(img_path).convert("L")   # "L" → 8-bit grayscale
    arr = np.array(img, dtype=np.float64)     # shape (112, 92) for ORL

    # Flatten: row-major (C order) gives a vector of length H × W
    vector = arr.flatten()

    if vector.size != expected_size:
        raise ValueError(
            f"Image {img_path} has {vector.size} pixels, "
            f"expected {expected_size}. "
            "Make sure you are using the correct ORL dataset."
        )

    return vector

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_dataset_root>")
        sys.exit(1)

    D, y = load_dataset(sys.argv[1])

    # Sanity checks
    assert D.shape == (400, 10304), f"Unexpected D shape: {D.shape}"
    assert y.shape == (400,),       f"Unexpected y shape: {y.shape}"
    assert y.min() == 1 and y.max() == 40, "Labels should be in range [1, 40]"

    print("\n[data_loader] All sanity checks passed!")
    print(f"  D dtype  : {D.dtype}")
    print(f"  y unique : {np.unique(y)}")
    print(f"  D[0,:5]  : {D[0, :5]}  (first 5 pixel values of image 0)")