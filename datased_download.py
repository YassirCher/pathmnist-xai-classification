"""
download_pathmnist.py
Loads PathMNIST from local NPZ and exports train/val/test arrays and metadata.

Run:
  python download_pathmnist.py
"""

import os
import json
import numpy as np
from typing import Tuple

VERBOSE = True

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def dataset_to_arrays(ds) -> Tuple[np.ndarray, np.ndarray]:
    """Convert medmnist dataset items (PIL or np) to numpy arrays with 3-channel RGB."""
    from PIL import Image
    images, labels = [], []

    for i in range(len(ds)):
        img, lab = ds[i]

        # Convert PIL to numpy
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Ensure HxWxC (3 channels)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        # Ensure dtype uint8
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # Normalize labels to int
        if isinstance(lab, (list, tuple, np.ndarray)):
            lab = int(lab[0])
        else:
            lab = int(lab)

        images.append(img)
        labels.append(lab)

        if VERBOSE and (i + 1) % 20000 == 0:
            print(f"Processed {i+1}/{len(ds)}...")

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y

def main():
    base_dir = r"C:\Users\yassi\Documents\projects\pathmnist XAI\dataset"
    ensure_dir(base_dir)

    try:
        import medmnist
        from medmnist import PathMNIST, INFO
    except ImportError:
        print("medmnist is not installed. Install with: pip install medmnist")
        raise

    local_npz = os.path.join(base_dir, "pathmnist.npz")
    if not os.path.exists(local_npz):
        raise FileNotFoundError(f"Expected NPZ at {local_npz}. Place and verify MD5 first.")

    info = INFO["pathmnist"]
    labels_map = info["label"] if isinstance(info["label"], dict) else {i: v for i, v in enumerate(info["label"])}
    n_classes = len(labels_map)

    print("Loading PathMNIST from local NPZ (download=False)...")
    train_dataset = PathMNIST(split="train", download=False, root=base_dir, size=28)
    val_dataset   = PathMNIST(split="val",   download=False, root=base_dir, size=28)
    test_dataset  = PathMNIST(split="test",  download=False, root=base_dir, size=28)

    print("Converting train split...")
    X_train, y_train = dataset_to_arrays(train_dataset)
    print("Converting val split...")
    X_val, y_val     = dataset_to_arrays(val_dataset)
    print("Converting test split...")
    X_test, y_test   = dataset_to_arrays(test_dataset)

    if VERBOSE:
        def mem_gb(a): return a.nbytes / (1024**3)
        print(f"Train: X={X_train.shape} ({mem_gb(X_train):.2f} GB), y={y_train.shape}")
        print(f"Val:   X={X_val.shape} ({mem_gb(X_val):.2f} GB), y={y_val.shape}")
        print(f"Test:  X={X_test.shape} ({mem_gb(X_test):.2f} GB), y={y_test.shape}")

    np.save(os.path.join(base_dir, "X_train.npy"), X_train)
    np.save(os.path.join(base_dir, "y_train.npy"), y_train)
    np.save(os.path.join(base_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(base_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(base_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(base_dir, "y_test.npy"),  y_test)

    meta = {
        "dataset": "PathMNIST",
        "num_classes": n_classes,
        "labels": {int(k): v for k, v in labels_map.items()},
        "image_size": [28, 28, 3],
        "source": "medmnist",
        "root": base_dir.replace("\\", "/")
    }
    with open(os.path.join(base_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved arrays and meta to: {base_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
