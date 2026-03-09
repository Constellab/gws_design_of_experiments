"""
Script to run UMAP dimensionality reduction in isolated virtual environment.
This script takes preprocessed data and parameters, performs UMAP reduction
and optional K-Means clustering, and outputs results as pickle files.
"""

import pickle
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def run_umap(input_file: str, output_file: str):
    """
    Run UMAP dimensionality reduction and optional clustering.

    Args:
        input_file: Path to pickle file containing input data and parameters
        output_file: Path to pickle file where results will be saved
    """
    # Load input data
    with open(input_file, "rb") as f:
        input_data = pickle.load(f)

    x_values = input_data["x_values"]
    n_neighbors = input_data["n_neighbors"]
    min_dist = input_data["min_dist"]
    metric = input_data["metric"]
    scale_data = input_data["scale_data"]
    n_clusters = input_data["n_clusters"]

    # Scaling if requested
    if scale_data:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_values)
    else:
        x_scaled = x_values

    # Apply UMAP for 2D
    print("[PROGRESS:20] Running UMAP 2D reduction...")
    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric=metric,
    )
    embedding_2d = reducer_2d.fit_transform(x_scaled)
    embedding_2d = np.asarray(embedding_2d[0] if isinstance(embedding_2d, tuple) else embedding_2d)

    # Apply UMAP for 3D
    print("[PROGRESS:50] Running UMAP 3D reduction...")
    reducer_3d = UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        metric=metric,
    )
    embedding_3d = reducer_3d.fit_transform(x_scaled)
    embedding_3d = np.asarray(embedding_3d[0] if isinstance(embedding_3d, tuple) else embedding_3d)

    # Perform clustering if requested
    cluster_labels = None
    if n_clusters is not None:
        print("[PROGRESS:80] Running K-Means clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_2d)

    # Save results
    output_data = {
        "embedding_2d": embedding_2d,
        "embedding_3d": embedding_3d,
        "cluster_labels": cluster_labels,
    }

    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)

    print("[PROGRESS:100] UMAP computation completed!")
    print("[SUCCESS] UMAP completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python _run_umap.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    run_umap(input_file, output_file)
