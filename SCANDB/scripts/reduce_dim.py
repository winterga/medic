from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np
import time

def dimensionality_reduction(features, n_components, method="PCA", random_state=42, dual_umap=False, scale=True):
    """
    Reduce feature dimensionality using PCA, UMAP, or t-SNE.
    
    Parameters:
        features (ndarray): Input feature matrix (num_samples x num_features)
        method (str): "PCA", "UMAP", or "TSNE"
        n_components (int): Number of output dimensions
        random_state (int): Random seed
        dual_umap (bool): If True, UMAP returns both 2D and 50D embeddings
        scale (bool): Whether to apply StandardScaler before reduction
    
    Returns:
        embedding (ndarray): Reduced feature matrix
        OR
        (embedding_2D, embedding_50D) if dual_umap=True and method="UMAP"
    """
    method = method.upper()
    start_time = time.time()

    if scale:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    else:
        features = normalize(features,"l2")
        print("using L2 norm")

    if method == "PCA":
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(features)

        # Cumulative variance explained
        explained_variance = np.cumsum(reducer.explained_variance_ratio_)
        print(f"[PCA] Explained variance by first {n_components} components: {explained_variance[-1]:.4f}")
        return embedding

    elif method == "UMAP":
        if dual_umap:
            # 2D UMAP for visualization
            reducer_2D = umap.UMAP(
                n_neighbors=100,         # Capture more global structure for large datasets
                min_dist=0.5,            # Lower values = tighter clusters; higher = more spread-out
                metric='euclidean',         # Cosine works well for deep features
                n_components=2,          # Dimensionality for visualization
                low_memory=True,         # Reduce memory for large datasets
                random_state=random_state,
                verbose=True
            )
            embedding_2D = reducer_2D.fit_transform(features)

            # 50D UMAP for clustering
            reducer_50D = umap.UMAP(
                n_neighbors=100,         # Use high neighborhood to preserve global structure
                min_dist=0.5,            # Higher min_dist preserves inter-cluster distance better
                metric='euclidean',
                n_components=n_components,         # Recommended for high-dim clustering (good SI/DB scores)
                low_memory=True,
                random_state=random_state,
                verbose=True
            )
            embedding_50D = reducer_50D.fit_transform(features)

            print(f"[UMAP] Completed in {(time.time() - start_time):.2f} seconds.")
            return embedding_2D, embedding_50D

        else:
            reducer = umap.UMAP(
                n_neighbors=100,
                min_dist=0.5,
                metric='euclidean',
                n_components=n_components,  # Custom output dim
                low_memory=True,
                random_state=random_state,
                verbose=True
            )
            embedding = reducer.fit_transform(features)
            print(f"[UMAP] Completed in {(time.time() - start_time):.2f} seconds.")
            return embedding


    elif method == "TSNE":
        reducer = TSNE(
            n_components=n_components,
            perplexity=30,              # Number of effective neighbors (~recommended: 5–50)
            n_iter=1000,                # Number of optimization steps
            learning_rate='auto',       # TSNE auto-tunes based on dataset
            random_state=random_state,
            verbose=1
        )
        embedding = reducer.fit_transform(features)
        print(f"[t-SNE] Completed in {(time.time() - start_time):.2f} seconds.")
        return embedding

    else:
        raise ValueError("Invalid method. Choose from 'PCA', 'UMAP', or 'TSNE'.")
