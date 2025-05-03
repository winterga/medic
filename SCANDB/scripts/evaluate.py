from sklearn.metrics import silhouette_score, davies_bouldin_score

def compute_cluster_metrics(features, pred_labels):    # Note that the predicted labels is the labels from HDBSCAN/DBSCAN clusterin
    # Remove noise points (-1) from DBSCAN/HDBSCAN
    mask = pred_labels != -1
    valid_features = features[mask]
    valid_pred_labels = pred_labels[mask]
    
    # Ensure we have at least 2 clusters
    unique_clusters = len(set(valid_pred_labels))
    if unique_clusters < 2:
        print("Warning: Not enough clusters for evaluation. Returning default values.")
        return -1, float("inf"), -1

    # Compute metrics
    si_score = silhouette_score(valid_features, valid_pred_labels)
    db_score = davies_bouldin_score(valid_features, valid_pred_labels)

    print(f"Silhouette: {si_score:.4f}, DB Score: {db_score:.4f}")
    return si_score, db_score

