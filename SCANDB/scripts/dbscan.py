from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import  save_frames
from visualizer import plot_dbscan, plot_umap_projection
from evaluate import compute_cluster_metrics
from reduce_dim import dimensionality_reduction



def compute_optimal_eps_knee(features, video_name, save_dir, n_neighbors=20):
    if len(features.shape) != 2:
        raise ValueError("Expected a 2D array (frames, feature_dim). Check input shape.")

    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)
    distances = np.sort(distances[:, n_neighbors - 1], axis=0)

    os.makedirs(save_dir, exist_ok=True)

    k_dist_save_path = os.path.join(save_dir, f"{video_name}_k_distance_graph.png")
    plt.figure(figsize=(6, 6))
    plt.plot(distances, marker='o')
    plt.title('k-Distance Graph')
    plt.xlabel('Points (sorted by distance)')
    plt.ylabel(f'{n_neighbors}-th NN Distance')
    plt.grid()
    plt.savefig(k_dist_save_path)
    plt.close()

    kneedle = KneeLocator(range(len(distances)),
                          distances,
                          S=1.0,
                          curve="convex",
                          direction="increasing")
    optimal_eps = distances[kneedle.knee]

    knee_save_path = os.path.join(save_dir, f"{video_name}_knee_graph.png")
    kneedle.plot_knee()
    plt.savefig(knee_save_path)
    plt.close()

    return n_neighbors, optimal_eps


def dbscan(features, eps, min_samples):
    
    # Apply DBSCAN clustering
    dbscan_model = DBSCAN(eps=eps * 2, min_samples=min_samples, metric="euclidean")  # origina; cosine
    labels = dbscan_model.fit_predict(features)
    
    # Noise ratio calculation
    noise_ratio = np.sum(labels == -1) / len(labels)

    # Count unique labels (including noise)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_info = dict(zip(unique_labels, label_counts))
    
    return labels, label_info, noise_ratio



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Implementation of DBSCAN")

    # Data manipulation
    parser.add_argument("--frames_dir", type=str, default="path/to/extracted_frames", help="enter directory contain subdirectorire of extracted videos")
    parser.add_argument("--features_dir", type=str, default="paths/to/extracted/features", help="features are pre-extracted to save time. enter the  directoty where all the extracted feaures from all the videos")


    # DBSCAN clustering
    parser.add_argument("--save_path", type=str, default="path/to/save/dbscan/clustered_images", help="enter path to save clustered images after clustering")
    parser.add_argument("--plot_dir", type=str, default="path/to/save/knee_plots", help="path to save all knee graphs")
    parser.add_argument("--umap_graph_dir", type=str, default="pathe/to/save/umap_graphs", help="path to all dbscan clustering")
    parser.add_argument("--dbscan_graph_dir", type=str, default="pathe/to/save/dbscan_graphs", help="path to all dbscan clustering")

    args = parser.parse_args()



    videos_to_cluster = sorted([os.path.join(args.frames_dir, video) for video in os.listdir(args.frames_dir)])
    all_features = sorted([os.path.join(args.frames_dir, video) for video in os.listdir(args.frames_dir)])

    meta_data = {}
    reduce_dim = True

    

    for i, (features_path, video_path) in enumerate(zip(all_features, videos_to_cluster)):
        # get video name
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0]

        # extract frames to save after clustering
        frames = sorted([os.path.join(video_path, image) for image in os.listdir(video_path)])

        # load pre-extracted spatial feature
        spatial_features = np.load(features_path)
        print("loading spatial features ----------------------------------------------")

        # reduce dimension
        umap_features2d, reduced_features = dimensionality_reduction(spatial_features, method="umap")
        print("extracting_umap features-------------------------------------------------------------------------------")

        # visualize 2d UMAP reduced features
        plot_umap_projection(umap_features2d, video_name, args.umap_graph_dir)

        # find optimal epsilon value for dbscan
        min_pts, eps = compute_optimal_eps_knee(reduced_features, video_name, args.plot_dir)

        # fit features to DBSCAN (Actual DBSCAN Clustering)
        labels, info, noise_ratio = dbscan(reduced_features, eps, min_pts)

        plot_dbscan(reduced_features, labels, video_name, args.dbscan_graph_dir, "DBSCAN")

        si_score, db_score = compute_cluster_metrics(reduced_features, labels)
   
        save_frames(frames, labels, video_name, args.save_path)


        # Save each video’s results separately
        meta_data = {
            "db_score": float(db_score),
            "si_score": float(si_score),
            "noise_ratio": float(noise_ratio),
            "label_info": {int(k): int(v) for k, v in info.items()}
        }

        # Directory to save JSON files
        result_path = os.makedirs("evaluation_results", exist_ok=True)
        json_path = f"{result_path}/{video_name}.json"
        with open(json_path, "w") as file:
            json.dump(meta_data, file, indent=4)

        print(f"Results saved: {json_path}")

