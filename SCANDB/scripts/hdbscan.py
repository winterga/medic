from hdbscan import HDBSCAN
import argparse
import os
import sys
import numpy as np


import json
import torch  
from utils import save_frames
from visualizer import plot_umap_projection, plot_dbscan
from evaluate import compute_cluster_metrics
from reduce_dim import dimensionality_reduction
from natsort import natsorted
import random
from hdbscan.validity import validity_index     # Density Based Clustering Validation
import pandas as pd




sys.path.append('/path/to/src')  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_hdbscan (features, min_cluster_size, min_samples):
    clusterer = HDBSCAN(min_cluster_size, min_samples, metric ="euclidean", gen_min_span_tree=True)
    clusterer.fit(features)
    labels = clusterer.labels_

    noise_ratio = sum(labels == -1) / len(labels)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_info = dict(zip(unique_labels, label_counts))
    print(f"length of unique labels {len(unique_labels)}")
    return labels, label_info, unique_labels, noise_ratio




if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Hdbscan for Kidney stone data")

    parser.add_argument("--output_dir", type=str, default="/home/oguinekj/Documents/MEDIC/extracted_frames", help="enter path to save frames from video extraction")

    parser.add_argument("--save_path", type=str, default="Hdbscan/clustered_image", help="enter path to save images after clustering")
    parser.add_argument("--plot_dir", type=str, default="Hdbscan/plots", help="path to save all graphs from this script")
    parser.add_argument("--cluster_path", type=str, default="Hdbscan/clustered_graphs", help="path to all dbscan clustering")
    parser.add_argument("--umap_path", type=str, default="Hdbscan/umap_cluster_initial", help="path to all dbscan clustering")
    parser.add_argument("--umap_path2", type=str, default="Hdbscan/umap_cluster_refined", help="path to all dbscan clustering")
    parser.add_argument("--features_path", type=str, default="features_logit_pred", help="enter path to save images after clustering")
    parser.add_argument("--json_path", type=str, default="Hdbscan/evaluation_results", help="path to save json file")


    # Hdbscan Parameters
    parser.add_argument("--min_cluster_size", type=int, default=100, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=100, help="Minimum cluster size for HDBSCAN")

    
    args = parser.parse_args()

    videos_to_cluster = sorted([os.path.join(args.output_dir, video) for video in os.listdir(args.output_dir)])
    _features = sorted([os.path.join(args.features_path, features) for features in os.listdir(args.features_path)])
    meta_data = {}

    # Directory to save JSON files
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                                                                                                                                                                                                                                                                                   
    for i, (features_path, video_path) in enumerate(zip(_features, videos_to_cluster)):
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0]
        frames = natsorted([os.path.join(video_path, image) for image in os.listdir(video_path)])

        features_2048 = np.load(features_path) 
        print(len(frames))
        print(len(features_2048))
        
        #features_2048 = dimensionality_reduction(features_2048, 128, "PCa", scale=False)

        features_2048 = dimensionality_reduction(features_2048, 128, "umap", scale=False)
        features_2048_2d = dimensionality_reduction(features_2048, 2, "Umap", scale=False)
       
        labels, info, unique_labels, noise = fit_hdbscan(features_2048, args.min_cluster_size, args.min_samples)

        si_score, db_score = compute_cluster_metrics(features_2048, labels)
        print(info)
    
        plot_dbscan(features_2048, labels, video_name, args.plot_dir)
        plot_umap_projection(features_2048_2d, video_name,args.umap_path, labels)
        plot_umap_projection(features_2048, video_name,args.umap_path2, labels)
        save_frames(frames, labels, video_name, args.save_path)
        

    
        meta_data = {
            "db_score": float(db_score),
            "si_score": float(si_score),
            "noise": float(noise),
            "label_info": {int(k): int(v) for k, v in info.items()},
            "validity (DBVC)": validity,
        }

        # Directory to save JSON files
        path = os.makedirs(args.json_path, exist_ok=True)
        json_path = os.path.join(args.json_path, f"{video_name}.json")
        with open(json_path, "w") as file:
            json.dump(meta_data, file, indent=4)

        print(f"Results saved: {json_path}")
