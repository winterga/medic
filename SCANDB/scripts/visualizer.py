import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm

import matplotlib
matplotlib.use('Agg')  # Ensures no GUI window pops up



def plot_umap_projection(umap_features, video_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
   
    c = range(len(umap_features))
    colorbar_label = 'Feature Index'

    plt.figure(figsize=(15, 6))
    scatter = plt.scatter(
        umap_features[:, 0], 
        umap_features[:, 1], 
        c=c, 
        s=5, 
        cmap='Spectral'
    )
    plt.colorbar(scatter, label=colorbar_label)
    plt.title(f'UMAP Projection of Features from Video: {video_name}', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    save_path = os.path.join(save_dir, f"{video_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()



def plot_dbscan(features, labels, video_name, path, type = "DBSCAN"):
    os.makedirs(path, exist_ok=True)
    
    plt.figure(figsize=(16, 8))
    unique_labels = set(labels)
  
    colors = cm.get_cmap('tab20', len(unique_labels))  

    for label in unique_labels:
        cluster_mask = (labels == label)
        color = 'k' if label == -1 else colors(label)  # Black for noise
        plt.scatter(
            features[cluster_mask, 0], 
            features[cluster_mask, 1], 
            c=[color], 
            s=5, 
            label=f'Cluster {label}' if label != -1 else 'Noise'
        )

    plt.title(f"{type} Clustering on {video_name}", fontsize=16)
    plt.xlabel("X Dimension")
    plt.ylabel("Y Dimension")
    plt.legend(markerscale=5, loc='best', fontsize=10)
    
    save_path = os.path.join(path, f"{video_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to prevent display

    print(f"Cluster plot saved to: {save_path}")