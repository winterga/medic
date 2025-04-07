import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

class ImagePathsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # root_dir has subfolders with images
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Recursively gather all .jpg/.png
        for subdir, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png')):
                    self.image_paths.append(os.path.join(subdir, f))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path  # return the path so we know where to save .npy

def load_resnet50_feature_extractor():
    feature_extractor_path = '/home-local/winterga/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth'
    feature_extractor = torch.load(feature_extractor_path, weights_only=False)
    # 2) Chop off the final FC layer to get a "feature extractor" -> outputs (N, 2048, 1, 1)
    # layers = list(resnet.children())[:-1]
    # fe = nn.Sequential(*layers)
    fe = nn.Sequential(*list(feature_extractor.children())[:-1]) # remove last FC layer so it outputs (N, 2048, 1, 1)
    return fe

@torch.no_grad()
def extract_features(data_loader, feature_extractor, device, out_dir):
    """
    data_loader: yields (batch_of_images, batch_of_paths)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for imgs, paths in data_loader:
        imgs = imgs.to(device)  # shape (B, 3, H, W)
        feats = feature_extractor(imgs)  # => (B, 2048, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)  # => (B, 2048)

        feats = feats.cpu().numpy()  # convert to numpy

        # Save each feature vector individually
        for i, p in enumerate(paths):
            # build the mirrored path in out_dir
            rel_path = os.path.relpath(p, start=data_loader.dataset.root_dir)
            # e.g. if p = /base/val/2/frame001.jpg => rel_path = val/2/frame001.jpg
            # convert extension to .npy
            base_name = os.path.splitext(rel_path)[0] + '.npy'
            out_path = os.path.join(out_dir, base_name)
            
            # ensure output subdirs exist
            subdir = os.path.dirname(out_path)
            os.makedirs(subdir, exist_ok=True)

            np.save(out_path, feats[i])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True,
                        help='Root folder with train/val/test subfolders')
    parser.add_argument('--out_dir', required=True,
                        help='Where to save extracted .npy features')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of CPU workers for DataLoader')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Resize dimension')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transform (resize + toTensor + normalize)
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # Build dataset & data loader
    dataset = ImagePathsDataset(root_dir=args.base_dir, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    fe = load_resnet50_feature_extractor().to(device)
    fe.eval()

    extract_features(data_loader, fe, device, args.out_dir)

if __name__ == "__main__":
    main()
