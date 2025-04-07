import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

def load_resnet50_feature_extractor():
    # 1) Load a standard ResNet-50
    # resnet = models.resnet50(pretrained=True)
    feature_extractor_path = '/home-local/winterga/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth'
    feature_extractor = torch.load(feature_extractor_path, weights_only=False)
    # 2) Chop off the final FC layer to get a "feature extractor" -> outputs (N, 2048, 1, 1)
    # layers = list(resnet.children())[:-1]
    # fe = nn.Sequential(*layers)
    fe = nn.Sequential(*list(feature_extractor.children())[:-1]) # remove last FC layer so it outputs (N, 2048, 1, 1)
    fe.eval()
    return fe

@torch.no_grad()
def extract_single_image(fe, img_path, transform, device):
    """
    Loads a single image from `img_path`, applies `transform`, and extracts
    a (2048,) feature vector using the ResNet feature extractor `fe`.
    """
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    feats = fe(img_tensor)  # -> (1, 2048, 1, 1) for ResNet-50
    feats = feats.squeeze()  # -> (2048,)
    return feats.cpu().numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the ResNet feature extractor
    fe = load_resnet50_feature_extractor().to(device)
    fe.eval()  # important for inference

    # 2. Define the transform, including resize and normalization
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # We expect base_dir to have: train/, val/, test/ subfolders
    # each of which has label folders: 0, 1, 2
    splits = ["train", "val", "test"]

    for split in splits:
        in_split_dir = os.path.join(args.base_dir, split)
        out_split_dir = os.path.join(args.out_dir, split)

        if not os.path.isdir(in_split_dir):
            print(f"Skipping split '{split}' because {in_split_dir} does not exist.")
            continue

        os.makedirs(out_split_dir, exist_ok=True)

        # We'll look for label folders: 0,1,2 (or any subfolder if you want it more flexible)
        label_folders = sorted([
            d for d in os.listdir(in_split_dir)
            if os.path.isdir(os.path.join(in_split_dir, d))
        ])

        for label_str in label_folders:
            # e.g. label_str = "0", "1", "2"
            in_label_dir = os.path.join(in_split_dir, label_str)
            out_label_dir = os.path.join(out_split_dir, label_str)

            os.makedirs(out_label_dir, exist_ok=True)

            # gather all image files
            image_files = [
                f for f in os.listdir(in_label_dir)
                if f.lower().endswith('.jpg') or f.lower().endswith('.png')
            ]

            for img_name in image_files:
                img_path = os.path.join(in_label_dir, img_name)
                # Extract (2048,) features
                feats = extract_single_image(fe, img_path, transform, device)

                # Save as .npy
                out_name = os.path.splitext(img_name)[0] + ".npy"
                out_path = os.path.join(out_label_dir, out_name)
                np.save(out_path, feats)

            print(f"Finished extracting features for label {label_str} in {split} set.")

    print("Feature extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True,
                        help="Path containing train/, val/, test/ with label subfolders (0,1,2).")
    parser.add_argument("--out_dir", required=True,
                        help="Where to store the mirrored directory structure of .npy features.")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image resize dimension (224 recommended).")
    args = parser.parse_args()

    main(args)
