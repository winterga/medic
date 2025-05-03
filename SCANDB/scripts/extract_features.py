import torch  
import torch.nn as nn 
import torchvision.transforms as transforms 
import torchvision.models as models

import os
import sys
import numpy as np
from PIL import Image
import argparse
from natsort import natsorted


sys.path.append('path/to/src_dir')  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# To load a model, check if the entire model was saved (torch.save(model, "model.pt"))  
# or if only the weights were saved (torch.save(model.state_dict(), "model.pt")).  

# 1. Full architecture - torch.save(model, "model.pt")  
# 2. Weights only - torch.save(model.state_dict(), "model.pt")  

# classes:
# class 1: outside
# class 2: inside
# class 3: menu


def load_model(model_path, state):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if state not in [1, 2]:
        raise ValueError("Invalid state value. Use 1 for full model or 2 for weights only.")

    if state == 1:
        # Load the full model (assumes it was saved with torch.save(model))
        model = torch.load(model_path, map_location=device)

    elif state == 2:
        # Load ResNet-50 and load weights (assumes torch.save(model.state_dict()) was used)
        model = models.resnet50(pretrained=False)
        state_dict = torch.load(model_path, map_location=device)

        # Load weights with flexibility
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    return model


def extract_features(images, model_path, state, device):
    model = load_model(model_path, state)
    model.eval()

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_features = []

    for i, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = feature_extractor(image)
            features = features.view(features.size(0), -1)  # Flatten
            all_features.append(features.cpu().numpy())

    all_features = np.vstack(all_features)  
    return all_features                     



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extracting Feautres using trained ResNet50 Feature extractor")

    # change directories to 
    parser.add_argument("--checkpoint_path", type=str, default="feature_extractor/checkpoints/Resnet50_021225_07/Resnet50_022125_12.pth", help="path to trained feature extractor checkpoint")
    parser.add_argument("--frames_dir", type=str, default="path/to/extracted_frames", help="paths to extracted frames usuallly a directory of subdirectories (containing images from different video)")
    parser.add_argument("--features_dir", type=str, default="path/to/save/extracted_features", help="path to save features")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    videos_to_cluster = sorted([os.path.join(args.frames_dir, video) for video in os.listdir(args.frames_dir)])
    for video_path in videos_to_cluster:
        video_name, ext =  os.path.splitext(os.path.basename(video_path))
        frames = natsorted([os.path.join(video_path, image) for image in os.listdir(video_path)])
        features = extract_features(frames, args.checkpoint_path, state=1, device=device)
        np.save(f"features_new/spatial_features_{video_name}.npy", features) 
