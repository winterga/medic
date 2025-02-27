import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

class VideoSegmentDataset(Dataset):
    def __init__(self, root_dir, seq_length=10, transform=None):
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.data = []
        
        # Iterate through each video folder
        for video_name in os.listdir(root_dir):
            video_path = os.path.join(root_dir, video_name)
            print(f"Organizing data for the following video path: {video_path}")
            if not os.path.isdir(video_path):
                continue
            
            for class_label in ["0", "1", "2"]:
                class_path = os.path.join(video_path, class_label)
                if not os.path.isdir(class_path):
                    continue
                
                frames = sorted(os.listdir(class_path))  # Sort to maintain order
                frame_paths = [os.path.join(class_path, f) for f in frames]
                
                # Generate sequences
                for i in range(0, len(frame_paths) - seq_length + 1, seq_length):
                    seq_frames = frame_paths[i:i+seq_length]
                    self.data.append((seq_frames, int(class_label)))
                print(f"Finished processing class {class_label} in video {video_name}")

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_paths, label = self.data[idx]
        
        frames = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        
        frames = torch.stack(frames)  # Shape: (seq_length, C, H, W)

        # Return frames and label, no feature extraction here
        return frames, label
