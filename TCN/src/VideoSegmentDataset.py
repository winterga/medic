# import os
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# # import random 
# import glob

# class VideoSegmentDataset(Dataset):
#     def __init__(self, root_dir, seq_length=8, sliding_step=1, transform=None):
#         """
#         Args:
#             root_dir (str): Path containing subfolders named '0', '1', '2', etc.
#             transform (callable, optional): Optional transform to be applied
#                 on a PIL image.
#             sequence_length (int): Number of frames in each sequence chunk.
#             sliding_step (int): How many frames to move forward when forming
#                 the next sequence (1 => fully sliding window).
#         """
#         self.root_dir = root_dir
#         self.seq_length = seq_length
#         self.sliding_step = sliding_step
#         # self.transform = transform if transform else transforms.Compose([
#         #     transforms.Resize((224, 224)),
#         #     transforms.ToTensor(),
#         # ])
#         self.transform = transform

#         self.sequence_items = [] # for holding a list of (sequence_of_paths, label)

#         # For each label folder, gather images by video
#         label_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

#         for label_str in label_folders:
#             label_path = os.path.join(root_dir, label_str)
#             if not os.path.isdir(label_path):
#                 print(f"Skipping {label_path} as it is not a directory.")
#                 continue
#             label = int(label_str)

#             all_images = glob.glob(os.path.join(label_path, '*.jpg'))

#             # Parse video_id and group

#             video_dict = {}
#             for img_path in all_images:
#                 filename = os.path.basename(img_path)
#                 parts = filename.split('_')
#                 # add the first 4 parts to the video_id
#                 video_id = ''.join(parts[:4])
                
#                 if video_id not in video_dict:
#                     video_dict[video_id] = []
#                 video_dict[video_id].append(img_path)

#             for video_id, img_paths in video_dict.items():
#                 img_paths.sort()

#                 # Create sliding windows of length sequence_length
#                 for start_idx in range(0, len(img_paths) - seq_length + 1, self.sliding_step):
#                     chunk_paths = img_paths[start_idx: start_idx + seq_length]

#                     # Store (list_of_paths, label) tuple
#                     self.sequence_items.append((chunk_paths, label))

#         # self.data = []



#         # for class_label in ["0", "1", "2"]:
#         #     class_folder = os.path.join(root_dir, class_label)
#         #     image_paths = glob.glob(os.path.join(class_folder, '*.jpg'))
#         #     for img in image_paths:
#         #         self.data.append((img, int(class_label)))

#         #     print(f"Finished processing class {class_label}")

#         # optionally suffle them
#         # rando.shuffle(self.data)


        
#         # Iterate through each video folder
#         # for video_name in os.listdir(root_dir):
#         #     video_path = os.path.join(root_dir, video_name)
#         #     print(f"Organizing data for the following video path: {video_path}")
#         #     if not os.path.isdir(video_path):
#         #         continue
            
#         #     for class_label in ["0", "1", "2"]:
#         #         class_path = os.path.join(video_path, class_label)
#         #         if not os.path.isdir(class_path):
#         #             continue
                
#         #         frames = sorted(os.listdir(class_path))  # Sort to maintain order
#         #         frame_paths = [os.path.join(class_path, f) for f in frames]
                
#         #         # Generate sequences
#         #         for i in range(0, len(frame_paths) - seq_length + 1, seq_length):
#         #             seq_frames = frame_paths[i:i+seq_length]
#         #             self.data.append((seq_frames, int(class_label)))
#         #         print(f"Finished processing class {class_label} in video {video_name}")

        
#     def __len__(self):
#         return len(self.sequence_items)

#     def __getitem__(self, idx):
#         """
#         Returns:
#             frames: Tensor of shape (sequence_length, C, H, W)
#             label:  int or Tensor (the label for the entire sequence)
#         """
#         chunk_paths, label = self.sequence_items[idx]
        
#         # Load each frame, apply transform
#         frames = []
#         for p in chunk_paths:
#             with Image.open(p).convert('RGB') as img:
#                 if self.transform is not None:
#                     img = self.transform(img)
#                 # shape => (C, H, W)
#                 frames.append(img)
        
#         # Stack => (sequence_length, C, H, W)
#         frames_tensor = torch.stack(frames, dim=0)
        
#         return frames_tensor, label

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoFeatureDataset(Dataset):
    """
    Loads consecutive .npy feature vectors (each .npy is shape (2048,)) 
    for sequences of length `seq_length`.
    """
    def __init__(self, root_dir, seq_length=8, sliding_step=1):
        """
        root_dir: Points to the directory containing subfolders '0','1','2', 
                  each with .npy feature files.
        """
        self.root_dir = root_dir
        self.seq_length = seq_length
        self.sliding_step = sliding_step
        self.sequence_items = []

        label_folders = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for label_str in label_folders:
            label_path = os.path.join(root_dir, label_str)
            label = int(label_str)  # 0, 1, or 2

            # find all .npy feature files
            npy_files = sorted(glob.glob(os.path.join(label_path, '*.npy')))
            
            # group by "video_id" if needed, or treat them all as a single sequence
            # example approach: parse a prefix from the filename
            video_dict = {}
            for path in npy_files:
                fname = os.path.basename(path)
                parts = fname.split('_')
                video_id = '_'.join(parts[:4])  # your logic for grouping
                if video_id not in video_dict:
                    video_dict[video_id] = []
                video_dict[video_id].append(path)

            # For each video, sort by filename, then chunk sequences
            for vid, paths in video_dict.items():
                paths.sort()
                for start_idx in range(0, len(paths) - self.seq_length + 1, self.sliding_step):
                    chunk = paths[start_idx:start_idx+self.seq_length]
                    self.sequence_items.append((chunk, label))

    def __len__(self):
        return len(self.sequence_items)

    def __getitem__(self, idx):
        chunk_paths, label = self.sequence_items[idx]
        feats = []
        for npy_path in chunk_paths:
            arr = np.load(npy_path)  # shape (2048,)
            feats.append(arr)
        # shape => (seq_length, 2048)
        feats = np.stack(feats, axis=0)
        feats_tensor = torch.from_numpy(feats).float()  # to torch.float32
        return feats_tensor, label

