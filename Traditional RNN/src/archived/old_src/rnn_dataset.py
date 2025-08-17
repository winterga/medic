import os
import re
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image

def natural_sort_key(path):
    """Extract frame number from the file name and sort numerically."""
    match = re.search(r'frame_(\d+)', os.path.basename(path))
    if match:
        return int(match.group(1))  # Return the frame number as an integer
    return 0  # If no match is found, return 0 (to avoid crashing)

def group_by_video(image_paths):
    """Group image paths by video identifier."""
    video_groups = {}
    for path in image_paths:
        # Extract the video name from the path, everything before '_frame'
        video_name = os.path.basename(path).split('_frame')[0]
        if video_name not in video_groups:
            video_groups[video_name] = []
        video_groups[video_name].append(path)
    return video_groups

def sort_by_frame_within_video(video_groups):
    """Sort images by frame number within each video group."""
    sorted_video_groups = {}
    for video_name, paths in video_groups.items():
        sorted_video_groups[video_name] = sorted(paths, key=natural_sort_key)  # Sort paths by frame number
    return sorted_video_groups

def is_frame_in_range(frame_number, ranges):
    """Check if the frame number is within any of the given ranges."""
    for start, end in ranges:
        if start <= frame_number <= end:
            return True
    return False

class RNNImageDataset(Dataset):
    def __init__(self, root, device, transform=None, sequence_length=9, train_list=None, valid_list=None, test_list=None):
        self.device = device
        self.sequence_length = sequence_length
        self.transform = Compose(transform) if transform else None

        train_list = {
            "02_05_2021_16_32_47_fullvideo.mp4": [
                (7983, 7999), (8630, 8646)
            ],
            "09_30_2022_11_38_00_fullvid.MP4": [
                (736, 749), (52670, 52683), (53047, 53060), (53078, 53091), (53099, 53112), (53116, 53129), (53721, 53734)
            ],
            "02_02_2024_08_41_58_fullvid.mp4": [
                (20980, 20993), (21098, 21111), (21215, 21228), (22017, 22030), (22322, 22335), (24002, 24015), (24865, 24878)
            ],
            "01_30_2023_07_11_48_fullvideo.MP4": [
                (58363, 58376)
            ],
            "11_07_2023_10_58_44_Ch2_001_CH001_V.MP4": [
                (7419, 7432), (7440, 7453), (9736, 9749)
            ],
            "12_22_2022_06_48_13_fullvid.MP4": [
                (18841, 18854), (19461, 19474), (34237, 34250), (37815, 37828), (38793, 38806), (40867, 40880), (43278, 43291), (44443, 44456), (52688, 52701), (53227, 53240), (55432, 55445), (59185, 59198), (59485, 59498), (60366, 60379)
            ]
        }

        valid_list = {
            "10_31_2022_07_18_14_fullvid.MP4": [
                (503, 516)
            ]
        }

        test_list = {
            "10_31_2023_14_36_44_fullvideo.mp4": [
                (4583, 4596), (5238, 5251), (7431, 7444), (8219, 8232), (9947, 9960), (10878, 10891), (12578, 12611), (13722, 13735), (14867, 14880)
            ],
            "08_22_2022_12_54_56_fullvid.MP4": [
                (78441, 78454), (789943, 78956), (78989, 79002), (79018, 79031)
            ]
        }

        self.train_list = train_list
        self.valid_list = valid_list
        self.test_list = test_list

        # Step 1: Read all image paths (ignoring folders)
        all_images = []
        for root_dir, _, files in os.walk(root):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Add more file extensions if needed
                    img_path = os.path.join(root_dir, file)
                    all_images.append(img_path)

        # Step 2: Group by video and sort within each video
        video_groups = group_by_video(all_images)
        sorted_video_groups = sort_by_frame_within_video(video_groups)
        
        # Step 3: Create sequences for each video
        self.samples = []
        for video_name, frames in sorted_video_groups.items():
            labeled_images = []
            for img_path in frames:
                frame_number = int(re.search(r'_frame_(\d+)', os.path.basename(img_path)).group(1))
                
                # Check if the frame is in any of the ranges from the train, valid, or test lists
                frame_in_train = is_frame_in_range(frame_number, self.train_list.get(video_name, []))
                frame_in_valid = is_frame_in_range(frame_number, self.valid_list.get(video_name, []))
                frame_in_test = is_frame_in_range(frame_number, self.test_list.get(video_name, []))

                # print(frame_number, frame_in_test,  frame_in_train, frame_in_valid)
                
                # Assign class based on whether the frame falls within any given ranges
                if frame_in_train or frame_in_valid or frame_in_test:
                    class_label = 1
                else:
                    class_label = 0

                labeled_images.append((img_path, class_label))
            
            # Create sequences within each video
            for i in range(len(labeled_images) - (self.sequence_length - 1)):
                sequence = labeled_images[i : i + self.sequence_length]
                paths = [s[0] for s in sequence]
                labels = [s[1] for s in sequence]

                # Sequence label is 1 if ANY frame in it is class 1
                sequence_label = 1 if any(label == 1 for label in labels) else 0

                self.samples.append((paths, sequence_label))

        self.num_samples = len(self.samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        paths, label = self.samples[idx]

        images = [Image.open(p).convert("RGB") for p in paths]
        if self.transform:
            images = [self.transform(img) for img in images]
        
        image_seq = torch.stack(images, dim=0)  # Shape: (9, C, H, W)
        label = torch.tensor(label, dtype=torch.float)  # Ensure compatibility with BCEWithLogitsLoss

        return image_seq, label, paths  # Paths included for debugging
