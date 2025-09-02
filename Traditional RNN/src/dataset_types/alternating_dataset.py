import random
import re
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class AlternatingSequenceDataset(Dataset):
    """
    Dataset for sequences designed to work with SSP (Sliding-Window Sequence Pooling).
    Each item returns:
        - images_tensor: [sequence_length, C, H, W]
        - paths: list of frame paths
        - transition_label: 1 if any frame in the sequence is a transition
        - transition_info: indices of transitions within the sequence
    """
    def __init__(self, root_dir, sequence_length=15, transform=None, transition_frames=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.transition_frames = transition_frames or {}

        # Gather all videos
        self.video_names = set()
        pattern = re.compile(r"(.+?_fullvid\.(?:mp4|MP4))_frame_\d+\.jpg")
        for fname in os.listdir(root_dir):
            match = pattern.match(fname)
            if match:
                self.video_names.add(match.group(1))
        self.video_names = sorted(self.video_names)

        # Map video -> frame paths
        self.video_to_frames = {}
        for fname in sorted(os.listdir(root_dir), key=self.natural_sort_key):
            video_name = self.extract_video_name(fname)
            if video_name:
                self.video_to_frames.setdefault(video_name, []).append(os.path.join(root_dir, fname))

        # Build sequences
        self.sequences = []
        for video, frames in self.video_to_frames.items():
            num_frames = len(frames)
            for i in range(num_frames - sequence_length + 1):
                seq = frames[i:i+sequence_length]
                if self.is_sequential(seq):
                    label = int(self.contains_transition(video, seq))
                    self.sequences.append((video, seq, label))

        # Alternate 0 and 1 sequences
        label_0 = [s for s in self.sequences if s[2]==0]
        label_1 = [s for s in self.sequences if s[2]==1]
        min_len = min(len(label_0), len(label_1))
        random.shuffle(label_0)
        random.shuffle(label_1)
        self.sequences = [val for pair in zip(label_0[:min_len], label_1[:min_len]) for val in pair]

        print(f"Dataset initialized: {len(self.sequences)} sequences, sequence length {sequence_length}")

    def natural_sort_key(self, fname):
        match = re.search(r'(?P<video_name>.*?)(?P<frame_number>\d+)(?:\.jpg|\.png)', fname)
        if match:
            return (match.group('video_name'), int(match.group('frame_number')))
        return (fname, 0)

    def extract_video_name(self, fname):
        match = re.search(r'(.*?)(_frame_)\d+', fname)
        return match.group(1) if match else None

    def extract_frame_number(self, path):
        match = re.search(r'_frame_(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else -1

    def is_sequential(self, paths):
        nums = [self.extract_frame_number(p) for p in paths]
        return nums == list(range(nums[0], nums[0]+len(nums)))

    def contains_transition(self, video_name, seq):
        if video_name not in self.transition_frames:
            return False
        frame_nums = [self.extract_frame_number(p) for p in seq]
        return any(f in set(self.transition_frames[video_name]) for f in frame_nums)

    def get_transition_positions(self, video_name, seq):
        if video_name not in self.transition_frames:
            return []
        frame_nums = [self.extract_frame_number(p) for p in seq]
        transition_set = set(self.transition_frames[video_name])
        return [i for i, f in enumerate(frame_nums) if f in transition_set]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        print("Getting Sequential seq")
        video_name, seq_paths, label = self.sequences[idx]

        images = []
        for p in seq_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images_tensor = torch.stack(images)  # [sequence_length, C, H, W]

        transition_info = self.get_transition_positions(video_name, seq_paths)

        return images_tensor, seq_paths, label, transition_info
