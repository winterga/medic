import random
import re
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None, transition_frames=None):
        if transition_frames is None:
            raise ValueError("Transition Frames is none")

        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.transition_frames = {}
        self.transition_frames_count = {}

        # Get video names
        self.video_names = set()
        pattern = re.compile(r"(.+\.mp4|.+\.MP4)_frame_\d+\.jpg")
        for entry in os.scandir(root_dir):
            if not entry.is_file():
                continue
            match = pattern.match(entry.name)
            if match:
                self.video_names.add(match.group(1))

        self.video_names = sorted(self.video_names)
        print("Vids", self.video_names)

        for video_name in self.video_names:
            self.transition_frames[video_name] = transition_frames[video_name]
            self.transition_frames_count[video_name] = len(self.transition_frames[video_name])

        # Load and sort all image paths
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.endswith(('.png', '.jpg'))]
        self.image_paths = sorted(self.image_paths, key=self.natural_sort_key)

        # Group by video name
        self.video_to_frames = {}
        for path in self.image_paths:
            video_name = self.extract_video_name(path)
            self.video_to_frames.setdefault(video_name, []).append(path)

        # Build flat sequential sequences (not prev/curr/next)
        self.sequences = []
        for video, frames in self.video_to_frames.items():
            for i in range(len(frames) - self.sequence_length + 1):
                seq = frames[i:i + self.sequence_length]
                if self.is_sequential(seq):
                    self.sequences.append((video, seq))

    def natural_sort_key(self, path):
        match = re.search(r'(?P<video_name>.*?)(?P<frame_number>\d+)(?:\.jpg|\.png)', os.path.basename(path))
        if match:
            return (match.group('video_name'), int(match.group('frame_number')))
        return (path, 0)

    def extract_video_name(self, path):
        match = re.search(r'(.*?)(_frame_)\d+', os.path.basename(path))
        return match.group(1) if match else None

    def extract_frame_number(self, path):
        match = re.search(r'_frame_(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else -1

    def is_sequential(self, frame_paths):
        frame_numbers = [self.extract_frame_number(p) for p in frame_paths]
        expected = list(range(frame_numbers[0], frame_numbers[0] + len(frame_numbers)))
        return frame_numbers == expected

    def get_transition_positions(self, video_name, seq, positions=None):
        if video_name not in self.transition_frames:
            return []

        transition_set = set(self.transition_frames[video_name])
        seq_len = len(seq)

        if positions is None:
            indices_to_check = range(seq_len)
        else:
            indices_to_check = [i if i >= 0 else seq_len + i for i in positions if -seq_len <= i < seq_len]

        return [i for i in indices_to_check if self.extract_frame_number(seq[i]) in transition_set]

    def contains_transition(self, video_name, seq, positions=None):
        if video_name not in self.transition_frames:
            return False
        transition_set = set(self.transition_frames[video_name])
        if positions is None:
            frame_numbers = [self.extract_frame_number(p) for p in seq]
        else:
            frame_numbers = [self.extract_frame_number(seq[i]) for i in positions if 0 <= i < len(seq)]
        return any(f in transition_set for f in frame_numbers)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        video_name, seq = self.sequences[idx]

        # Load images
        images = []
        for path in seq:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images_tensor = torch.stack(images)  # [sequence_length, C, H, W]

        # Transition info
        transition_positions = self.get_transition_positions(video_name, seq)
        transition_label = int(bool(transition_positions))
        transition_info = {"positions": transition_positions}

        return images_tensor, seq, transition_label, transition_info

    def restrict_to_transition_windows(self, min_context=400, max_context=1000):
        print(f"[INFO] Restricting to transition-centered windows: min={min_context}, max={max_context}")
        new_sequences = []

        for video_name in self.video_names:
            if video_name not in self.transition_frames:
                continue

            transitions = sorted(self.transition_frames[video_name])
            all_paths = self.video_to_frames[video_name]
            frame_indices = [self.extract_frame_number(p) for p in all_paths]
            frame_to_path = dict(zip(frame_indices, all_paths))

            windows = []
            for t in transitions:
                before = random.randint(min_context, max_context)
                after = random.randint(min_context, max_context)
                start = max(0, t - before)
                end = t + after
                windows.append((start, end))

            # Merge overlapping windows
            merged_windows = []
            for start, end in sorted(windows):
                if not merged_windows or start > merged_windows[-1][1]:
                    merged_windows.append([start, end])
                else:
                    merged_windows[-1][1] = max(merged_windows[-1][1], end)

            # Build sequences from merged windows
            for start_frame, end_frame in merged_windows:
                selected_frames = [f for f in range(start_frame, end_frame + 1) if f in frame_to_path]
                if len(selected_frames) < self.sequence_length:
                    continue
                sorted_paths = [frame_to_path[f] for f in sorted(selected_frames)]
                for i in range(len(sorted_paths) - self.sequence_length + 1):
                    seq = sorted_paths[i:i + self.sequence_length]
                    if self.is_sequential(seq):
                        new_sequences.append((video_name, seq))

        self.sequences = new_sequences
        print(f"[INFO] Total sequences after restriction: {len(self.sequences)}")
