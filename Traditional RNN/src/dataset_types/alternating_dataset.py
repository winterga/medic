import random
import re
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

class AlternatingSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None, transition_frames=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.transition_frames = {}
        self.transition_frames_count = {}

        self.video_names = set()
        pattern = re.compile(r"(.+?_fullvid\.(?:mp4|MP4))_frame_\d+\.jpg")

        with os.scandir(root_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                match = pattern.match(entry.name)
                if match:
                    self.video_names.add(match.group(1))

        self.video_names = sorted(self.video_names)
        # print(self.video_names)
        # print(len(self.transition_frames[self.video_names[0]]))

        for video_name in self.video_names:
            self.transition_frames[video_name] = transition_frames[video_name]
            self.transition_frames_count[video_name] = int(len(self.transition_frames[video_name]))
            print("Transition frames count for video:", video_name, "is", self.transition_frames_count[video_name])

        # Load and sort image paths
        self.image_paths = [os.path.join(root_dir, fname) 
                            for fname in os.listdir(root_dir) 
                            if fname.endswith(('.png', '.jpg'))]
        self.image_paths = sorted(self.image_paths, key=self.natural_sort_key)

        # Group by video
        self.video_to_frames = {}
        for path in self.image_paths:
            video_name = self.extract_video_name(path)
            self.video_to_frames.setdefault(video_name, []).append(path)

        # Build labeled triplet sequences
        label_0_sequences = []
        label_1_sequences = []

        for video, frames in self.video_to_frames.items():
            print("Video", video, "has", len(frames), "frames")
            num_frames = len(frames)
            triplet_span = 3 * sequence_length
            print("Number of sequences:", len(frames) - triplet_span + 1)
            for i in range(num_frames - triplet_span + 1):
                prev_seq = frames[i : i + sequence_length]
                curr_seq = frames[i + sequence_length : i + 2 * sequence_length]
                next_seq = frames[i + 2 * sequence_length : i + 3 * sequence_length]
                full_seq = prev_seq + curr_seq + next_seq

                if self.is_sequential(full_seq):
                    label = int(self.contains_transition(video, curr_seq))
                    if label == 1:
                        label_1_sequences.append((video, prev_seq, curr_seq, next_seq, label))
                    else:
                        label_0_sequences.append((video, prev_seq, curr_seq, next_seq, label))
        print("Label 0 sequences:", len(label_0_sequences))
        print("Label 1 sequences:", len(label_1_sequences))

        # Shuffle both sets
        random.shuffle(label_0_sequences)
        random.shuffle(label_1_sequences)

        # Match lengths: use min length to balance
        min_len = min(len(label_1_sequences), len(label_0_sequences))
        print("Min length for balancing:", min_len)
        label_0_sequences = label_0_sequences[:min_len]
        label_1_sequences = label_1_sequences[:min_len]

        # Alternate between 0 and 1
        self.triplet_sequences = []
        for i in range(min_len):
            self.triplet_sequences.append(label_0_sequences[i])
            self.triplet_sequences.append(label_1_sequences[i])

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
        """
        Return the positions (indices) within `seq` that match transition frames.
        If `positions` is provided, only those indices are checked (can include negatives).
        """
        if video_name not in self.transition_frames:
            return []

        transition_set = set(self.transition_frames[video_name])
        seq_len = len(seq)

        if positions is None:
            indices_to_check = range(seq_len)
        else:
            indices_to_check = [
                i if i >= 0 else seq_len + i
                for i in positions
                if -seq_len <= i < seq_len
            ]

        matching_positions = []
        for i in indices_to_check:
            frame_num = self.extract_frame_number(seq[i])
            if frame_num in transition_set:
                matching_positions.append(int(i))  # 👈 ensure it's a Python int

        return matching_positions
    
    def contains_transition(self, video_name, seq):
        """
        Check if the sequence contains a transition frame based on the transition_frames dictionary.
        A transition is present if any of the frame numbers in the sequence match the transition frames.
        """
        if video_name in self.transition_frames:
            transition_set = set(self.transition_frames[video_name])
            frame_numbers = [self.extract_frame_number(p) for p in seq]
            return any(f in transition_set for f in frame_numbers)
        return False

    def __len__(self):
        return len(self.triplet_sequences)

    def __getitem__(self, idx):
        video_name, prev_seq, curr_seq, next_seq, label = self.triplet_sequences[idx]
        sequences = [prev_seq, curr_seq, next_seq]

        all_images = []
        all_paths = []

        for seq in sequences:
            images = []
            for path in seq:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            all_images.append(torch.stack(images))  # shape: (sequence_length, C, H, W)
            all_paths.append(seq)

        images_tensor = torch.stack(all_images)  # shape: (3, sequence_length, C, H, W)

        # Get transition positions
        prev_positions = self.get_transition_positions(video_name, prev_seq, positions=[-2, -1])
        curr_positions = self.get_transition_positions(video_name, curr_seq)
        next_positions = self.get_transition_positions(video_name, next_seq, positions=[0, 1])

        transition_label = int(bool(curr_positions))  # Still using curr_seq for main label

        transition_info = {
            "prev": prev_positions,
            "curr": curr_positions,
            "next": next_positions
        }

        return images_tensor, all_paths, transition_label, transition_info