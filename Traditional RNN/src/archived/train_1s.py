import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
import torch.utils.data as data
from torch.utils.data import Subset
import torch.optim as optim
from collections import Counter
import time, os, copy
from .resnet import resnet50
import tqdm
import random
import sys
import re
from statistics import mean, stdev
from .truth_data import transition_frames

from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torch.nn.init as init
import torch.nn.functional as F
from pathlib import Path

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

class SequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None, transition_frames=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.transition_frames = {}
        self.transition_frames_count = {}

        # TODO: Why do we do this?
        self.video_names = set()
        pattern = re.compile(r"(.+\.mp4|.+\.MP4)_frame_\d+\.jpg")

        with os.scandir(root_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                match = pattern.match(entry.name)
                if match:
                    self.video_names.add(match.group(1))

        self.video_names = sorted(self.video_names)
        print("Vids", self.video_names)

        for video_name in self.video_names:
            self.transition_frames[video_name] = transition_frames[video_name]
            self.transition_frames_count[video_name] = int(len(self.transition_frames[video_name]))
            print("Transition frames count for video:", video_name, "is", self.transition_frames_count[video_name])

        # Load and sort all image paths
        self.image_paths = [os.path.join(root_dir, fname) 
                            for fname in os.listdir(root_dir) 
                            if fname.endswith(('.png', '.jpg'))]
        self.image_paths = sorted(self.image_paths, key=self.natural_sort_key)

        # Group by video name
        self.video_to_frames = {}
        for path in self.image_paths:
            video_name = self.extract_video_name(path)
            if video_name not in self.video_to_frames:
                self.video_to_frames[video_name] = []
            self.video_to_frames[video_name].append(path)

        self.triplet_sequences = []
        for video, frames in self.video_to_frames.items():
            num_frames = len(frames)
            triplet_span = 3 * sequence_length
            for i in range(num_frames - triplet_span + 1):
                # Slice into prev, curr, next
                prev_seq = frames[i : i + sequence_length]
                curr_seq = frames[i + sequence_length : i + 2 * sequence_length]
                next_seq = frames[i + 2 * sequence_length : i + 3 * sequence_length]
                full_seq = prev_seq + curr_seq + next_seq

                if self.is_sequential(full_seq):
                    self.triplet_sequences.append((video, prev_seq, curr_seq, next_seq))

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
    
    def contains_transition(self, video_name, seq, positions=None):
        """
        Check if the sequence contains a transition frame.
        
        If `positions` is provided, only check those indices in the sequence.
        Otherwise, check the whole sequence.
        """
        if video_name not in self.transition_frames:
            return False

        transition_set = set(self.transition_frames[video_name])
        print("Transition Set:", transition_set)

        if positions is None:
            frame_numbers = [self.extract_frame_number(p) for p in seq]
        else:
            frame_numbers = [self.extract_frame_number(seq[i]) for i in positions if 0 <= i < len(seq)]

        return any(f in transition_set for f in frame_numbers)

    def __len__(self):
        return len(self.triplet_sequences)

    def __getitem__(self, idx):
        video_name, prev_seq, curr_seq, next_seq = self.triplet_sequences[idx]
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

        # Label = does the current sequence contain a transition?
        # Get transition positions
        prev_positions = self.get_transition_positions(video_name, prev_seq, positions=[-2, -1])
        curr_positions = self.get_transition_positions(video_name, curr_seq)  # check all
        next_positions = self.get_transition_positions(video_name, next_seq, positions=[0, 1])

        # Binary label
        transition_label = int(bool(prev_positions or curr_positions or next_positions))

        transition_info = {
            "prev": prev_positions,
            "curr": curr_positions,
            "next": next_positions
        }

        return images_tensor, all_paths, transition_label, transition_info

class StrongTransitionLSTM(nn.Module):

    def __init__(self, feature_dim=6153, num_classes=3, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        feature_dim: Dimension of CNN feature output (2048 for ResNet-50's second-to-last layer)
        num_classes: Number of classes predicted by the CNN (3 for your 3 classes: 0, 1, 2)
        """
        super().__init__()
        
        # Update input_dim to the combined size (features + predictions)
        # input_dim = feature_dim + num_classes  # Combine CNN features and predictions
        
        # LSTM setup for sequence input (CNN features and predictions combined)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layer to output transition score
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

        self.apply_xavier_init()

    def apply_xavier_init(self):
        # Apply Xavier initialization to LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)  # or init.xavier_normal_(param)
            elif 'bias' in name:
                init.zeros_(param)  # Initialize biases to zero

                hidden_size = param.shape[0] // 4
                param.data[hidden_size:2*hidden_size] = 1.0
        
        # Apply Xavier initialization to fully connected layer
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x: Input tensor of combined CNN features and predictions (shape: [B, S, input_dim])
        """
        # Forward pass through the LSTM layer
        features, _ = self.lstm(x)  # (B, seq_len, 2*hidden_dim)
        
        # Apply dropout to the last time step
        out = self.dropout(features[:, -1, :])  # Last timestep
        
        # Output from fully connected layer
        out = self.fc(out)
        
        # Sigmoid output for binary classification (transition/no-transition)
        # Probability of Positive (1) Class [Transition in this case]
        return features, out.squeeze(1), torch.sigmoid(out).squeeze(1)  # Output shape: (B,)
    
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.original_cnn_model = torch.load("/home/user/Documents/GitHub/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth")

        # Freeze all parameters in the original model
        for param in self.original_cnn_model.parameters():
            param.requires_grad = False

        # Remove the final classification layers (usually the last fully connected layer)
        self.features = nn.Sequential(*list(self.original_cnn_model.children())[:-1])
        self.logits = self.original_cnn_model.fc

    def forward(self, x):
        # Extract features from the second-to-last layer
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        y = self.logits(x)
        return x, y

def get_subset_indices(dataset, subset_percentage=0.1):
    """
    Get indices for a subset of the dataset (e.g., 10% of the data).
    """
    # Randomly shuffle indices
    indices = list(range(len(dataset)))
    subset_size = int(len(indices) * subset_percentage)
    subset_indices = random.sample(indices, subset_size)  # Randomly sample 10% of the data
    return subset_indices

def apply_gpu_transforms(image, transform_list):
    """
    Apply a series of GPU-based augmentations to an image tensor.
    :param image: A tensor image already on the GPU.
    :param transform_list: List of transforms to be applied to the image.
    :return: Transformed image on the GPU.
    """
    for t in transform_list:
        image = t(image)
    return image

def load_train(params, hyper_params):
    cpu_train_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]
    cpu_valid_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]
    cpu_test_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]

    # GPU-based transforms for heavy augmentations
    gpu_train_list = []
    
    if hyper_params['normalize']:
        cpu_train_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]))
        cpu_valid_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])),
        cpu_test_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]))
    image_transforms = {
        'train': v2.Compose(cpu_train_list),
        'valid': v2.Compose(cpu_valid_list),
        'test': v2.Compose(cpu_test_list)
    }
    
    # Load data from folders
    dataset_map = {
        'alternating': AlternatingSequenceDataset,
        'sequential': SequenceDataset
    }
    try:
        dataset_class = dataset_map[hyper_params['training_style']]
    except KeyError:
        raise ValueError(f"Invalid training_style: {hyper_params['training_style']}")

    train_dataset = dataset_class(
        root_dir='/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/train/',
        transform=image_transforms['train'],
        transition_frames=transition_frames
    )
    dataset = {
        'train': train_dataset,
        'valid': SequenceDataset(root_dir='/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/val/', transform=image_transforms['valid'], transition_frames=transition_frames),
        'test': SequenceDataset(root_dir='/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/test/', transform=image_transforms['test'], transition_frames=transition_frames)
    }

    
    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        'valid': len(dataset['valid']),
        'test': len(dataset['test'])
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'valid': data.DataLoader(dataset['valid'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'test': data.DataLoader(dataset['test'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cnn_model = CNNModel()
     # The model needs to accept 4 different sizes:
    # 1. 3x5 (5 images, 3 parts) w/ Logits          -> 3  * (2048 + 3)  = 6153
    # 2. 15 (15 images, single part) w/ Logits      -> 15 * (2048 + 3)  = 30765
    # 3. 3x5 (5 images, 3 parts) w/out Logits       -> 3  * (2048)      = 6144
    # 4. 15 (15 images, single part) w/out Logits   -> 15 * (2048)      = 30720
    input_dim = 0
    if hyper_params['images_in_batch'] == '3x5' and hyper_params['no_add_logits'] :
        input_dim = 6144
    elif hyper_params['images_in_batch'] == '3x5' and not hyper_params['no_add_logits']:
        input_dim = 6153
    elif hyper_params['images_in_batch'] == '15' and hyper_params['no_add_logits']:
        input_dim = 2048
    elif hyper_params['images_in_batch'] == '15' and not hyper_params['no_add_logits']:
        input_dim = 2051
    model_ft = StrongTransitionLSTM(input_dim).to(device)
    activation = nn.Softmax(dim=1)
    # class_weights = torch.tensor([1, 10]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    optimizer = optim.AdamW(model_ft.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params['num_epochs'], eta_min=hyper_params['learning_rate']/100.)

    return model_ft, cnn_model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list


def train_model(params, hyper_params):
    since = time.time()

    model, cnn_model, dataloaders, data_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list = load_train(params, hyper_params)
    model.to(device)
    print(f"Train: {next(model.parameters()).device}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = 0
    logger = open(os.path.join(params['save_dir'], params['name'] + '.txt'), "w")

    for epoch in range(hyper_params['num_epochs']):
        print('Epoch {}/{}'.format(epoch, hyper_params['num_epochs'] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid', 'test']:
            # criterion = nn.CrossEntropyLoss(weight=train_weights if phase == 'train' else valid_weights)
            # model.train() if phase == 'train' else model.eval()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   


            with tqdm.tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch}', unit='batch') as pbar:
                mod = 5
                i = 0

                epoch_loss = 0
                epoch_loss_0s = 0
                epoch_loss_1s = 0
                total_correct_sequences = 0
                total_correct_sequences_0s = 0
                total_correct_sequences_1s = 0
                total_sample_sequences = 0
                total_sample_sequences_0s = 0
                total_sample_sequences_1s = 0
                predicted_sequences_count = 0
                truth_sequence_count = 0

                # Sequences to transitions calculation
                """
                1. We calculate a transition as a set of 2-3 sequential sequences. Why 2-3?:
                Permitted frames to be considered a sequence: [4], [0-4], [0]
                - Batches of 3 sequences are moved left 3 frames
                - That means if the image was right-[0], it would then be mid-[2], and then be [4]. 
                
                2. We can't count directly, because it could be a set of sequences where [1, 0, 1] and this could be a sequence.
                But, we also don't want to hardcode the value and say [1, 0, 1, 1, 1] as 2 sequences. 
                a. See a [1]
                    a. See [1] before seeing 2 [0]s
                        a. Transition ends at seeing 2 consecutive [0]s
                    b. See 2 [0]s before seeing another [1]
                        a. No transition
                b. See a [0]
                    a. No transition
                **THIS STATES THAT [1, 1, 1, 1, 0, 1, 1, 1] WOULD BE ONE (1) SEQUENCE**

                3. We need to then look at the paths that annotate a transition to consider an EXACT frame as the transition point

                """
                current_sequence_labels = []
                transition_count = 0
                correct_transition_count = 0

                video_transitions_frames = {}

                ## PLOTTING
                all_features = []
                all_labels = []
                ## PLOTTING

                # Iterate over data.
                for j, (sequences, paths, label, info) in enumerate(dataloaders[phase]):

                    # Train should be all transition sequences, but validation/test should be every 5th
                    if phase == 'train' or i % mod == 0:
                        sequences = sequences.to(device, non_blocking=True) 
                        label = label.to(device, non_blocking=True).float() 
                        # inputs = apply_gpu_transforms(inputs, gpu_train_list)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):

                            # Example input to CNN
                            # sequences: [Batch, 3, Sequence, Channel, H, W]
                            B, num_parts, S, C, H, W = sequences.shape  # Batch size, sequence length, channels, height, width
                            print("Shape of sequences:", sequences.shape)

                            batch_preds = []
                            batch_sequence = []
                            
                            if(hyper_params['images_in_batch'] == '3x5'):
                                # batch_paths = path
                                for i in range(num_parts):  # prev, curr, next
                                    seq = sequences[:, i]  # [B, S, C, H, W]
                                    features, logits = cnn_model(seq.view(B * S, C, H, W))  # [B*S, F], [B*S, 3]
                                    features = features.view(B, S, -1)  # [B, S, F]
                                    logits = logits.view(B, S, -1)  # [B, S, 3]
                                    probs = F.softmax(logits, dim=-1)  # [B, S, 3] - softmax across the class dimension

                                    # Get the predicted class index` (0, 1, or 2)
                                    _, preds = torch.max(probs, 2)
                                    batch_preds.append(preds.tolist())  # [B, S, 3]


                                    # Concatenate features and predictions
                                    if hyper_params['no_add_logits']:
                                        batch_sequence.append(features)  # [B, S, F]
                                    else:
                                        combined = torch.cat([features, logits], dim=-1)  # [B, S, F+3]
                                        batch_sequence.append(combined)

                                full_sequence = torch.cat(batch_sequence, dim=-1)  # [B, S, 3*(F+3)]
                            
                            elif (hyper_params['images_in_batch'] == '15'):
                                # sequences: [B, 3, S, C, H, W] → merge across 3 to form flat [B, 15, C, H, W]
                                flat_sequence = sequences.view(B, 3 * S, C, H, W)  # [B, 15, C, H, W]
                                # print("SHAPE", flat_sequence.shape)
                                
                                # Flatten batch and sequence to feed into CNN
                                seq = flat_sequence  # [B, 15, C, H, W]
                                features, logits = cnn_model(seq.view(B * 15, C, H, W))  # [B*15, F], [B*15, 3]

                                # Reshape back to batch form
                                features = features.view(B, 15, -1)  # [B, 15, F]
                                logits = logits.view(B, 15, -1)      # [B, 15, 3]
                                probs = F.softmax(logits, dim=-1)    # [B, 15, 3]

                                _, preds = torch.max(probs, 2)       # [B, 15]
                                batch_preds.append(preds.tolist())

                                if hyper_params['no_add_logits']:
                                    full_sequence = features  # [B, 15, F]
                                else:
                                    full_sequence = torch.cat([features, logits], dim=-1)  # [B, 15, F+3]

                            else:
                                print("NONEXISTENT")
                                sys.exit(1)

                            lstm_features, outputs, probs = model(full_sequence)  # [B, 1]
                            all_features.append(lstm_features.cpu().detach().numpy())  # Store LSTM features for later analysis
                            all_labels.append(label.cpu().detach().numpy())  # Store labels for later analysis
                            loss = criterion(outputs, label)  # [B, 1] vs [B, 1]

                            # TSNE
                            # if phase == 'valid' and label.item() == 1 and epoch > 70:
                            #     from sklearn.manifold import TSNE
                            #     import matplotlib.pyplot as plt

                            #     # Step 1: Reduce features from [B*S, D] to [B*S, 2]
                            #     tsne_input = lstm_features.reshape(-1, lstm_features.shape[-1])  # [B*S, 2*hidden_dim]
                            #     tsne = TSNE(n_components=2, perplexity=2, random_state=42)
                            #     tsne_result = tsne.fit_transform(tsne_input.cpu().detach().numpy())  # convert to NumPy
                            #     plt.figure(figsize=(6, 5))
                            #     plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=[0, 0, 1, 1, 1], cmap='coolwarm', s=100)
                            #     plt.title("t-SNE of LSTM Features")
                            #     plt.xlabel("Dim 1")
                            #     plt.ylabel("Dim 2")
                            #     plt.grid(True)
                            #     plt.show()

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            
                            epoch_loss += loss.item()
                            epoch_loss_0s += loss.item() if label.item() == 0 else 0
                            epoch_loss_1s += loss.item() if label.item() == 1 else 0
                            
                            preds = (outputs > 0.5).float()  # Convert logits to binary predictions (0 or 1)
                            filenames = [[p[0].split('/')[-1] for p in group] for group in paths]
                            print(f"{outputs.item():.6f}", preds.item(), label.item(), batch_preds, info)
                            for group in filenames:
                                print([path for path in group])
                            
                            total_correct_sequences += torch.sum(preds == label.data)  # Count correct predictions
                            total_sample_sequences += label.size(0)  # Increment total samples
                            total_sample_sequences_0s += (label == 0).sum().item()
                            total_sample_sequences_1s += (label == 1).sum().item()
                            predicted_sequences_count += (preds.item() == 1)
                            total_correct_sequences_0s += torch.sum((preds == label.data) & (label == 0)).item()
                            total_correct_sequences_1s += torch.sum((preds == label.data) & (label == 1)).item()
                            truth_sequence_count += (label.item() == 1)
                            correct_transition_count += (preds.item() == label.item() == 1)

                            # Always start with 1 or add (0 | 1) to a current sequence
                            if phase != 'train':
                                if not (preds.item() == 0 and len(current_sequence_labels) == 0):
                                    current_sequence_labels.append((preds.item(), paths))
                                    curr_preds = [p[0] for p in current_sequence_labels]
                                    paths = [p[1] for p in current_sequence_labels][0]


                                    # curr_preds is a list of predictions for the current sequence
                                    # Check if the final two (2) of curr_preds `curr_preds[-2:]` ends in [0, 0] -- end of transition
                                    if curr_preds[-2:] == [0, 0] or j == int(len(dataloaders[phase]) / mod) * mod:

                                        # We need to be sure to account for the modular -- for example, if mod == 5, then we need to check if the last batch is the last batch
                                        # We only need to count the ones where it is > 2 (or do we? - consider that an image should be in 1/2/3 transitions?)
                                        if curr_preds.count(1) >= 2 or j == int(len(dataloaders[phase]) / mod) * mod:
                                            # Count transitions
                                            transition_count += 1
                                        

                                            # Print total transition frame start and end
                                            path_start = paths[0][0][0]
                                            path_end = paths[2][4][0]
                                            print(f"Transition located starting @ path_start: {path_start} and end: {path_end}")

                                            # Predict the frame using the middle of the sequence.
                                            """
                                            EXAMPLE: path_start: ['/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/val/10_31_2022_07_18_14_fullvid.MP4_frame_00496.jpg'] and end: ['/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/val/10_31_2022_07_18_14_fullvid.MP4_frame_00510.jpg']
                                            """

                                            ## First frame  
                                            first_pathname = Path(path_start).name  # Get just the filename
                                            first_match = re.match(r"(.*\.MP4)_frame_(\d+)\.jpg", first_pathname, re.IGNORECASE)
                                            second_pathname = Path(path_end).name  # Get just the filename
                                            second_match = re.match(r"(.*\.MP4)_frame_(\d+)\.jpg", second_pathname, re.IGNORECASE)

                                            if first_match and second_match:
                                                first_video_name = first_match.group(1)
                                                first_frame_number = int(first_match.group(2))
                                                second_video_name = second_match.group(1)
                                                second_frame_number = int(second_match.group(2))
                                                if first_video_name != second_video_name:
                                                    print("Warning: Video names do not match!")
                                                    sys.exit(1)
                                                if first_frame_number >= second_frame_number:
                                                    print("Warning: First frame number is not less than second frame number!")
                                                    sys.exit(1)

                                                # Calculate the middle frame number
                                                middle_frame_number = (first_frame_number + second_frame_number) // 2
                                                video_transitions_frames.setdefault(first_video_name, []).append(middle_frame_number)

                                            else:
                                                print("No match found")

                                            # Associate the frame to the specific video and then count

                                        # Clear the sequences
                                        current_sequence_labels = []

                    i += 1                    
                    pbar.update(1)

                # if (phase == 'valid' or phase == 'test') and epoch > 70:
                #     from sklearn.manifold import TSNE
                #     import matplotlib.pyplot as plt

                #     features_np = np.concatenate(all_features, axis=0)[:, -1, :]  # shape: (N, D)

                #     # Step 1: Reduce features from [B*S, D] to [B*S, 2]
                #     # tsne_input = all_features  # [B*S, 2*hidden_dim]
                #     print(features_np.shape) 
                #     tsne = TSNE(n_components=2, perplexity=2, random_state=42)
                #     tsne_result = tsne.fit_transform(features_np)  # convert to NumPy
                #     plt.figure(figsize=(6, 5))
                #     plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
                #     plt.title("t-SNE of LSTM Features")
                #     plt.xlabel("Dim 1")
                #     plt.ylabel("Dim 2")
                #     plt.grid(True)
                #     plt.show()
                    
                #     import umap
                #     umap_model = umap.UMAP(n_components=2, random_state=42)
                #     # features_np = np.concatenate(all_features, axis=0)  # shape: (N, 512)
                #     print("Features shape for UMAP:", features_np.shape)
                #     umap_results = umap_model.fit_transform(features_np)


                #     # Plot
                #     plt.figure(figsize=(8, 6))
                #     scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=all_labels, cmap='coolwarm', s=60)
                #     plt.colorbar(scatter, label='Transition Label (1=Yes, 0=No)')
                #     plt.title("UMAP of Sequence Features")
                #     plt.xlabel("UMAP 1")
                #     plt.ylabel("UMAP 2")
                #     plt.grid(True)

                    # for i, (x, y) in enumerate(umap_results):
                    #     plt.annotate(str(i), (x + 0.1, y + 0.1), fontsize=8)
                    # plt.show()

                # Best Valid Epoch to run test against
                if phase == 'valid' and (epoch_loss/total_sample_sequences) < best_loss:
                    best_loss = epoch_loss / total_sample_sequences
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(params['save_dir'], f'{params["name"]}.pth'))
                    print(f"Best model weights updated @ Epoch {epoch} and saved to {params['save_dir']}/{params['name']}.pth")

                print(f"Avg Epoch {phase} {epoch} Loss", epoch_loss / total_sample_sequences)
                print(f"Avg Epoch {phase} {epoch} 0s Loss", epoch_loss_0s / total_sample_sequences_0s)
                print(f"Avg Epoch {phase} {epoch} 1s Loss", epoch_loss_1s / total_sample_sequences_1s)
                accuracy = total_correct_sequences / total_sample_sequences
                print(f"{phase.capitalize()} Accuracy: {accuracy * 100:.2f}%")
                print(f"{phase.capitalize()} Total Correct Sequences: {total_correct_sequences} out of {total_sample_sequences}")
                print(f"{phase.capitalize()} Total Correct 0s: {total_correct_sequences_0s} out of {total_sample_sequences_0s}")
                print(f"{phase.capitalize()} Total Correct 1s: {total_correct_sequences_1s} out of {total_sample_sequences_1s}")
                print(f"***SEQUENCE DATA EPOCH {epoch} {phase}***")
                print("# of Predicted Sequences:\t", predicted_sequences_count)
                print("# of Truth Sequences:\t\t", truth_sequence_count)
                # [p:a] - [0:1] does not work; only [1:1]
                print("Predicted Correct Transitions: ", correct_transition_count)
                print(f"***SEQUENCE DATA EPOCH {epoch} {phase}***")

                # Note about statistics
                """
                predicted_sequences_count: Number of sequences that were predicted as transitions (1)
                truth_sequence_count: Number of sequences that were actually transitions (1)
                correct_transition_count: Number of sequences that were correctly predicted as transitions (1)
                """

                print('')
                
                print(f"***FRAME DATA EPOCH {epoch} {phase}***")
                print(dataloaders[phase].dataset.transition_frames)
                total_unmatched_truths = 0
                total_unmatched_preds = 0
                for video_name, truth_frames in dataloaders[phase].dataset.transition_frames.items():
                    truths = truth_frames
                    preds = video_transitions_frames.get(video_name, [])
                    matched_pairs, unmatched_truths, unmatched_preds = match_predictions_to_truths(truths, preds)
                    print('Video: ', video_name)
                    print('Truth Frames: \t', truths)
                    print('Pred Frames: \t', preds)
                    print('Match Frames: \t', matched_pairs)
                    print('Unm Truths: ', unmatched_truths)
                    print('Unm Preds:', unmatched_preds)

                    total_unmatched_truths += len(unmatched_truths)
                    total_unmatched_preds += len(unmatched_preds)
                # print("# of Predicted Frames: ", transition_count)
                # print("# of Truth Frames: ", dataloaders[phase].dataset.transition_frames)
                # print("Video Transitions Frames: ", video_transitions_frames)
                print('# of Unmatched Truth Frames: ', total_unmatched_truths)
                print('# of Unmatched Pred Frames: ', total_unmatched_preds)
                distances = [abs(t - p) for t, p in matched_pairs]

                # Compute statistics
                if len(distances) > 0:
                    avg_distance = mean(distances)
                    min_distance = min(distances)
                    max_distance = max(distances)

                    print(f"Average distance: {avg_distance}")
                    print(f"Minimum distance: {min_distance}")
                    print(f"Maximum distance: {max_distance}")
                if len(distances) > 1:
                    std_dev_distance = stdev(distances)
                    print(f"Standard deviation: {std_dev_distance}")
                else: 
                    print("Standard deviation: Not enough data to compute")
                print(f"***FRAME DATA EPOCH {epoch} {phase}***")

            scheduler.step()

            print(f"PHASE {phase} DONE")
            print()

        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val loss: {best_loss:.4f} at epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    # experiment.end()
    return model


# Leniency of 30 frames (or 1 second) is given to best match
def match_predictions_to_truths(truths, preds, max_distance=30):
    truths = sorted(truths)
    preds = sorted(preds)
    matched_truths = set()
    matched_preds = set()
    matched_pairs = []

    for pred in preds:
        best_match = None
        best_distance = float('inf')
        for i, truth in enumerate(truths):
            if i in matched_truths:
                continue  # already matched
            distance = abs(pred - truth)
            if distance <= max_distance and distance < best_distance:
                best_match = i
                best_distance = distance
        if best_match is not None:
            matched_truths.add(best_match)
            matched_preds.add(pred)
            matched_pairs.append((truths[best_match], pred))

    unmatched_truths = [truths[i] for i in range(len(truths)) if i not in matched_truths]
    unmatched_preds = [pred for pred in preds if pred not in matched_preds]

    return matched_pairs, unmatched_truths, unmatched_preds

# 1. Run Experiment
# 2. Get Model from checkpoint
# 3. Place those 15 specific images into TSNE for the model.