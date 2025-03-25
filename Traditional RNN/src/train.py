import time
import torch
from torchvision.transforms import Compose
from torch.utils import data
import functools
import tqdm
import numpy as np
import random
from .image_transforms import apply_transforms, apply_customaugment_transforms, apply_autoaugment_transforms, RESIZE, TO_TENSOR, RANDOM_ROTATION, RANDOM_HORIZONTAL_FLIP, NORMALIZE, COLOR_JITTER, GAUSSIAN_BLUR
from torchvision.models import resnet50
from .rnn_dataset import RNNImageDataset as RNNDataset
import torch.optim as optim
from torchvision import transforms as v2
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
import json
import torch
import torch.nn as nn
import pprint

def init_weights(m):
    """Custom weight initialization function."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)

    elif isinstance(m, nn.TransformerEncoderLayer):
        # Initialize feedforward network inside Transformer
        for param in [m.linear1.weight, m.linear2.weight]:
            nn.init.xavier_normal_(param)
        nn.init.zeros_(m.linear1.bias)
        nn.init.zeros_(m.linear2.bias)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(CNNFeatureExtractor, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Removing the last classification layer

    def forward(self, x):
        features = self.base_model(x)
        return features.view(features.size(0), -1)  # Flatten the output
    
class CNNFeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(CNNFeatureExtractor, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  

        # Freeze the feature extractor
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        features = self.base_model(x)
        return features.view(features.size(0), -1)

class CNN_LSTM_SequenceModel(nn.Module):
    def __init__(self, cnn_model, hidden_size=512, num_classes=2, num_layers=2):
        super(CNN_LSTM_SequenceModel, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor(cnn_model)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape  
        x = x.view(batch_size * seq_len, c, h, w)  
        cnn_out = self.cnn_extractor(x)  
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  
        lstm_out, _ = self.lstm(cnn_out)
        # print(lstm_out)
        logits = self.fc(lstm_out)  
        return logits

def custom_collate_fn(batch):
    """
    Custom collate function for batching sequences of images and their corresponding labels.
    
    Args:
    - batch: List of tuples (image_sequence, label) where image_sequence is a tensor
      and label is the corresponding label.
    
    Returns:
    - A tuple containing:
      1. image_sequences: A tensor of shape [batch_size, seq_len, channels, height, width].
      2. labels: A tensor of shape [batch_size].
    """
    # Separate image sequences and labels
    image_sequences, labels, paths = zip(*batch)
    
    # Stack image sequences into a single tensor of shape [batch_size, seq_len, channels, height, width]
    image_sequences = torch.stack(image_sequences, dim=0)
    
    # Stack labels into a tensor of shape [batch_size]
    labels = torch.tensor(labels)
    
    return image_sequences, labels, paths

def all_from_same_video(file_list):
    video_ids = {path.split('/')[-1].split('_frame')[0] for path in file_list}

    # Extract the frame numbers from the file paths
    frame_numbers = sorted([int(path.split('_frame_')[-1].split('.')[0]) for path in file_list])
    
    # Check if the list of frame numbers is continuous (no gaps)
    for i in range(1, len(frame_numbers)):
        if frame_numbers[i] != frame_numbers[i - 1] + 1:
            return False  # If there's a gap between frames, return False
    
    # Check if all files belong to the same video
    video_ids = {path.split('/')[-1].split('_frame')[0] for path in file_list}
    return len(video_ids) == 1

def seed_worker(seed, worker_id):
    torch.manual_seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.cuda.manual_seed(seed + worker_id)
    torch.cuda.manual_seed_all(seed + worker_id)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # May slow down training a bit, but makes sure results are reproducible
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_train(params, hyper_params):
    print("Loading train")
    # Define the CPU transformations
    cpu_transforms = [
        RESIZE(size=hyper_params['img_size']),
        TO_TENSOR(),
        NORMALIZE(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    print("Train Dir:", params['train_dir'])
    print("Valid Dir:", params['valid_dir'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_ft = CNN_LSTM_SequenceModel(resnet50())
    model_ft = CNN_LSTM_SequenceModel(torch.load('/home/user/Documents/GitHub/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth'))
    activation = nn.Sigmoid()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model_ft.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params['t_max'], eta_min=hyper_params['learning_rate']/100.)

    dataset = {
        'train': RNNDataset(params['train_dir'], device, cpu_transforms),
        'valid': RNNDataset(params['valid_dir'], device, cpu_transforms),
        'test': RNNDataset(params['test_dir'], device, cpu_transforms)
    }
    dataset_sizes = {
        'train': len(dataset['train']),
        'valid': len(dataset['valid']),
        'test': len(dataset['test'])
    }
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=1, shuffle=False,
                                 num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True, 
                                 prefetch_factor=4, worker_init_fn=functools.partial(seed_worker, 11111), persistent_workers=True, collate_fn=custom_collate_fn),
        'valid': data.DataLoader(dataset['valid'], batch_size=1, shuffle=False,
                                 num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True, 
                                 prefetch_factor=4, worker_init_fn=functools.partial(seed_worker, 11111), persistent_workers=True, collate_fn=custom_collate_fn),
        'test': data.DataLoader(dataset['test'], batch_size=1, shuffle=False,
                                 num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True, 
                                 prefetch_factor=4, worker_init_fn=functools.partial(seed_worker, 11111), persistent_workers=True, collate_fn=custom_collate_fn),
    }
    return model_ft, dataloaders, dataset_sizes, criterion, optimizer, scheduler, activation, device

def train_model(params, hyper_params):
    # Early Stopping Variable Declaration
    patience = hyper_params.get('patience')  # Early stopping patience

    # Training Variable Declarationw
    since = time.time()
    [ 
        model,
        dataloaders, 
        data_sizes, 
        criterion, 
        optimizer, 
        scheduler, 
        activation,
        device
    ] = load_train(params, hyper_params)
    
    model.to(device)
    batch_size = 1

    for epoch in range(hyper_params['num_epochs']):
        print('Epoch {}/{}'.format(epoch, hyper_params['num_epochs'] - 1))
        print('-' * 10)

        epoch_loss = 0.0
    

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid', 'test']:
            vids_count = {}
            model.train() if phase == 'train' else model.eval()

            all_preds = []
            all_labels = []
            all_probs = []

            running_corrects = 0
            total_samples = 0

            with tqdm.tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch}', unit='batch') as pbar:
                
                print("Size: ", len(dataloaders[phase].dataset))
                
                vid_count = 0
                vid_count_correct = 0
                count = 0
                mod = 6
                current_label = -1
                total_weighted_avg = 0
                done = False
                for images, labels, paths in dataloaders[phase]:
                    # print(paths)
                    paths = paths[0]
                    if(all_from_same_video(paths)):
                        done = False
                        # if count % mod == 0:
                        # if (phase == 'train' and count % mod == 0) or phase != 'train':
                        images = images.to(device)
                        labels = labels.to(device)
                        # print(labels, paths)
                        optimizer.zero_grad() # Clear gradients from previous batch
                        # logit, weighted_avg, probs = model(images)  # Forward pass
                        logits = model(images)
                        # print(logits, labels)
                        loss = criterion(logits, labels.unsqueeze(1).float())

                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5)
                        print(logits, loss)#, probs, preds, labels, paths)

                        running_corrects += (preds == labels).sum().item()
                        vid_count_correct += (preds == labels).sum().item()
                        vid_count += 1
                        total_samples += labels.size(0)
                        current_label = labels[0].item()
                        # total_weighted_avg += weighted_avg.item()

                        all_preds.extend(preds.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())
                        all_probs.extend(probs.detach().cpu().numpy())
                        # print("Preds", preds)
                        # print("Labels", labels)
                        # print("Probs", probs)
                        # print("WA", weighted_avg)

                        # logit = torch.log(weighted_avg / (1 - weighted_avg)).unsqueeze(0)
                        # loss = criterion(logit, labels.unsqueeze(1).float())
                        epoch_loss += loss.item()

                        if phase == 'train':
                            loss.backward()  # Backward pass
                            optimizer.step()  # Update weights

                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

                    else:
                        print('Not Same paths:', paths)
                        if (not done):
                            video = paths[0].split('/')[-1].split('_frame')[0]
                            print(labels)
                            vids_count[video] = f'{current_label} - {vid_count_correct} - {vid_count} - {vid_count_correct / vid_count} - {total_weighted_avg / vid_count} - {total_weighted_avg}'
                            done = True
                            vid_count = 0
                            vid_count_correct = 0
                            current_label = -1
                            total_weighted_avg = 0
                    # pbar.update(mod if phase == 'train' else 1)
                    pbar.update(1)
                    count += 1
                # count += 1

            pprint.pprint(vids_count, sort_dicts=False)
            epoch_accuracy = running_corrects / total_samples
            epoch_loss /= len(dataloaders[phase])
            print(f"{phase.capitalize()} Epoch {epoch} Accuracy: {epoch_accuracy:.4f}")
            print(f"{phase.capitalize()} Epoch {epoch} Loss: {epoch_loss:.4f}")
            print(f"Running Corrects: {running_corrects} -- Total Samples: {total_samples}")

            try:
                roc_auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                roc_auc = float('nan')  # In case of only one class present

            # Compute Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)

            print(f"{phase.capitalize()} Epoch {epoch} ROC-AUC: {roc_auc:.4f}")

            """
            [[TP FN]
             [FP TN]]
            """
            print("Confusion Matrix:")
            print(cm)

            # Print detailed classification report
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, digits=4))
                    
    return model