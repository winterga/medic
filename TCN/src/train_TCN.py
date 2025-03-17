import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
import torch.utils.data as data
from torch.utils.data import Subset
import torch.optim as optim

import time, os, copy
from .resnet import resnet50
import tqdm
import random

from .TCN import TCN, TCNWrapper
from .VideoSegmentDataset import VideoSegmentDataset

import matplotlib
matplotlib.use('Agg')  # So we can save plots without X-server
import matplotlib.pyplot as plt


def get_subset_indices(dataset, subset_percentage=0.1):
    """
    Get indices for a subset of the dataset (e.g., 10% of the data).
    """
    indices = list(range(len(dataset)))
    subset_size = int(len(indices) * subset_percentage)
    subset_indices = random.sample(indices, subset_size)
    return subset_indices


def apply_gpu_transforms(image, transform_list):
    """
    If you want to apply GPU-based transforms.
    Currently empty by default.
    """
    for t in transform_list:
        image = t(image)
    return image


def load_train(params, hyper_params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Basic CPU transforms
    cpu_train_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]
    cpu_valid_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]

    # Add normalization if desired
    if hyper_params['normalize']:
        cpu_train_list.append(
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        cpu_valid_list.append(
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    # Compose transforms
    train_transform = v2.Compose(cpu_train_list)
    valid_transform = v2.Compose(cpu_valid_list)

    # Load the custom VideoSegmentDataset
    train_dataset = VideoSegmentDataset(
        root_dir=params['train_dir'],
        seq_length=hyper_params['seq_length'],
        sliding_step=1,
        transform=train_transform
    )
    valid_dataset = VideoSegmentDataset(
        root_dir=params['valid_dir'],
        seq_length=hyper_params['seq_length'],
        sliding_step=1,
        transform=valid_transform
    )

    dataset_sizes = {
        'train': len(train_dataset),
        'valid': len(valid_dataset)
    }

    # If you want to only use a subset (e.g., 10%), do it here:
    subset_percentage = 0.1
    train_subset_indices = get_subset_indices(train_dataset, subset_percentage)
    valid_subset_indices = get_subset_indices(valid_dataset, subset_percentage)

    train_subset = Subset(train_dataset, train_subset_indices)
    valid_subset = Subset(valid_dataset, valid_subset_indices)

    # Create DataLoaders
    dataloaders = {
        'train': data.DataLoader(train_subset,
                                 batch_size=hyper_params['batch_size'],
                                 shuffle=True,
                                 num_workers=hyper_params['cpu_count'],
                                 pin_memory=True,
                                 drop_last=True),
        'valid': data.DataLoader(valid_subset,
                                 batch_size=hyper_params['batch_size'],
                                 shuffle=False,
                                 num_workers=hyper_params['cpu_count'],
                                 pin_memory=True,
                                 drop_last=True)
    }

    # Load your feature extractor (ResNet-50)
    feature_extractor_path = '/home-local/winterga/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth'
    feature_extractor = torch.load(feature_extractor_path,
                                   map_location=device, weights_only=False)
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1]) # remove last FC layer so it outputs (N, 2048, 1, 1)

    # Build TCN with 2048 input channels (ResNet50 outputs)
    tcn_model = TCN(num_inputs=2048,
                    num_channels=[16, 32, 64],
                    kernel_size=3,
                    dropout=0.1)

    # Wrap with TCNWrapper -> yields final (N, 3)
    model = TCNWrapper(tcn_model,
                       FE_model=feature_extractor,
                       input_shape='NLC',
                       num_classes=hyper_params['n_classes']).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=hyper_params['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=hyper_params['num_epochs'],
                                                     eta_min=hyper_params['learning_rate']/100.)

    return model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device


def train_model(params, hyper_params):
    since = time.time()

    model, dataloaders, data_sizes, criterion, optimizer, scheduler, device = load_train(params, hyper_params)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # We will store losses for plotting
    train_loss_history = []
    val_loss_history = []

    num_epochs = hyper_params['num_epochs']

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            with tqdm.tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch+1}', unit='batch') as pbar:
                for inputs, labels in dataloaders[phase]:
                    # inputs => (N, L, C, H, W)
                    # labels => (N,) for sequence-level
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)  # => (N, 3)
                        _, preds = torch.max(outputs, dim=1)  # class indices
                        
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.update(1)

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                scheduler.step()  # step LR scheduler only after training phase
                print(f'Train Loss: {epoch_loss:.4f}  |  Acc: {epoch_acc:.4f}')
            else:
                val_loss_history.append(epoch_loss)
                print(f'Valid Loss: {epoch_loss:.4f}  |  Acc: {epoch_acc:.4f}')
                # track best
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(params['save_dir'], f'{params["name"]}_best.pth'))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')

    # load best weights
    model.load_state_dict(best_model_wts)

    # Save final model
    torch.save(model.state_dict(), os.path.join(params['save_dir'], f'{params["name"]}_final.pth'))

    # ----------- Plot training vs. validation loss -----------
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure to checkpoints folder
    loss_plot_path = os.path.join(params['save_dir'], f'{params["name"]}_loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")

    return model
