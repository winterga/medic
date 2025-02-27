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
from .TCN import VideoSegmenetDataset

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

    # GPU-based transforms for heavy augmentations
    gpu_train_list = []
    
    if hyper_params['normalize']:
        cpu_train_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]))
        cpu_valid_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]))
    image_transforms = {
        'train': v2.Compose(cpu_train_list),
        'valid': v2.Compose(cpu_valid_list)
    }
    # Load data from folders
    dataset = {
        'train': VideoSegmentDataset(root_dir=params['train_dir'], seq_length=hyper_params['seq_length'], transform=video_transform),
        'valid': VideoSegmentDataset(root_dir=params['valid_dir'], seq_length=hyper_params['seq_length'], transform=video_transform),
    }
    
    def remap_labels(dataset, old_label, new_label):
        dataset.targets = [new_label if label == old_label else label for label in dataset.targets]
        dataset.samples = [(path, new_label if label == old_label else label) for path, label in dataset.samples]

    # Remap labels for binary classification
    unique_train_labels = set(dataset['train'].targets)
    unique_valid_labels = set(dataset['valid'].targets)

    # print(f"Unique labels in train dataset: {unique_train_labels}") # print out old labels
    # print(f"Unique labels in valid dataset: {unique_valid_labels}") # print out old labels
    # remap_labels(dataset['train'], old_label=3, new_label=1) # remap labels
    # remap_labels(dataset['valid'], old_label=3, new_label=1)

    # unique_train_labels = set(dataset['train'].targets) # get new labels
    # unique_valid_labels = set(dataset['valid'].targets)

    # print(f"Unique labels in train dataset: {unique_train_labels}") # print new labels
    # print(f"Unique labels in valid dataset: {unique_valid_labels}")
    
    subset_percentage = .1
    # Get subset indices for each dataset (10% of each class)
    subset_indices_train = get_subset_indices(dataset['train'], subset_percentage)
    subset_indices_valid = get_subset_indices(dataset['valid'], subset_percentage)
    
    # Create subsets for each data split
    train_subset = Subset(dataset['train'], subset_indices_train)
    valid_subset = Subset(dataset['valid'], subset_indices_valid)
    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        'valid': len(dataset['valid'])
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(train_subset, batch_size=hyper_params['batch_size'], shuffle=True,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'valid': data.DataLoader(valid_subset, batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load feature extractor
    feature_extractor = torch.load('/home/local/VANDERBILT/winterga/medic/feature_extractor/checkpoints/Resnet50_021225_07/Resnet50_021225_07.pth', map_location=device)
        
    # Set up TCN model for training
    tcn_model = TCN(num_inputs=3, num_channels=[16, 32, 64], kernel_size=3, dropout=0.1)
    model = TCNWrapper(tcn_model, feature_extractor, input_shape='NLC')  # Wrap TCN with ResNet feature extractor
    model.to(device) # Move model to GPU
    
    activation = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    
    # Ensure only TCN-specific parameters get updated during training
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=hyper_params['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params['num_epochs'], eta_min=hyper_params['learning_rate']/100.)

    return model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list


def train_model(params, hyper_params):
    since = time.time()

    model, dataloaders, data_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list = load_train(params, hyper_params)
    model.to(device)
    print(f"Train: {next(model.parameters()).device}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc = 0.0


    logger = open(os.path.join(params['save_dir'], params['name'] + '.txt'), "w")

    for epoch in range(hyper_params['num_epochs']):
        print('Epoch {}/{}'.format(epoch, hyper_params['num_epochs'] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            class_corrects = {i: 0 for i in range(3)}
            class_totals = {i: 0 for i in range(3)}

            with tqdm.tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch}', unit='batch') as pbar:

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, non_blocking=True)
                    inputs = apply_gpu_transforms(inputs, gpu_train_list)
                    labels = labels.to(device, non_blocking=True)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)  # Model outputs logits, not probabilities
                        _, preds = torch.max(outputs, 1)  # Get the index of the max logit (class prediction)
                        
                        # Ensure labels are of the correct shape
                        labels = labels.long()  # CrossEntropyLoss expects labels as integers
                                        
                        # Calculate loss
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pred_labels = preds.byte().cpu().numpy()
                    labels = labels.cpu().numpy()

                    for i in range(len(labels)):
                        label = labels[i].item()
                        class_totals[label] += 1
                        if preds[i] == labels[i]:
                            class_corrects[label] += 1

                    pbar.update(1)

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            per_class_accuracy = {cls: (class_corrects[cls] / class_totals[cls]) * 100 if class_totals[cls] > 0 else 0 
                                  for cls in range(3)}  # Assuming 4 classes

            print(f"\nPer-Class Accuracy:")
            for cls, accuracy in per_class_accuracy.items():
                print(f"Class {cls}: {accuracy:.2f}%")

            # deep copy the model
            if phase == 'train':
                print(f'Training Loss: {epoch_loss}')
                print(f'Training Acc: {epoch_acc}')
            if phase == 'valid':
                print(f"Validate Acc: {epoch_acc}")
                print(f"Validate Loss: {epoch_loss}")
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, os.path.join(params['save_dir'], f'{params["name"]}.pth'))

                if epoch_loss > best_acc:
                    best_acc = epoch_loss

        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logger = open(os.path.join(params['save_dir'], params['name'] + '.txt'), "w")
    logger.write('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.write('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model