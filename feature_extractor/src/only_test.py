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

def get_class_weights(dataset):
    """ Compute class weights for handling imbalance """
    class_counts = Counter(dataset.targets)  # Count occurrences of each class
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())

    # Compute weights: inverse frequency
    weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}

    # Convert to tensor
    weight_tensor = torch.tensor([weights[i] for i in range(num_classes)], dtype=torch.float)

    print(f"Class Weights: {weight_tensor}")
    return weight_tensor

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
    dataset = {
        'train': datasets.ImageFolder(root=params['train_dir'], transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=params['valid_dir'], transform=image_transforms['valid']),
        'test': datasets.ImageFolder(root=params['test_dir'], transform=image_transforms['test'])
    }
    
    def remap_labels(dataset, old_label, new_label):
        dataset.targets = [new_label if label == old_label else label for label in dataset.targets]
        dataset.samples = [(path, new_label if label == old_label else label) for path, label in dataset.samples]

    # Remap labels for binary classification
    unique_train_labels = set(dataset['train'].targets)
    unique_valid_labels = set(dataset['valid'].targets)
    unique_test_labels = set(dataset['test'].targets)

    print(f"Unique labels in train dataset: {unique_train_labels}")
    print(f"Unique labels in valid dataset: {unique_valid_labels}")
    print(f"Unique labels in test dataset: {unique_test_labels}")
    remap_labels(dataset['train'], old_label=3, new_label=1)
    remap_labels(dataset['valid'], old_label=3, new_label=1)
    remap_labels(dataset['test'], old_label=3, new_label=1)

    unique_train_labels = set(dataset['train'].targets)
    unique_valid_labels = set(dataset['valid'].targets)
    unique_test_labels = set(dataset['test'].targets)

    print(f"Unique labels in train dataset: {unique_train_labels}")
    print(f"Unique labels in valid dataset: {unique_valid_labels}")
    print(f"Unique labels in test dataset: {unique_test_labels}")
    
    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        'valid': len(dataset['valid']),
        'test': len(dataset['test'])
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=hyper_params['batch_size'], shuffle=True,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'valid': data.DataLoader(dataset['valid'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'test': data.DataLoader(dataset['test'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_ft = torch.load("/home/user/Documents/GitHub/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth")
    activation = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_ft.parameters(), lr=hyper_params['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params['num_epochs'], eta_min=hyper_params['learning_rate']/100.)

    return model_ft, dataloaders, dataset_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list


def test_model(params, hyper_params):
    since = time.time()

    model, dataloaders, data_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list = load_train(params, hyper_params)
    model.to(device)
    print(f"Train: {next(model.parameters()).device}")

    train_weights = get_class_weights(dataloaders['train'].dataset).to(device)
    valid_weights = get_class_weights(dataloaders['valid'].dataset).to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    logger = open(os.path.join(params['save_dir'], params['name'] + '.txt'), "w")

    for epoch in range(hyper_params['num_epochs']):
        print('Epoch {}/{}'.format(epoch, hyper_params['num_epochs'] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test']:
            criterion = nn.CrossEntropyLoss(weight=train_weights if phase == 'train' else valid_weights)
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
                    inputs = inputs.to(device, non_blocking=True) # tensor(2.6400) tensor(-2.1179)
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
            print("Overall Loss:", epoch_loss)
            print("Overall Acc:", epoch_acc)
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

        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # experiment.end()
    return model