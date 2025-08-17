import time
import copy
import torch
import os
from torch.cuda.amp import GradScaler, autocast  # Corrected import statement
from torchvision import datasets
from torchvision.transforms import Compose
from torch.utils import data
import functools
import tqdm
import numpy as np
import random
from .image_transforms import apply_customaugment_transforms, TO_TENSOR
from torchvision import transforms as v2
import matplotlib.pyplot as plt

class TraditionalDataset(torch.utils.data.Dataset):
    def __init__(self, root, device, cpu_transform):
        print("Root", root)
        self.dataset = datasets.ImageFolder(root=root)
        print("Size2:", len(self.dataset))
        self.cpu_transform = Compose(cpu_transform)
        self.to_tensor = TO_TENSOR()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_idx = idx // (self.num_augmentations + 1)
        img, label = self.dataset[img_idx]
        # is_original = (idx % (self.num_augmentations + 1) == 0)
        
        img = self.to_tensor(img)
        return img, label