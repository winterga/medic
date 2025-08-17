# from comet_ml import Experiment

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import yaml
import torch
import numpy as np
import random

from datetime import datetime

from .train import train_model
import glob
import argparse
import multiprocessing
import warnings

warnings.filterwarnings("ignore")

def create_exp_name(params, hyperparams):
    # update names
    now = datetime.now()
    base_name = f'{hyperparams["architecture"]}_{now.strftime("%m%d%y")}'
    existing_dirs = glob.glob(os.path.join(params['save_dir'],base_name+'*'+os.path.sep))  # search for existing dirs withthis name
    if len(existing_dirs) >=1:
        last_dir = sorted(existing_dirs)[-1]
        next_count = int(last_dir[-3:-1]) + 1
    else:
        next_count=1
    params['name'] = f'{base_name}_{next_count:02}'
    params['save_dir'] = os.path.join(params['save_dir'], params['name'])
    print(params['save_dir'])
    os.makedirs(params['save_dir'], exist_ok=True)
    return params, hyperparams

import os

def set_data_directories(base_dir):
    """
    Function to set directories for train, validation, and test.
    The base directory should contain subdirectories: 'train', 'val', 'test'.
    """
    print(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"The base directory {base_dir} does not exist.")
    
    # Set the directories based on the base directory
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Check if the subdirectories exist
    if not os.path.isdir(train_dir):
        raise ValueError(f"The train directory {train_dir} does not exist.")
    if not os.path.isdir(valid_dir):
        raise ValueError(f"The validation directory {valid_dir} does not exist.")
    if not os.path.isdir(test_dir):
        raise ValueError(f"The test directory {test_dir} does not exist.")
    
    return {
        'train_dir': train_dir,
        'valid_dir': valid_dir,
        'test_dir': test_dir
    }


def run_experiment(params, hyper_params):
    torch.cuda.empty_cache()
    params, hyper_params = create_exp_name(params, hyper_params)
    params.update(set_data_directories(params['base_dir']))


    train_model(params, hyper_params)
    # hyper_params['batch_size'] =1
    # test_model(params, hyper_params)

def set_manual_seed(seed_value=42):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # May slow down training a bit, but makes sure results are reproducible
    torch.use_deterministic_algorithms(True, warn_only=True)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser(description='Run experiment with different hyperparameters.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the optimizer')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training/validation iterations before testing')
    parser.add_argument('--t_max', type=int, default=150, help='Value for learning rate scheduler -- likely to equal # of epochs but could be different if running experiments with diff epochs')
    parser.add_argument('--manual_seed', type=int, default=42, help='Seed Value')
    parser.add_argument('--patience', type=int, default=4, help='Number of epochs that must find a better validation accuracy for or the model stops training')
    parser.add_argument('--k_fold', type=int, default=1, help='Number of folds for the data to be split into')
    parser.add_argument('--n_channels', type=int, default=3, help='Number of channels; 3 for normal images - 4 for using the segmentation model')
    parser.add_argument('--rotate', action='store_true', help='Enable rotation augmentation')
    parser.add_argument('--flip', action='store_true', help='Enable horizontal flip augmentation')
    parser.add_argument('--color_jitter', action='store_true', help='Enable color jitter augmentation')
    parser.add_argument('--blur', action='store_true', help='Enable Gaussian blur augmentation')
    parser.add_argument('--posterize', action='store_true', help='Enable posterization augmentation')
    parser.add_argument('--sharpness', action='store_true', help='Enable sharpness adjustment augmentation')
    parser.add_argument('--perspective', action='store_true', help='Enable perspective transformation augmentation')
    parser.add_argument('--elastic', action='store_true', help='Enable elastic transformation augmentation')
    parser.add_argument('--grayscale', action='store_true', help='Enable grayscale augmentation')
    parser.add_argument('--architecture', type=str, default='Resnet50', help='Architecture to use for the model')
    parser.add_argument('--cpu_count', type=int, default=num_cores+1, help='Number of CPU cores to use')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use a pretrained model')
    parser.add_argument('--num_augs', type=int, default=0, help='Number of augmentations to use')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing train, val, and test subdirectories')
    parser.add_argument('--embedding_size', type=int, default=2, help='Size of the embedding layer')
    args = parser.parse_args()
    print(f'Arch: {args.architecture}')
    print(f'Pretrained: {args.pretrained}')

    print(f'********num_cores is {num_cores}********')
    # Set directories dynamically based on the base directory
    data_dirs = set_data_directories(args.base_dir)
    hyper_params = {
        "learning_rate": args.learning_rate,  # Get from command-line argument
        "weight_decay": args.weight_decay,    # Get from command-line argument
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "t_max": args.t_max,
        "img_size": 512,
        'criterion': 'BCEWithLogits',
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "architecture": args.architecture,
        'n_channels': args.n_channels,
        'n_classes': 1,
        'cpu_count': args.cpu_count-5,
        'activation': 'Sigmoid',
        'augmentation': 'bestaug',
        'normalize':True,
        'manual_seed': args.manual_seed,
        'patience': args.patience, 
        'k_fold': args.k_fold,
        'rotate': args.rotate,
        'flip': args.flip,
        'color_jitter': args.color_jitter,
        'blur': args.blur,
        'posterize': args.posterize,
        'sharpness': args.sharpness,
        'perspective': args.perspective,
        'elastic': args.elastic,
        'grayscale': args.grayscale,
        'pretrained': args.pretrained,
        'num_augs': args.num_augs,
        'embedding_size': args.embedding_size,
    }
    params = {
        'train_dir': data_dirs['train_dir'],
        'valid_dir': data_dirs['valid_dir'],
        'test_dir': data_dirs['test_dir'],
        'save_dir': 'checkpoints',
        'seg_model_path': 'models/unet_residual_prediction.pth',
        'vis_seg': False,
        'base_dir': args.base_dir,
    }
    # config = yaml.safe_load(open('config.yaml'))

    # Set a manual seed here
    set_manual_seed(hyper_params['manual_seed'])
    # torch.backends.cudnn.benchmark = True
    
    run_experiment(params, hyper_params)
