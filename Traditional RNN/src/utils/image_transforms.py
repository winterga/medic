import torchvision.transforms as v2
import torch
import random

# Define transformation constants using only torchvision
RESIZE = v2.Resize
TO_TENSOR = v2.ToTensor
RANDOM_ROTATION = v2.RandomRotation
RANDOM_HORIZONTAL_FLIP = v2.RandomHorizontalFlip
RANDOM_VERTICAL_FLIP = v2.RandomVerticalFlip
RANDOM_RESIZED_CROP = v2.RandomResizedCrop
RANDOM_AFFINE = v2.RandomAffine
COLOR_JITTER = v2.ColorJitter
GAUSSIAN_BLUR = v2.GaussianBlur
RANDOM_POSTERIZE = v2.RandomPosterize
RANDOM_ADJUST_SHARPNESS = v2.RandomAdjustSharpness
RANDOM_PERSPECTIVE = v2.RandomPerspective
ELASTIC_TRANSFORM = v2.ElasticTransform
RANDOM_ERASING = v2.RandomErasing
NORMALIZE = v2.Normalize

# Define RANDOM_APPLY to apply any of the defined transformations randomly
def RANDOM_APPLY(transforms, p=0.5):
    return v2.RandomApply(transforms, p=p)

# Function to apply a custom list of transformations
def apply_transforms(image, transforms):
    """
    Apply a list of transformations to an image.

    Args:
        image (PIL.Image or Tensor): The image to transform.
        transforms (list): A list of transformations to apply.

    Returns:
        Transformed image.
    """
    composed_transforms = v2.Compose(transforms)
    return composed_transforms(image)

# New function to apply AutoAugment with ImageNet policy
def apply_autoaugment_transforms(image):
    """
    Apply AutoAugment using ImageNet policy to an image.

    Args:
        image (Tensor): The image to transform (expected to be torch.float32).

    Returns:
        Transformed image (back to torch.float32).
    """
    transform = v2.transforms.Compose([
        v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),  # or any other policy
        v2.transforms.ToTensor(),
        v2.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_image = transform(image)

    # Apply transformations (Normalization after converting back to torch.float32 -- AutoAugment requires uint8)
    # transformed_image = autoaugment_transforms(image_uint8).to(torch.float32) / 255.0
    # transformed_image = NORMALIZE(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transformed_image)

    # Convert back to float32 and scale to [0, 1]
    return transformed_image

def apply_customaugment_transforms(image):
    transform = v2.transforms.Compose([
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.))], p=0.4),  # Blur 40% of the time
        v2.RandomApply([v2.RandomPosterize(bits=4)], p=0.3),  # Posterize 30% of the time
        v2.RandomApply([v2.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),  # Sharpness 30% of the time
        v2.RandomApply([v2.RandomRotation(degrees=[0, 180])], p=0.5),  # Rotation of 0 or 180 degrees, 50% of the time
        v2.transforms.ToTensor(),
        v2.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_image = transform(image)

    # Apply transformations (Normalization after converting back to torch.float32 -- AutoAugment requires uint8)
    # transformed_image = autoaugment_transforms(image_uint8).to(torch.float32) / 255.0
    # transformed_image = NORMALIZE(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(transformed_image)

    # Convert back to float32 and scale to [0, 1]
    return transformed_image

# List of augmentation operations that can be applied
RAND_AUGMENTS = [
    v2.RandomRotation(degrees=30),               # Random rotation between -30 and +30 degrees
    v2.RandomHorizontalFlip(p=0.5),              # Random horizontal flip with 50% probability
    v2.ColorJitter(brightness=0.2),               # Random brightness adjustment
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine (translate)
    v2.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),  # Random perspective
    v2.RandomVerticalFlip(p=0.5),                # Random vertical flip with 50% probability
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)), # Random resized crop
    v2.RandomAdjustSharpness(sharpness_factor=2), # Random sharpness adjustment
    v2.RandomPosterize(bits=4),                   # Random posterization
    v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Gaussian blur
]

# Function to apply n random augmentations from the list
# Small Values (m = 1 to 2): For small, subtle transformations.
# Medium Values (m = 3 to 5): For moderate transformations, where the augmentation is visible but not too aggressive.
# Large Values (m = 6 to 10): For strong transformations, where the image may undergo significant changes.
def apply_randaugment_transforms(image, n=1, m=1):
    selected_transforms = random.sample(RAND_AUGMENTS, n)
    
    # Check if image has 4 dimensions (e.g., [1, C, H, W]) and squeeze it to 3D
    if image.ndimension() == 4:
        image = image.squeeze(0)

    # Convert tensor to PIL image
    pil_image = v2.ToPILImage()(image)
    
    for transform in selected_transforms:
        if isinstance(transform, v2.ColorJitter):
            # Apply magnitude adjustment for color jitter
            transform = v2.ColorJitter(
                brightness=m * 0.1, contrast=m * 0.1, saturation=m * 0.1, hue=m * 0.1
            )
        elif isinstance(transform, v2.RandomRotation):
            # Apply magnitude adjustment for rotation
            transform = v2.RandomRotation(degrees=m * 15)
        elif isinstance(transform, v2.RandomAffine):
            # Apply magnitude adjustment for affine (translate, scale)
            transform = v2.RandomAffine(degrees=m * 10, translate=(m * 0.05, m * 0.05))
        
        # Apply the transformation to the PIL image
        pil_image = transform(pil_image)
    
    # Convert back to tensor if needed
    image = v2.ToTensor()(pil_image)
    return image

