from .conv import CausalConv1d
import torch.nn as nn
import torch
from numpy.typing import ArrayLike
from .resnet import resnet50

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# TCN Implementation

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, activation=nn.ReLU()):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        with torch.no_grad():
            res = self.residual(x) # shortcut connection
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        return x + res # add residual connection (skip forward for previous layer)
    
    
class TCN(nn.Module):
    def __init__(self, 
                 num_inputs: int, 
                 num_channels: ArrayLike, 
                 kernel_size: int, 
                 dropout: float = 0.1):
        super().__init__() 
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i # exponential increase in dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
            
            
    def forward(self, x):
        print(f"Allocated memory2: {torch.cuda.memory_allocated() / 1024**2} MB")
        print(f"Cached memory2: {torch.cuda.memory_reserved() / 1024**2} MB")
        return self.network(x)
            
            
            
class TCNWrapper(nn.Module):
    def __init__(self, tcn: TCN, FE_model, input_shape='NCL'):
        super().__init__()
        self.tcn = tcn
        self.input_shape = input_shape

        # Load ResNet-50 as feature extractor
        self.feature_extractor = FE_model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze feature extractor parameters
        
    def forward(self, x):
        """
        x: Tensor of shape (batch, frames, channels, height, width) -> (N, L, C, H, W)
        """

        print(x.shape)
        batch_size, seq_len, c, h, w = x.shape  # Extract batch and sequence dimensions
        
        # Flatten batch and time dimension for ResNet processing
        x = x.view(batch_size * seq_len, c, h, w)  # Reshape to (N*L, C, H, W)
        x = self.feature_extractor(x)  # Pass through ResNet-50 -> Outputs (N*L, feature_dim)
        
        # Reshape back to (N, L, feature_dim)
        feature_dim = x.shape[-1]  # Get extracted feature dimension
        x = x.view(batch_size, seq_len, feature_dim)  # Reshape to (N, L, feature_dim)
        
        if self.input_shape == 'NLC':
            x = x.transpose(1,2)  # Convert to (N, C, L) format for TCN
        
        print(f"Feature extractor output shape: {x.shape}")
        x = self.tcn(x)  # Pass features through TCN
        
        return x
        
    

# Test run
if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")
    
    # Load feature extractor
    # resnet_model_path = '/home/local/VANDERBILT/winterga/medic/feature_extractor/checkpoints/Resnet50_021225_07/Resnet50_021225_07.pth'
    resnet_model_path = '/home/local/VANDERBILT/winterga/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth'
    FE = torch.load(f=resnet_model_path, map_location=device, weights_only=False)


    
    # Define TCN
    tcn = TCN(num_inputs=3, num_channels=[16, 32, 64], kernel_size=3, dropout=0.1)
    model = TCNWrapper(tcn, FE, input_shape='NLC')
    model=model.to(device)
    print(f"TCN Parameters ({len(list(model.parameters()))} parameters):")
    for name, param in model.named_parameters():
        print(name)
    # print(f"TCN parameters: {model.parameters()}")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")
    
    # # Sample input
    # batch_size = 4
    # sequence_length = 50
    # input_channels = 3
    # image_height=256
    # image_width=256
    # x = torch.randn(batch_size, sequence_length, input_channels, image_height, image_width) # (N, L, C, H, W)
    # x=x.to(device)
    # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
    # print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")
    
    # # Forward pass
    # output = model(x) # Will get passed to FE with shape (N*L, C, H, W)
    # print(output.shape) # Expected output: torch.Size([8, 64, 50])
    