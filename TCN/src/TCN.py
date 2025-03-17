# from .conv import CausalConv1d
# import torch.nn as nn
# import torch
# from numpy.typing import ArrayLike
# from .resnet import resnet50

# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# # TCN Implementation

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, activation=nn.ReLU()):
#         super().__init__()
#         self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
#         self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
#         self.activation = activation
#         self.dropout = nn.Dropout(dropout)
#         self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
#     def forward(self, x):
#         with torch.no_grad():
#             res = self.residual(x) # shortcut connection
#         x = self.activation(self.conv1(x))
#         x = self.dropout(x)
#         x = self.activation(self.conv2(x))
#         x = self.dropout(x)
#         return x + res # add residual connection (skip forward for previous layer)
    
    
# class TCN(nn.Module):
#     def __init__(self, 
#                  num_inputs: int, 
#                  num_channels: ArrayLike, 
#                  kernel_size: int, 
#                  dropout: float = 0.1):
#         super().__init__() 
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation = 2 ** i # exponential increase in dilation
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout))
#         self.network = nn.Sequential(*layers)
            
            
#     def forward(self, x):
#         print(f"Allocated memory2: {torch.cuda.memory_allocated() / 1024**2} MB")
#         print(f"Cached memory2: {torch.cuda.memory_reserved() / 1024**2} MB")
#         return self.network(x)
            
            
            
# class TCNWrapper(nn.Module):
#     def __init__(self, tcn: TCN, FE_model, input_shape='NCL', num_classes=3):
#         """
#         Current version attempts to classify each frame in a video.
#         num_classes = 3 for 3 endoscopy classes (0, 1, 2)
#         """
#         super().__init__()
#         self.tcn = tcn
#         self.input_shape = input_shape

#         # Load ResNet-50 as feature extractor
#         self.feature_extractor = FE_model
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False  # Freeze feature extractor parameters

#         self.final_channels = tcn.network[-1].conv2.out_channels # Get final number of channels from TCN
#         self.classifier = nn.Linear(self.final_channels, num_classes)
        
#     def forward(self, x):
#         """
#         x: Tensor of shape (batch, frames, channels, height, width) -> (N, L, C, H, W)
#         Returns: (N, L, num_classes) # frame-level classification
#         """


#         batch_size, seq_len, c, h, w = x.shape  # Extract batch and sequence dimensions
        
#         # Flatten batch and time dimension for ResNet processing
#         x = x.view(batch_size * seq_len, c, h, w)  # Reshape to (N*L, C, H, W)
#         x = self.feature_extractor(x)  # Pass through ResNet-50 -> Outputs (N*L, feature_dim)
        
#         # Reshape back to (N, L, feature_dim)
#         feature_dim = x.shape[-1]  # Get extracted feature dimension
#         x = x.view(batch_size, seq_len, feature_dim)  # Reshape to (N, L, feature_dim)
        
#         # TCN expects input in (N, C, L) format. If input_shape is NLC, transpose to NCL
#         if self.input_shape == 'NLC':
#             x = x.transpose(1,2)  # Transpose to (N, feature_dim, L)
        
#         # print(f"Feature extractor output shape: {x.shape}")
#         tcn_out = self.tcn(x)  # Pass features through TCN
        
#         # Frame-level classification
#         # We want to classify each time-step, so we will do: (N, final_channels, L) -> (N, L, final_channels)
#         tcn_out = tcn_out.transpose(1,2)  # Transpose to (N, L, final_channels)

#         logits = self.classifier(tcn.out)

#         return logits
        
    

# # Test run
# if __name__ == "__main__":
#     torch.cuda.empty_cache()

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")
#     print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
#     print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")
    
#     # Load feature extractor
#     # resnet_model_path = '/home/local/VANDERBILT/winterga/medic/feature_extractor/checkpoints/Resnet50_021225_07/Resnet50_021225_07.pth'
#     resnet_model_path = '/home/local/VANDERBILT/winterga/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth'
#     FE = torch.load(f=resnet_model_path, map_location=device, weights_only=False)


    
#     # Define TCN
#     tcn = TCN(num_inputs=3, num_channels=[16, 32, 64], kernel_size=3, dropout=0.1)
#     model = TCNWrapper(tcn, FE, input_shape='NLC')
#     model=model.to(device)
#     print(f"TCN Parameters ({len(list(model.parameters()))} parameters):")
#     for name, param in model.named_parameters():
#         print(name)
#     # print(f"TCN parameters: {model.parameters()}")
#     print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
#     print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")
    
#     # # Sample input
#     # batch_size = 4
#     # sequence_length = 50
#     # input_channels = 3
#     # image_height=256
#     # image_width=256
#     # x = torch.randn(batch_size, sequence_length, input_channels, image_height, image_width) # (N, L, C, H, W)
#     # x=x.to(device)
#     # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
#     # print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2} MB")
    
#     # # Forward pass
#     # output = model(x) # Will get passed to FE with shape (N*L, C, H, W)
#     # print(output.shape) # Expected output: torch.Size([8, 64, 50])

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
            res = self.residual(x)  # shortcut connection
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.activation(self.conv2(x))
        x = self.dropout(x)
        return x + res  # add residual connection


class TCN(nn.Module):
    def __init__(self,
                 num_inputs: int,       # e.g. 2048 if from ResNet-50
                 num_channels: ArrayLike, 
                 kernel_size: int, 
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i  # exponential increase in dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
            
    def forward(self, x):
        # x shape: (N, num_inputs, L)
        # We return shape: (N, final_channels, L)
        return self.network(x)


class TCNWrapper(nn.Module):
    """
    For *sequence-level* classification into 3 classes.
    We get a single prediction per sequence: shape (N, 3).
    """
    def __init__(self, tcn: TCN, FE_model, input_shape='NCL', num_classes=3):
        super().__init__()
        self.tcn = tcn
        self.input_shape = input_shape

        # Load ResNet-50 as feature extractor
        self.feature_extractor = FE_model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze feature extractor parameters

        # The last TCN layer typically has num_channels[-1] channels
        self.final_channels = tcn.network[-1].conv2.out_channels
        self.classifier = nn.Linear(self.final_channels, num_classes)
        
    def forward(self, x):
        """
        x: shape (N, L, C, H, W)
           N = batch size
           L = sequence length
           C,H,W = channel,height,width for each frame
        Returns: (N, num_classes)  # one prediction per entire sequence
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Flatten (N,L) -> (N*L) to pass all frames into ResNet
        x = x.view(batch_size * seq_len, c, h, w)  # => (N*L, C, H, W)
        x = self.feature_extractor(x)              # => (N*L, 2048, 1, 1) typically
        x = torch.flatten(x, start_dim=1)          # => (N*L, 2048)

        feature_dim = x.shape[-1]                  # usually 2048

        # Reshape back: => (N, L, 2048)
        x = x.view(batch_size, seq_len, feature_dim)

        # TCN wants (N, feature_dim, L) if input_shape=='NLC'
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)  # => (N, 2048, L)

        # Pass through TCN => (N, final_channels, L)
        tcn_out = self.tcn(x)

        # Now we want a SINGLE classification per sequence
        # We'll average pool across the time dimension L:
        # shape => (N, final_channels)
        tcn_out = tcn_out.mean(dim=2)

        # Finally get logits => (N, num_classes)
        logits = self.classifier(tcn_out)

        return logits


if __name__ == "__main__":
    # Example usage
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Suppose we load a pre-trained feature extractor
    # resnet_model_path = '/path/to/resnet50.pth'
    # FE = torch.load(resnet_model_path, map_location=device)

    # For demonstration, let's just create a dummy FE that outputs 2048 features
    class DummyFE(nn.Module):
        def __init__(self, out_dim=2048):
            super().__init__()
            self.linear = nn.Linear(1, out_dim)  # dummy

        def forward(self, x):
            # x => (N*L, C, H, W)
            # We'll just create random 2048 dims for demonstration
            batch_size = x.shape[0]
            return torch.randn(batch_size, 2048, device=x.device)

    FE = DummyFE()

    # Define TCN with 2048 input channels
    tcn = TCN(num_inputs=2048, num_channels=[16, 32, 64], kernel_size=3, dropout=0.1)
    model = TCNWrapper(tcn, FE, input_shape='NLC').to(device)

    # Create sample input
    N, L, C, H, W = (2, 8, 3, 224, 224)
    dummy_input = torch.randn(N, L, C, H, W).to(device)

    # Forward pass
    output = model(dummy_input)  # => (N, 3)
    print("Output shape:", output.shape)  # (2, 3)
