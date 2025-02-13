from conv import CausalConv1d
import torch.nn as nn
import torch
from numpy.typing import ArrayLike
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
        return self.network(x)
            
            
            
class TCNWrapper(nn.Module):
    def __init__(self, tcn: TCN, input_shape='NCL'):
        super().__init__()
        self.tcn = tcn
        self.input_shape = input_shape
        
    def forward(self, x):
        if self.input_shape == 'NLC':
            x = x.transpose(1,2)
        x = self.tcn(x)
        
        return x
        
    

# Test run
if __name__ == "__main__":
    # Define TCN
    tcn = TCN(num_inputs=10, num_channels=[16, 32, 64], kernel_size=3, dropout=0.1)
    model = TCNWrapper(tcn, input_shape='NLC')
    
    # Sample input
    batch_size = 8
    sequence_length = 50
    input_channels = 10
    x = torch.randn(batch_size, sequence_length, input_channels) # Shape NLC
    
    # Forward pass
    output = model(x)
    print(output.shape) # Expected output: torch.Size([8, 64, 50])