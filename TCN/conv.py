import os
import warnings
import torch.nn as nn

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation, **kwargs)
        
    def forward(self, x):
        # apply convolution
        out = super().forward(x)
        # Remove future information by trimming extra padding
        return out[:, :, :-self.padding[0]]