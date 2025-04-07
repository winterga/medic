# import os
# import warnings
# import torch.nn as nn

# class CausalConv1d(nn.Conv1d):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
#         super().__init__(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation, **kwargs)
        
#     def forward(self, x):
#         # apply convolution
#         out = super().forward(x)
#         # Remove future information by trimming extra padding
#         return out[:, :, :-self.padding[0]]


# conv.py
import torch.nn as nn

class CausalConv1d(nn.Conv1d):
    """
    A 1D convolution that pads to maintain sequence length.
    We do full left-padding and then slice off 'future' frames,
    so the output shape matches the input's time dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,  # left-padding
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        # x => (N, C, L)
        # conv => (N, out_channels, L + (kernel_size-1)*dilation)
        out = super().forward(x)
        # Trim the right side so output length matches input length L
        return out[:, :, : x.size(2)]

