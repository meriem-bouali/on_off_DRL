#! https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/7

import torch.nn as nn

"""
    A PyTorch implementation of Depthwise Separable Convolution.

    This layer decomposes a standard convolution into two steps:
      1. Depthwise convolution  — applies a spatial convolution independently
         to each input channel (no cross-channel mixing).
      2. Pointwise convolution — applies a 1x1 convolution to mix features
         across channels and produce the final output.

    Benefits:
      - Greatly reduces the number of parameters and FLOPs compared to standard Conv2d.
      - Commonly used in efficient network architectures like MobileNet.
"""

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride=1, padding=1):
        super().__init__()

        # Depthwise convolution:
        # - This performs a separate spatial convolution for each input channel independently.
        # - groups=in_channels tells PyTorch to split the input into 'in_channels' groups,
        #   each with its own K×K filter, so there is no cross-channel information sharing yet.
        # - out_channels is set equal to in_channels because each channel gets exactly one filter.
        # - Effectively: captures spatial patterns within each channel but does not combine channels.
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
       
        # Pointwise convolution:
        # - 1×1 kernel mixes information across channels
        # - This is where the independent channel features from depthwise conv are combined
        # - Produces the desired out_channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        # Step 1: Per-channel spatial filtering (depthwise conv)
        x = self.depthwise(x)

        # Step 2: Channel mixing (pointwise conv)
        x = self.pointwise(x)

        # Output after depthwise separable convolution
        return x