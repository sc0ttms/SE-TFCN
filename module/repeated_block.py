# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn as nn
from torchinfo import summary


sys.path.append(os.getcwd())
from module.dilated_block import DilatedBlock


class RepeatedBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, num_dilated):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_dilated = num_dilated

        dilated_layer = []
        for i in range(self.num_dilated):
            dilation = 2 ** i
            dilated_layer += [
                DilatedBlock(
                    num_channels=self.num_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=(1, (self.kernel_size[1] - 1) // 2 * 2 * dilation),
                    dilation=(1, dilation),
                )
            ]

        self.net = nn.Sequential(*dilated_layer)

    def forward(self, inputs):
        # inputs [B, C, F, T] -> outputs [B, C, F, T]
        outputs = self.net(inputs)
        return outputs


if __name__ == "__main__":
    print(f"Test RepeatedBlock Module Start...")

    # get model
    model = RepeatedBlock([16, 64], kernel_size=(3, 3), stride=1, num_dilated=8)
    # get inputs
    inputs = torch.randn([2, 16, 256, 201])
    # print network
    summary(model, input_size=inputs.shape)
    # forward
    outputs = model(inputs)

    print(f"Test RepeatedBlock Module End...")

    pass
