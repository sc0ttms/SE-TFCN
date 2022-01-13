# -*- coding: utf-8 -*-

import sys
import os
import torch
import torch.nn as nn
from torchinfo import summary


sys.path.append(os.getcwd())
from audio.feature import offline_laplace_norm, cumulative_laplace_norm
from module.input_block import InputBlock
from module.repeated_block import RepeatedBlock
from module.output_block import OutputBlock


class TFCN(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, num_repeated, num_dilated):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_repeated = num_repeated
        self.num_dilated = num_dilated

        input_layer = InputBlock(
            num_channels=self.num_channels[:2],
            kernel_size=self.kernel_size[0],
            stride=self.stride,
            padding=((self.kernel_size[0][0] - 1) // 2, (self.kernel_size[0][1] - 1) // 2),
        )

        repeated_layer = []
        for i in range(self.num_repeated):
            repeated_layer += [
                RepeatedBlock(
                    num_channels=self.num_channels[1:3],
                    kernel_size=self.kernel_size[1],
                    stride=self.stride,
                    num_dilated=self.num_dilated,
                )
            ]

        output_layer = OutputBlock(num_channels=self.num_channels[:2][::-1])

        self.net = nn.Sequential(input_layer, *repeated_layer, output_layer)

    def forward(self, inputs):
        # inputs [B, 1, F, T] -> outputs [B, 1, F, T]
        normed = cumulative_laplace_norm(inputs)
        outputs = self.net(normed)
        return outputs


if __name__ == "__main__":
    print(f"Test TFCN Module Start...")

    # get model
    model = TFCN([1, 16, 64], kernel_size=[(7, 5), (3, 3)], stride=1, num_repeated=4, num_dilated=8)
    # get inputs
    inputs = torch.randn([2, 1, 257, 201])
    # print network
    summary(model, input_size=inputs.shape)
    # forward
    outputs = model(inputs)

    print(f"Test TFCN Module End...")

    pass
