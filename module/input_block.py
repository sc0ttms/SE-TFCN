# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchinfo import summary


class InputBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.net = nn.Sequential(
            nn.Conv2d(
                self.num_channels[0],
                self.num_channels[1],
                kernel_size=tuple(self.kernel_size),
                stride=self.stride,
                padding=self.padding,
                bias=False,
            ),
            nn.BatchNorm2d(self.num_channels[1]),
        )

    def forward(self, inputs):
        # inputs [B, 1, F, T] -> outputs [B, C, F, T]
        outputs = self.net(inputs)
        return outputs


if __name__ == "__main__":
    print(f"Test InputBlock Module Start...")

    # get model
    model = InputBlock([1, 16], kernel_size=(7, 5), stride=1, padding=(3, 2))
    # get inputs
    inputs = torch.randn([2, 1, 256, 201])
    # print network
    summary(model, input_size=inputs.shape)
    # forward
    outputs = model(inputs)

    print(f"Test InputBlock Module End...")

    pass
