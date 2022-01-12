# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchinfo import summary


class DilatedBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding, dilation):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.net = nn.Sequential(
            # conv_0
            nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(self.num_channels[1]),
            # conv_1
            nn.Conv2d(
                self.num_channels[1],
                self.num_channels[1],
                kernel_size=tuple(self.kernel_size),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.num_channels[1],
                bias=False,
            ),
            nn.PReLU(),
            nn.BatchNorm2d(self.num_channels[1]),
            # conv_2
            nn.Conv2d(self.num_channels[1], self.num_channels[0], kernel_size=1, bias=False),
        )

    def forward(self, inputs):
        # inputs [B, C, F, T] -> outputs [B, C, F, T]
        outputs = self.net(inputs) + inputs
        return outputs


if __name__ == "__main__":
    print(f"Test DilatedBlock Module Start...")

    # get model
    # model = DilatedBlock([16, 64], kernel_size=(3, 3), stride=1, padding=(1, 1), dilation=(1, 1))
    model = DilatedBlock([16, 64], kernel_size=(3, 3), stride=1, padding=(1, 2), dilation=(1, 2))
    # get inputs
    inputs = torch.randn([2, 16, 256, 201])
    # print network
    summary(model, input_size=inputs.shape)
    # forward
    outputs = model(inputs)

    print(f"Test DilatedBlock Module End...")

    pass
