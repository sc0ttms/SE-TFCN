# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchinfo import summary


class OutputBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels

        self.net = nn.Sequential(nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=1), nn.PReLU())

    def forward(self, inputs):
        # inputs [B, C, F, T] -> outputs [B, 1, F, T]
        outputs = self.net(inputs)
        return outputs


if __name__ == "__main__":
    print(f"Test OutputBlock Module Start...")

    # get model
    model = OutputBlock([16, 1])
    # get inputs
    inputs = torch.randn([2, 16, 256, 201])
    # print network
    summary(model, input_size=inputs.shape)
    # forward
    outputs = model(inputs)

    print(f"Test OutputBlock Module End...")

    pass
