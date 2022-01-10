# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from thop import profile


class InputBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, stride, padding):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.net = nn.Sequential(
            nn.BatchNorm2d(self.num_channels[0]),
            nn.Conv2d(
                self.num_channels[0],
                self.num_channels[1],
                kernel_size=tuple(self.kernel_size),
                stride=self.stride,
                padding=self.padding,
            ),
        )

    def forward(self, inputs):
        # inputs [B, 1, F, T] -> outputs [B, C, F, T]
        outputs = self.net(inputs)[:, :, :, : -self.padding[1]]
        return outputs


if __name__ == "__main__":
    print(f"Test InputBlock Module Start...")

    # get model
    model = InputBlock([1, 16], kernel_size=(5, 7), stride=1, padding=(2, 6))
    # get inputs
    inputs = torch.randn([2, 1, 256, 201])
    # print network
    macs, params = profile(model, inputs=(inputs,), custom_ops={})
    print(f"flops {macs / 1e9:.6f} G, params {params / 1e6:.6f} M")
    # forward
    outputs = model(inputs)

    print(f"Test InputBlock Module End...")

    pass
