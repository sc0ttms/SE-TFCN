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

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.05)

    @staticmethod
    def cumulative_laplace_norm(input):
        """cumulative laplace norm
        Args:
            input (float): [B, C, F, T]
        Returns:
            normed (float): [B, C, F, T]
        """
        [batch_size, num_channels, num_freqs, num_frames] = input.shape
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(num_freqs, num_freqs * num_frames + 1, num_freqs, dtype=input.dtype)
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def forward(self, inputs):
        # inputs [B, 1, F, T] -> outputs [B, C, F, T]
        inputs = self.cumulative_laplace_norm(inputs)
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
