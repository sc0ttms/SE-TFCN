# -*- coding: utf-8 -*-

import torch
import numpy as np

EPS = np.finfo(np.float32).eps


def is_clipped(data, clipping_threshold=0.99):
    return any(abs(data) > clipping_threshold)


def sub_sample(noisy, clean, samples):
    """random select fixed-length data from noisy and clean

    Args:
        noisy (float): noisy data
        clean (float): clean data
        samples (int): fixed length

    Returns:
        noisy, clean: fixed-length noisy and clean
    """
    length = len(noisy)

    if length > samples:
        start_idx = np.random.randint(length - samples)
        end_idx = start_idx + samples
        noisy = noisy[start_idx:end_idx]
        clean = clean[start_idx:end_idx]
    elif length < samples:
        noisy = np.append(noisy, np.zeros(samples - length))
        clean = np.append(clean, np.zeros(samples - length))
    else:
        pass

    assert len(noisy) == len(clean) == samples

    return noisy, clean


def offline_laplace_norm(input):
    """
    Args:
        input: [B, C, F, T]
    Returns:
        [B, C, F, T]
    """
    # utterance-level mu
    mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)

    normed = input / (mu + 1e-5)

    return normed


def cumulative_laplace_norm(input):
    """
    Args:
        input: [B, C, F, T]
    Returns:
    """
    batch_size, num_channels, num_freqs, num_frames = input.size()
    input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

    step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
    cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

    entry_count = torch.arange(num_freqs, num_freqs * num_frames + 1, num_freqs, dtype=input.dtype, device=input.device)
    entry_count = entry_count.reshape(1, num_frames)  # [1, T]
    entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

    cumulative_mean = cumulative_sum / entry_count  # B, T
    cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

    normed = input / (cumulative_mean + 1e-7)

    return normed.reshape(batch_size, num_channels, num_freqs, num_frames)
