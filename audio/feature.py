# -*- coding: utf-8 -*-

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
