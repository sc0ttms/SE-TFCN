# -*- coding: utf-8 -*-

import os
import zipfile
import torch
import torch.nn as nn
from tqdm import tqdm


def unzip(zip_path, unzip_path=None):
    """unzip

    Args:
        zip_path (str): zip path
        unzip_path (str, optional): unzip path. Defaults to None.

    Returns:
        unzip_path: unzip path
    """

    # set unzip path
    if unzip_path == None:
        unzip_path = os.path.splitext(zip_path)[0]

    # unzip
    with zipfile.ZipFile(zip_path) as zf:
        for file in tqdm(zf.infolist(), desc="unzip..."):
            try:
                zf.extract(file, unzip_path)
            except zipfile.error as e:
                print(e)

    return unzip_path


def prepare_empty_path(paths, resume=False):
    """prepare empty path

    Args:
        paths (list): path list
        resume (bool, optional): whether to resume. Defaults to False.
    """
    for path in paths:
        if resume:
            assert os.path.exists(path)
        else:
            os.makedirs(path, exist_ok=True)


def print_size_of_model(model):
    """print size of model

    Args:
        model (torch.nn.Module): model
    """
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def print_networks(models: list):
    """print networks

    Args:
        models (list): models list
    """
    print(f"Contains {len(models)} models, the number of the parameters is: ")

    params_of_all_networks = 0
    for idx, model in enumerate(models, start=1):
        params_of_network = 0
        for param in model.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")


def flatten_parameters(m):
    if isinstance(m, nn.LSTM):
        m.flatten_parameters()
