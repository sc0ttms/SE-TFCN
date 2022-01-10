# -*- coding: utf-8 -*-

import os
import re
import argparse
import toml
import librosa
import pandas as pd
import numpy as np


def split_data(data, valid_ratio=0.1, test_ratio=0.1):
    # get ln en
    data_len = len(data)
    valid_len = int(data_len * valid_ratio)
    test_len = int(data_len * test_ratio)

    # random choice idx for valid and test
    data_idx = np.arange(data_len)
    vt_mask = np.random.choice(data_idx, valid_len + test_len, replace=False)
    valid_mask = vt_mask[:valid_len]
    test_mask = vt_mask[valid_len + 1 :]

    # gen set
    train_files = [data[file] for file in range(data_len) if file not in vt_mask]
    valid_files = [data[file] for file in range(data_len) if file in valid_mask]
    test_files = [data[file] for file in range(data_len) if file in test_mask]

    return train_files, valid_files, test_files


def gen(set_path, config):
    # get clean and noisy path
    noisy_path = os.path.join(set_path, "noisy")
    clean_path = os.path.join(set_path, "clean")

    # get dataset args
    audio_format = config["dataset"]["audio_format"]
    valid_ratio = config["dataset"]["valid_ratio"]
    test_ratio = config["dataset"]["test_ratio"]

    # find all files
    noisy_files = librosa.util.find_files(noisy_path, ext=audio_format)
    noisy_files.sort(key=lambda x: int(re.findall("\d+", x)[-1]))
    clean_files = librosa.util.find_files(clean_path, ext=audio_format)
    clean_files.sort(key=lambda x: int(re.findall("\d+", x)[-1]))
    assert len(noisy_files) == len(clean_files)

    # merge noisy and clean
    noisy_clean_files = [file for file in zip(noisy_files, clean_files)]

    # split noisy_clean_files
    train_files, valid_files, test_files = split_data(noisy_clean_files, valid_ratio=valid_ratio, test_ratio=test_ratio)

    # save set to csv
    save_path = os.path.join(os.getcwd(), "dataset_csv")
    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame(train_files)
    df.to_csv(os.path.join(save_path, "train.csv"), index=False, header=None)
    df = pd.DataFrame(valid_files)
    df.to_csv(os.path.join(save_path, "valid.csv"), index=False, header=None)
    df = pd.DataFrame(test_files)
    df.to_csv(os.path.join(save_path, "test.csv"), index=False, header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gen")
    parser.add_argument("-c", "--config", required=True, type=str, help="Config (*.toml).")
    args = parser.parse_args()

    # get gen csv config
    config = toml.load(args.config)
    # get seed
    seed = config["random"]["seed"]
    # set seed
    np.random.seed(seed)

    # get dataset path
    dataset_path = config["path"]["dataset"]

    # gen set
    gen(dataset_path, config)
