# -*- encoding: utf-8 -*-

import sys
import os
import argparse
import toml
import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.getcwd())
from audio.feature import sub_sample


class DNS_Dataset(Dataset):
    def __init__(self, set_path, config, mode="train"):
        super().__init__()
        assert mode in ["train", "valid", "test"]

        # set mode
        self.mode = mode

        # get args
        self.sr = config["dataset"]["sr"]
        self.n_fft = config["dataset"]["n_fft"]
        self.win_len = config["dataset"]["win_len"]
        self.hop_len = config["dataset"]["hop_len"]
        self.samples = int(config["dataset"]["audio_len"] * self.sr)

        # get path
        data_csv_path = os.path.join(set_path, mode + ".csv")
        # get noisy, clean files
        noisy_clean_files = pd.read_csv(data_csv_path).values
        noisy_files = noisy_clean_files[:, 0].reshape(1, len(noisy_clean_files))[0]
        clean_files = noisy_clean_files[:, 1].reshape(1, len(noisy_clean_files))[0]
        # limit
        limit = config["dataset"]["limit"]
        if limit:
            noisy_files = noisy_files[:limit]
            clean_files = clean_files[:limit]
        self.noisy_files = noisy_files
        self.clean_files = clean_files

        # set len
        self.length = len(self.noisy_files)
        print(f"number of {self.mode} files {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.mode in ["train"]:
            # load noisy
            noisy_file = self.noisy_files[idx]
            noisy, _ = librosa.load(noisy_file, sr=self.sr)
            # load clean
            clean_file = self.clean_files[idx]
            clean, _ = librosa.load(clean_file, sr=self.sr)
            # get target samples
            noisy, clean = sub_sample(noisy, clean, self.samples)

            return noisy, clean
        elif self.mode in ["valid", "test"]:
            # load noisy
            noisy_file = self.noisy_files[idx]
            noisy, _ = librosa.load(noisy_file, sr=self.sr)
            # load clean
            clean_file = self.clean_files[idx]
            clean, _ = librosa.load(clean_file, sr=self.sr)

            return noisy, clean, noisy_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset")
    parser.add_argument("-c", "--config", required=True, type=str, help="Config (*.toml).")
    args = parser.parse_args()

    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get dataset config
    config = toml.load(args.config)
    # get seed
    seed = config["random"]["seed"]
    np.random.seed(seed)
    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]
    pin_memory = config["dataloader"]["pin_memory"]

    # get train_iter
    train_set = DNS_Dataset(dataset_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    for noisy, clean in tqdm(train_iter, desc="train_iter"):
        print(noisy.shape, clean.shape)
        print(noisy.dtype, clean.dtype)

    # get valid_iter
    valid_set = DNS_Dataset(dataset_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory,
    )
    for noisy, clean, _ in tqdm(valid_iter, desc="valid_iter"):
        print(noisy.shape, clean.shape)
        print(noisy.dtype, clean.dtype)

    # get test_iter
    test_set = DNS_Dataset(dataset_path, config, mode="test")
    test_iter = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory,
    )
    for noisy, clean, noisy_file in tqdm(valid_iter, desc="valid_iter"):
        print(noisy.shape, clean.shape, noisy_file)
        print(noisy.dtype, clean.dtype, noisy_file)

    pass
