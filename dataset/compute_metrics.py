# -*- coding: utf-8 -*-

import sys
import os
import argparse
import toml
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append(os.getcwd())
from audio.metrics import SI_SDR, STOI, WB_PESQ, NB_PESQ, REGISTERED_METRICS


def calculate_metric(noisy_file, clean_file, sr=16000, metric_type="STOI", pre_load=False):
    # get noisy, clean
    if pre_load == False:
        noisy, _ = librosa.load(noisy_file, sr=sr)
        clean, _ = librosa.load(clean_file, sr=sr)
    else:
        noisy = noisy_file
        clean = clean_file
    assert len(noisy) == len(clean)

    # get metric score
    if metric_type in ["SI_SDR"]:
        return SI_SDR(noisy, clean)
    elif metric_type in ["STOI"]:
        return STOI(noisy, clean, sr=sr)
    elif metric_type in ["WB_PESQ"]:
        return WB_PESQ(noisy, clean)
    elif metric_type in ["NB_PESQ"]:
        return NB_PESQ(noisy, clean)


def compute_metric(noisy_files, clean_files, metrics, n_folds=1, n_jobs=8, pre_load=False):
    for metric_type, _ in metrics.items():
        assert metric_type in REGISTERED_METRICS

        split_num = len(noisy_files) // n_folds
        score = []
        for n in range(n_folds):
            metric_score = Parallel(n_jobs=n_jobs)(
                delayed(calculate_metric)(
                    noisy_file,
                    clean_file,
                    sr=8000 if metric_type in ["NB_PESQ"] else 16000,
                    metric_type=metric_type,
                    pre_load=pre_load,
                )
                for noisy_file, clean_file in tqdm(
                    zip(
                        noisy_files[n * split_num : (n + 1) * split_num],
                        clean_files[n * split_num : (n + 1) * split_num],
                    )
                )
            )
            score.append(np.mean(metric_score))
        metrics[metric_type] = np.mean(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute_metrics")
    parser.add_argument("-c", "--config", required=True, type=str, help="Config (*.toml).")
    args = parser.parse_args()

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get set path
    train_path = os.path.join(dataset_path, "train.csv")
    valid_path = os.path.join(dataset_path, "valid.csv")
    test_path = os.path.join(dataset_path, "test.csv")

    # get train files
    train_files = pd.read_csv(train_path).values
    train_noisy_files = train_files[:, 0].reshape(1, len(train_files))[0]
    train_clean_files = train_files[:, 1].reshape(1, len(train_files))[0]
    # get valid files
    valid_files = pd.read_csv(valid_path).values
    valid_noisy_files = valid_files[:, 0].reshape(1, len(valid_files))[0]
    valid_clean_files = valid_files[:, 1].reshape(1, len(valid_files))[0]
    # get test files
    test_files = pd.read_csv(test_path).values
    test_noisy_files = test_files[:, 0].reshape(1, len(test_files))[0]
    test_clean_files = test_files[:, 1].reshape(1, len(test_files))[0]

    # get compute metrics config
    config = toml.load(args.config)
    # get n_jobs
    n_folds = config["ppl"]["n_folds"]
    n_jobs = config["ppl"]["n_jobs"]

    # get metrics
    metrics = {
        "SI_SDR": [],
        "STOI": [],
        "WB_PESQ": [],
        "NB_PESQ": [],
    }

    # compute train metrics
    compute_metric(
        train_noisy_files, train_clean_files, metrics, n_folds=n_folds, n_jobs=n_jobs, pre_load=False,
    )
    # save train metrics
    df = pd.DataFrame(metrics, index=["train"])
    df.to_csv(os.path.join(dataset_path, "train_metrics.csv"))

    # get metrics
    metrics = {
        "SI_SDR": [],
        "STOI": [],
        "WB_PESQ": [],
        "NB_PESQ": [],
    }

    # compute valid metrics
    compute_metric(
        valid_noisy_files, valid_clean_files, metrics, n_folds=n_folds, n_jobs=n_jobs, pre_load=False,
    )
    # save train metrics
    df = pd.DataFrame(metrics, index=["valid"])
    df.to_csv(os.path.join(dataset_path, "valid_metrics.csv"))

    # get metrics
    metrics = {
        "SI_SDR": [],
        "STOI": [],
        "WB_PESQ": [],
        "NB_PESQ": [],
    }

    # compute test metrics
    compute_metric(
        test_noisy_files, test_clean_files, metrics, n_folds=n_folds, n_jobs=n_jobs, pre_load=False,
    )
    # save train metrics
    df = pd.DataFrame(metrics, index=["test"])
    df.to_csv(os.path.join(dataset_path, "test_metrics.csv"))
