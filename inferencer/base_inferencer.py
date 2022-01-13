# -*- coding: utf-8 -*-

import sys
import os
import argparse
import toml
import pandas as pd
import soundfile as sf
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from trainer.base_trainer import BaseTrainer
from module.tfcn import TFCN
from dataset.dataset import DNS_Dataset
from dataset.compute_metrics import compute_metric
from audio.feature import is_clipped
from audio.utils import prepare_empty_path


class BaseInferencer(BaseTrainer):
    def __init__(self, config, model, test_iter, device="cpu"):
        super().__init__(config, model, test_iter, test_iter, device=device)
        # init path
        self.output_path = os.path.join(self.base_path, "enhanced", "base")
        self.logs_path = os.path.join(self.base_path, "logs", "inference", "base")
        self.metrics_path = os.path.join(self.base_path, "metrics", "base")
        # mkdir path
        prepare_empty_path([self.output_path, self.logs_path, self.metrics_path])

        # init writer_text_enh_clipped_step
        self.writer_text_enh_clipped_step = 1

    def check_clipped(self, enh, enh_file):
        if is_clipped(enh):
            self.writer.add_text(
                tag="enh_clipped", text_string=enh_file, global_step=self.writer_text_enh_clipped_step,
            )
        self.writer_text_enh_clipped_step += 1

    def save_metrics(self, enh_list, clean_list, n_folds=1, n_jobs=8):
        # get metrics
        metrics = {
            "SI_SDR": [],
            "STOI": [],
            "WB_PESQ": [],
            "NB_PESQ": [],
        }

        # compute enh metrics
        compute_metric(
            enh_list, clean_list, metrics, n_folds=n_folds, n_jobs=n_jobs, pre_load=True,
        )

        # save train metrics
        df = pd.DataFrame(metrics, index=["enh"])
        df.to_csv(os.path.join(self.metrics_path, "enh_metrics.csv"))

    def save_audio(self, audio_list, audio_files, n_folds=1, n_jobs=8):
        split_num = len(audio_list) // n_folds
        for n in range(n_folds):
            Parallel(n_jobs=n_jobs)(
                delayed(sf.write)(audio_file, audio, samplerate=self.sr)
                for audio_file, audio in tqdm(
                    zip(
                        audio_files[n * split_num : (n + 1) * split_num],
                        audio_list[n * split_num : (n + 1) * split_num],
                    )
                )
            )

    @torch.no_grad()
    def valid_epoch(self, epoch):
        noisy_list = []
        clean_list = []
        noisy_files = []
        enh_list = []
        enh_files = []

        loss_total = 0.0
        for noisy, clean, noisy_file in tqdm(self.valid_iter, desc="inference"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T]
            noisy_lps, noisy_phase = self.audio_stft(noisy)
            clean_lps, _ = self.audio_stft(clean)

            noisy_lps = noisy_lps.unsqueeze(dim=1)
            enh_lps = self.model(noisy_lps)
            enh_lps = enh_lps.squeeze(dim=1)

            # [B, S]
            enh = self.audio_istft(enh_lps, noisy_phase)

            loss = self.loss(enh_lps, clean_lps)

            loss_total += loss.item()

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enh = enh.detach().squeeze(0).cpu().numpy()
            assert len(noisy) == len(clean) == len(enh)

            for i in range(len(noisy_file)):
                enh_file = os.path.join(self.output_path, os.path.basename(noisy_file[i]).replace("noisy", "enh_noisy"))
                self.check_clipped(enh[i], enh_file)
                enh_files.append(enh_file)

            noisy_list = np.concatenate([noisy_list, noisy], axis=0) if len(noisy_list) else noisy
            clean_list = np.concatenate([clean_list, clean], axis=0) if len(clean_list) else clean
            enh_list = np.concatenate([enh_list, enh], axis=0) if len(enh_list) else enh
            noisy_files = np.concatenate([noisy_files, noisy_file], axis=0) if len(noisy_files) else noisy_file

        # update learning rate
        self.scheduler.step(loss_total / len(self.valid_iter))

        # visual audio
        for i in range(self.visual_samples):
            self.audio_visualization(noisy_list[i], clean_list[i], enh_list[i], os.path.basename(noisy_files[i]), epoch)

        # logs
        self.writer.add_scalar("loss/inference", loss_total / len(self.valid_iter), epoch)

        # visual metrics and get valid score
        self.save_metrics(enh_list, clean_list, n_folds=self.n_folds, n_jobs=self.n_jobs)
        # save audio
        self.save_audio(enh_list, enh_files, n_folds=self.n_folds, n_jobs=self.n_jobs)

    def __call__(self):
        # init logs
        self.init_logs()

        # to device
        self.model = self.model.to(self.device)
        # load model
        self.load_pre_model()

        # set to eval
        self.set_model_to_eval_mode()
        self.valid_epoch(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inferencer")
    parser.add_argument("-c", "--config", required=True, type=str, help="Config (*.toml).")
    args = parser.parse_args()

    # config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # get config
    config = toml.load(args.config)

    # get dataset path
    dataset_path = os.path.join(os.getcwd(), "dataset_csv")

    # get dataloader args
    batch_size = config["dataloader"]["batch_size"]
    num_workers = 0 if device == "cpu" else config["dataloader"]["num_workers"]
    drop_last = config["dataloader"]["drop_last"]
    pin_memory = config["dataloader"]["pin_memory"]

    # get test_iter
    test_set = DNS_Dataset(dataset_path, config, mode="test")
    test_iter = DataLoader(
        test_set,
        batch_size=batch_size[1],
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # config model
    model = globals().get(config["model"]["name"])(
        num_channels=config["model"]["num_channels"],
        kernel_size=config["model"]["kernel_size"],
        stride=config["model"]["stride"],
        num_repeated=config["model"]["num_repeated"],
        num_dilated=config["model"]["num_dilated"],
    )

    # inferencer
    inference = BaseInferencer(config, model, test_iter, device)

    # inference
    inference()
    pass
