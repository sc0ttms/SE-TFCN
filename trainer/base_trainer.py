# -*- coding: utf-8 -*-

import sys
import os
import argparse
import toml
from tqdm import tqdm
from torchinfo import summary
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from dataset.dataset import DNS_Dataset
from module.tfcn import TFCN
from dataset.compute_metrics import compute_metric
from audio.utils import prepare_empty_path
from audio.metrics import SI_SDR, transform_pesq_range
from audio.feature import EPS

plt.switch_backend("agg")


class BaseTrainer:
    def __init__(self, config, model, train_iter, valid_iter, device="cpu"):
        # get config
        self.config = config
        # get model
        self.model = model
        # get device
        self.device = device
        # get dataset iter
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        # get meta args
        self.use_cudnn = False if self.device == "cpu" else config["meta"]["use_cudnn"]
        self.use_amp = False if self.device == "cpu" else config["meta"]["use_amp"]
        # get base path
        self.base_path = config["path"]["base"]
        # get pre model path
        self.pre_model_path = config["path"]["pre_model"]
        # get ppl args
        self.n_folds = config["ppl"]["n_folds"]
        self.n_jobs = config["ppl"]["n_jobs"]
        # get visual args
        self.visual_samples = config["visual"]["samples"]
        # get checkpoint args
        self.save_checkpoint_interval = config["checkpoint"]["save_interval"]
        # get grad args
        self.clip_grad_norm_value = config["grad"]["clip_grad_norm_value"]
        # get dataset args
        self.sr = config["dataset"]["sr"]
        self.n_fft = config["dataset"]["n_fft"]
        self.win_len = config["dataset"]["win_len"]
        self.hop_len = config["dataset"]["hop_len"]
        self.audio_len = config["dataset"]["audio_len"]
        self.window = torch.hann_window(self.win_len, periodic=False, device=self.device)
        # get train args
        self.resume = config["train"]["resume"]
        self.epochs = config["train"]["epochs"]
        self.valid_start_epoch = config["train"]["valid_start_epoch"]
        self.valid_interval = config["train"]["valid_interval"]
        self.visual_interval = config["train"]["visual_interval"]

        # init cudnn
        torch.backends.cudnn.enabled = self.use_cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # init common args
        self.start_epoch = 1
        self.best_score = np.finfo(np.float32).max

        # init path
        self.checkpoints_path = os.path.join(self.base_path, "checkpoints", "base")
        self.logs_path = os.path.join(self.base_path, "logs", "train", "base")
        # mkdir path
        prepare_empty_path([self.checkpoints_path, self.logs_path], self.resume)

        # init amp
        self.scaler = GradScaler(enabled=self.use_amp)

        # init optimizer
        self.optimizer = getattr(torch.optim, config["optimizer"]["name"])(
            params=self.model.parameters(), lr=config["optimizer"]["lr"],
        )

        # init lr scheduler
        self.scheduler_metric = 0.0
        self.scheduler = getattr(torch.optim.lr_scheduler, config["lr_scheduler"]["name"])(
            self.optimizer,
            mode=config["lr_scheduler"]["mode"],
            factor=config["lr_scheduler"]["factor"],
            patience=config["lr_scheduler"]["patience"],
            threshold=config["lr_scheduler"]["threshold"],
            min_lr=config["lr_scheduler"]["min_lr"],
            verbose=config["lr_scheduler"]["verbose"],
        )

        # print params
        self.print_networks()

    def print_networks(self):
        summary(self.model, input_size=(2, 1, 257, 201))

    @staticmethod
    def loss(enh_lps, clean_lps):
        out = (enh_lps - clean_lps) ** 2
        out = torch.sqrt(torch.mean(out, dim=1))
        out = torch.mean(out, dim=1)
        return torch.mean(out)

    def load_pre_model(self):
        load_model = torch.load(self.pre_model_path, map_location="cpu")
        self.model.load_state_dict(load_model)

        print(f"Load pre model done...")

    def init_logs(self):
        # init logs
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.logs_path, f"start_epoch_{self.start_epoch}"), max_queue=5, flush_secs=60,
        )

        # logs config
        self.writer.add_text(
            tag="config", text_string=f"<pre>  \n{toml.dumps(self.config)}  \n</pre>", global_step=1,
        )

    def save_checkpoint(self, epoch, is_best_epoch=False):
        print(f"Saving {epoch} epoch checkpoint, {is_best_epoch} best checkpoint...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

        checkpoint_name_prefix = "best" if is_best_epoch else "latest"
        torch.save(state_dict, os.path.join(self.checkpoints_path, checkpoint_name_prefix + "_checkpoint.tar"))
        torch.save(state_dict["model"], os.path.join(self.checkpoints_path, checkpoint_name_prefix + "_model.pth"))

    def resume_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoints_path, "latest_checkpoint.tar")
        assert os.path.exists(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model.load_state_dict(checkpoint["model"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        print(f"Latest checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

    def is_best_epoch(self, score):
        if score < self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def audio_visualization(self, noisy, clean, enh, name, epoch):
        self.writer.add_audio(f"audio/{name}/noisy", noisy, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/{name}/clean", clean, epoch, sample_rate=self.sr)
        self.writer.add_audio(f"audio/{name}/enh", enh, epoch, sample_rate=self.sr)

        # Visualize the spectrogram of noisy speech, clean speech, and enhanced speech
        noisy_mag, _ = librosa.magphase(librosa.stft(noisy, n_fft=320, hop_length=160, win_length=320))
        clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))
        enh_mag, _ = librosa.magphase(librosa.stft(enh, n_fft=320, hop_length=160, win_length=320))
        fig, axes = plt.subplots(3, 1, figsize=(6, 6))
        for k, mag in enumerate([noisy_mag, clean_mag, enh_mag]):
            axes[k].set_title(
                f"mean: {np.mean(mag):.3f}, "
                f"std: {np.std(mag):.3f}, "
                f"max: {np.max(mag):.3f}, "
                f"min: {np.min(mag):.3f}"
            )
            librosa.display.specshow(
                librosa.amplitude_to_db(mag, ref=np.max), cmap="magma", y_axis="linear", ax=axes[k], sr=16000
            )
        plt.tight_layout()
        self.writer.add_figure(f"spec/{name}", fig, epoch)

    def metrics_visualization(self, enh_list, clean_list, epoch, n_folds=1, n_jobs=8):
        # get metrics
        metrics = {
            "STOI": [],
            "WB_PESQ": [],
        }

        # compute enh metrics
        compute_metric(
            enh_list, clean_list, metrics, n_folds=n_folds, n_jobs=n_jobs, pre_load=True,
        )

        self.writer.add_scalar("STOI/valid", metrics["STOI"], epoch)
        self.writer.add_scalar("WB_PESQ/valid", metrics["WB_PESQ"], epoch)

        return ((metrics["STOI"]) + transform_pesq_range(metrics["WB_PESQ"])) / 2

    def update_scheduler(self, metric):
        self.scheduler.step(metric)

    def set_model_to_train_mode(self):
        self.model.train()

    def set_model_to_eval_mode(self):
        self.model.eval()

    def audio_stft(self, audio):
        # audio [B, S]
        # [B, S] -> [B, F, T]
        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=True,
        )

        mag = torch.abs(spec)
        lps = torch.log(mag ** 2 + EPS)
        phase = torch.angle(spec)
        return lps, phase

    def audio_istft(self, lps, phase):
        # lps [B, F, T]
        # phase [B, F, T]
        mag = torch.exp(lps / 2)
        # [B, F, T]
        cspec = mag * torch.exp(1j * phase)
        # [B, S]
        audio = torch.istft(
            cspec,
            self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            return_complex=False,
        )
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        return audio

    def train_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean in tqdm(self.train_iter, desc="train"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T]
            noisy_lps, _ = self.audio_stft(noisy)
            clean_lps, _ = self.audio_stft(clean)

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                noisy_lps = noisy_lps.unsqueeze(dim=1)
                enh_lps = self.model(noisy_lps)
                enh_lps = enh_lps.squeeze(dim=1)

            loss = self.loss(enh_lps, clean_lps)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

        # logs
        self.writer.add_scalar("loss/train", loss_total / len(self.train_iter), epoch)
        self.writer.add_scalar("lr", self.optimizer.state_dict()["param_groups"][0]["lr"], epoch)

    @torch.no_grad()
    def valid_epoch(self, epoch):
        loss_total = 0.0
        for noisy, clean, noisy_file in tqdm(self.valid_iter, desc="valid"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T]
            noisy_lps, noisy_phase = self.audio_stft(noisy)
            clean_lps, _ = self.audio_stft(clean)

            noisy_lps = noisy_lps.unsqueeze(dim=1)
            enh_lps = self.model(noisy_lps)
            enh_lps = enh_lps.squeeze(dim=1)

            loss = self.loss(enh_lps, clean_lps)

            loss_total += loss.item()

        # update learning rate
        self.update_scheduler(loss_total / len(self.valid_iter))

        # logs
        self.writer.add_scalar("loss/valid", loss_total / len(self.valid_iter), epoch)

        return loss_total / len(self.valid_iter)

    @torch.no_grad()
    def visual_epoch(self, epoch):
        for idx, noisy, clean, noisy_file in tqdm(enumerate(self.valid_iter), desc="visual"):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            # [B, S] -> [B, F, T]
            noisy_lps, noisy_phase = self.audio_stft(noisy)

            noisy_lps = noisy_lps.unsqueeze(dim=1)
            enh_lps = self.model(noisy_lps)
            enh_lps = enh_lps.squeeze(dim=1)

            enh = self.audio_istft(enh_lps, noisy_phase)
            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enh = enh.detach().squeeze(0).cpu().numpy()
            assert len(noisy) == len(clean) == len(enh)

            if idx > self.visual_samples:
                self.audio_visualization(noisy, clean, enh, os.path.basename(noisy_file), epoch)
            else:
                break

    def __call__(self):
        # to device
        self.model.to(self.device)

        # init pre load model
        if self.pre_model_path:
            self.load_pre_model()

        # resume
        if self.resume:
            self.resume_checkpoint()

        # init logs
        self.init_logs()

        # loop train
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"{'=' * 20} {epoch} epoch start {'=' * 20}")

            self.set_model_to_train_mode()
            self.train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self.save_checkpoint(epoch)

            # valid
            if epoch % self.valid_interval == 0 and epoch >= self.valid_start_epoch:
                print(f"Train has finished, Valid is in progress...")

                self.set_model_to_eval_mode()
                score = self.valid_epoch(epoch)

                if self.is_best_epoch(score):
                    self.save_checkpoint(epoch, is_best_epoch=True)

            # visual
            if epoch % self.visual_interval == 0:
                self.visual_epoch(epoch)

            print(f"{'=' * 20} {epoch} epoch end {'=' * 20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="base trainer")
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

    # get train_iter
    train_set = DNS_Dataset(dataset_path, config, mode="train")
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size[0],
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    # get valid_iter
    valid_set = DNS_Dataset(dataset_path, config, mode="valid")
    valid_iter = DataLoader(
        valid_set,
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

    # trainer
    trainer = BaseTrainer(config, model, train_iter, valid_iter, device)

    # train
    trainer()

    pass
