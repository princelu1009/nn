#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import math
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import random

######data augumentation
def apply_spec_augment(x, time_mask_max=30, feature_mask_max=6, num_masks=2):
    """
    Apply masking to the feature tensor.
    x shape: (Time, Features, Channels) -> e.g., (256, 33, 3)
    """
    T, F, _ = x.shape
    x_aug = x.clone()

    # 1. Time Masking (Vertical strips)
    for _ in range(num_masks):
        t = random.randint(0, time_mask_max)
        t0 = random.randint(0, T - t)
        x_aug[t0 : t0 + t, :, :] = 0

    # 2. Feature Masking (Horizontal strips)
    # We treat your 33 dimensions as the frequency axis
    for _ in range(num_masks):
        f = random.randint(0, feature_mask_max)
        f0 = random.randint(0, F - f)
        x_aug[:, f0 : f0 + f, :] = 0

    return x_aug

class TransformerGenreClassifier(nn.Module):
    def __init__(self, timeseries_length, input_dim, num_heads, num_layers, output_dim, dropout=0.4, d_model=128):
        super().__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, self.d_model)
        
        # 1. Convolutional Feature Extractor (The "Front-end")
        # Project per-timestep features first, then use 1D convolutions over time.
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Learnable Class Token (CLS)
        # We add an extra "summary" token at the start of the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        
        # 3. Learnable Positional Encoding
        # Updated to (timeseries_length + 1) to account for the CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, timeseries_length + 1, self.d_model))
        
        # 4. Transformer Encoder with Pre-LayerNorm (More stable for deep models)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation='gelu', # GELU is often better than ReLU for Transformers
            batch_first=True,
            norm_first=True    # Pre-LayerNorm architecture
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. MLP Head
        self.norm = nn.LayerNorm(self.d_model)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, output_dim)
        )

    def forward(self, x):
        # x: (Time, Batch, Mel, Channel) -> flatten mel/channel into Conv1d channels
        x = x.permute(1, 2, 3, 0).reshape(x.shape[1], -1, x.shape[0])
        # x : (B, input_dim , Time)
        x = x.transpose(1, 2)
        # x : (B, Time, input_dim)
        x = self.input_proj(x)
        # (B, T, d_model)
        x = x.transpose(1, 2)
        # (B, d_model, T)

        # Extract features
        x = self.conv_extractor(x)
        x = x.transpose(1, 2)
        
        # Prepend CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 129, 128)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Take the output of the CLS token only
        cls_out = self.norm(x[:, 0]) 
        
        logits = self.mlp_head(cls_out)
        return logits, None
 
    @staticmethod
    def accuracy(logits, target_idx):
        """
        Utility: compute classification accuracy (percentage).

        logits: (B, C)
        target_idx: (B,)
        """
        pred = torch.argmax(logits, dim=1)
        return (pred == target_idx).float().mean().item() * 100.0


# ============================================================
# 2) Feature extraction (audio -> fixed-length sequence)
# ============================================================
def extract_audio_features(
    file_path: str,
    timeseries_length: int = 256,
    hop_length: int = 256,
    n_mels: int = 33,
    target_sr: int | None = None,
) -> np.ndarray:
    """
    Extract a fixed-length 3-channel log-mel tensor from an audio file.

    Channels:
      - original waveform mel
      - harmonic mel
      - percussive mel

    Returned shape:
      (timeseries_length, n_mels, 3) e.g. (256, 33, 3)
    """
    y, sr = librosa.load(file_path, sr=target_sr)

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    s_orig = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_s_orig = librosa.power_to_db(s_orig, ref=np.max)

    s_harm = librosa.feature.melspectrogram(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_s_harm = librosa.power_to_db(s_harm, ref=np.max)

    s_perc = librosa.feature.melspectrogram(
        y=y_percussive,
        sr=sr,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_s_perc = librosa.power_to_db(s_perc, ref=np.max)

    feats = np.stack([log_s_orig, log_s_harm, log_s_perc], axis=0).astype(np.float32)

    _, _, T = feats.shape
    if T < timeseries_length:
        pad_width = timeseries_length - T
        feats = np.pad(feats, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
    else:
        feats = feats[:, :, :timeseries_length]

    return feats.transpose(2, 1, 0)
# ============================================================
# 3) Dataset: reading CSV rows and loading audio
# ============================================================
class AudioCSVDataset(Dataset):
    """
    A PyTorch Dataset that:
      - reads one row at a time from a dataframe
      - resolves the audio file path
      - extracts features (128 x 33)
      - returns tensors for training or inference

    For train/val:
      returns (audio_id, x, y)
    For test:
      returns (audio_id, x)   (no label)
    """
    def __init__(
        self,
        df: "pd.DataFrame",
        audio_root: str,
        label2idx: Dict[str, int] | None,
        timeseries_length: int,
        hop_length: int,
        n_mels: int,
        target_sr: int | None,
        cache_features: bool = False,
        fail_on_missing: bool = False,
        is_train:bool = False,
    ):
        """
        df columns required: ID, label, set
        label2idx: None allowed for test set (no labels)

        cache_features:
          - If True, store extracted features in RAM after first extraction.
          - Faster for repeated epochs, but increases memory usage.
        fail_on_missing:
          - If True, raise error when audio file missing.
          - If False, return a zero feature matrix (acts like a "blank" audio).
        """
        self.df = df.reset_index(drop=True).copy()
        self.audio_root = audio_root
        self.label2idx = label2idx
        self.timeseries_length = timeseries_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_sr = target_sr
        self.cache_features = cache_features
        self.fail_on_missing = fail_on_missing
        self.is_train = is_train

        # simple in-memory cache: idx -> numpy array (timeseries_length, n_mels, 3)
        self._cache: Dict[int, np.ndarray] = {}

    def __len__(self):
        # number of rows/examples
        return len(self.df)

    def _resolve_path(self, audio_id: str) -> str:
        """
        Convert the ID in CSV into an actual file path.
        If audio_id already contains subfolders, os.path.join still works.
        """
        return os.path.join(self.audio_root, audio_id)

    def __getitem__(self, idx: int):
        """
        One sample from the dataset.

        Returns:
          - train/val mode: (audio_id, x, y)
          - test mode:      (audio_id, x)
        """
        row = self.df.iloc[idx]
        audio_id = str(row["ID"]).strip()

        # label might be missing (e.g. test set)
        label_str = None if pd.isna(row.get("label", np.nan)) else str(row.get("label", "")).strip()

        path = self._resolve_path(audio_id)

        # Handle missing file
        if not os.path.isfile(path):
            msg = f"Missing audio file: {path}"
            if self.fail_on_missing:
                raise FileNotFoundError(msg)

            # If missing file and not failing, return zero features.
            # This keeps training running but might hurt accuracy if many are missing.
            feats = np.zeros((self.timeseries_length, self.n_mels, 3), dtype=np.float32)
        else:
            # Use cached features if enabled and already computed
            if self.cache_features and idx in self._cache:
                feats = self._cache[idx]
            else:
                feats = extract_audio_features(
                    path,
                    timeseries_length=self.timeseries_length,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    target_sr=self.target_sr,
                )
                if self.cache_features:
                    self._cache[idx] = feats

        # x: torch tensor of shape (timeseries_length, n_mels, 3)
        x = torch.from_numpy(feats)

       
        if self.is_train:
            x = apply_spec_augment(x)

        # If label2idx is None, we are in inference/test mode.
        if self.label2idx is None:
            return audio_id, x

        # Convert label string -> numeric class index
        if label_str not in self.label2idx:
            raise ValueError(f"Label '{label_str}' not in label2idx. Check CSV label normalization.")
        y = torch.tensor(self.label2idx[label_str], dtype=torch.long)

        return audio_id, x, y


# ============================================================
# 4) Collate functions: how to batch variable items
# ============================================================
def collate_train(batch):
    """
    Batch builder for train/val.

    Input batch items: List[(id, x, y)]
      x: (256, 33, 3)

    Output:
      ids: list of strings length B
      xs: tensor (seq_len=256, B, n_mels=33, channels=3)
      ys: tensor (B,)
    """
    ids = [b[0] for b in batch]

    # stack along batch dimension: (B, 256, 33, 3)
    xs = torch.stack([b[1] for b in batch], dim=0)

    # labels: (B,)
    ys = torch.stack([b[2] for b in batch], dim=0)

    xs = xs.permute(1, 0, 2, 3)  # (256, B, 33, 3)
    return ids, xs, ys


def collate_test(batch):
    """
    Batch builder for test/inference.

    Input items: List[(id, x)]
    Output:
      ids: list[str]
      xs:  (256, B, 33, 3)
    """
    ids = [b[0] for b in batch]
    xs = torch.stack([b[1] for b in batch], dim=0)  # (B, 256, 33, 3)
    xs = xs.permute(1, 0, 2, 3)                     # (256, B, 33, 3)
    return ids, xs


# ============================================================
# 5) Training utilities and hyper-parameter container
# ============================================================
@dataclass
class HParams:
    """
    A dataclass just to store hyperparameters neatly and save them to JSON.
    Helpful for reproducibility.
    """
    input_dim: int
    d_model: int
    num_heads: int
    # hidden_dim: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    validate_every: int
    timeseries_length: int
    hop_length: int
    n_mels: int
    target_sr: int | None
    num_workers: int
    seed: int


def set_seed(seed: int):
    """
    Make training deterministic-ish (still not perfectly deterministic on GPU, but better).
    Fixes random seeds for:
      - python random
      - numpy
      - torch CPU/GPU
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_map(df: "pd.DataFrame") -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build mapping from string labels to integer indices.

    IMPORTANT:
    We only build from TRAIN labels.
    Why?
      - Avoid "data leakage" from val/test labels
      - In competitions, test labels may be hidden

    Returns:
      label2idx: e.g. {"disco":0, "jazz":1, ...}
      idx2label: reverse mapping
    """
    labels = sorted({str(x).strip() for x in df["label"].dropna().tolist()})
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label


def run_eval(model, loader, device):
    """
    Evaluate on validation set.

    Steps:
      - model.eval() disables dropout etc.
      - torch.no_grad() disables gradient tracking (faster + less memory)
      - compute average loss and average accuracy across batches
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for _, x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            total_acc += TransformerGenreClassifier.accuracy(logits, y)
            n_batches += 1

    if n_batches == 0:
        return math.nan, math.nan
    return total_loss / n_batches, total_acc / n_batches


# ============================================================
# 6) Main training script (argument parsing + training loop)
# ============================================================
def main():
    # --------------------------
    # (A) Parse command line args
    # --------------------------
    ap = argparse.ArgumentParser()

    # data paths
    ap.add_argument("--csv_path", type=str, required=True, help="CSV with columns: ID,label,set")
    ap.add_argument("--audio_root", type=str, required=True, help="Root folder containing audio files")
    ap.add_argument("--out_dir", type=str, default="checkpoint_csv", help="Output directory for checkpoints & maps")

    # model hyperparams (external adjustable)
    ap.add_argument("--input_dim", type=int, default=99)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=8)
    # ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)

    # optimization hyperparams
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--validate_every", type=int, default=1, help="Validate every N epochs")

    # feature hyperparams (external adjustable)
    ap.add_argument("--timeseries_length", type=int, default=256)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=33)

    # For target_sr:
    # - If 0: use librosa default behavior (sr=None in our code becomes None => default 22050)
    # - Else: resample audio to this sampling rate
    ap.add_argument("--target_sr", type=int, default=22050, help="0 means librosa default; else resample to this SR")

    # dataloader settings
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_features", action="store_true", help="Cache extracted features in RAM (faster, more RAM)")
    ap.add_argument("--fail_on_missing", action="store_true", help="Raise error if any audio file missing")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Convert target_sr from int argument to either None or actual sr
    target_sr = None if args.target_sr == 0 else int(args.target_sr)

    # Store all hyperparams in a single object (easy to save)
    effective_input_dim = args.n_mels * 3
    if args.input_dim != effective_input_dim:
        print(
            f"[WARN] Overriding --input_dim={args.input_dim} to match "
            f"3-channel features (n_mels * 3 = {effective_input_dim})."
        )

    hp = HParams(
        input_dim=effective_input_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        # hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        validate_every=args.validate_every,
        timeseries_length=args.timeseries_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        target_sr=target_sr,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if hp.timeseries_length >= 512 and hp.batch_size > 32:
        print(
            f"[WARN] timeseries_length={hp.timeseries_length} with batch_size={hp.batch_size} "
            "may use a lot of memory. Consider batch_size=32 or smaller."
        )

    # --------------------------
    # (B) Setup output folder + seed
    # --------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(hp.seed)

    # --------------------------
    # (C) Read CSV and split sets
    # --------------------------
    df = pd.read_csv(args.csv_path)

    # Normalize column names (trim spaces)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    if "ID" not in df.columns or "set" not in df.columns:
        raise ValueError("CSV must include columns: ID, set, and (for train/val) label.")
    if "label" not in df.columns:
        # If label column missing, create it (mostly for inference-only cases)
        df["label"] = np.nan

    # Normalize string values
    df["ID"] = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip()

    # Split by "set" column
    train_df = df[df["set"] == "train"].copy()
    val_df = df[df["set"].isin(["val", "dev", "valid", "validation"])].copy()
    test_df = df[df["set"] == "test"].copy()

    if len(train_df) == 0:
        raise ValueError("No train rows found (set=train).")
    if len(val_df) == 0:
        print("[WARN] No val/dev rows found. Validation will be skipped, and best ckpt won't be meaningful.")

    # Build label mapping from training data
    label2idx, idx2label = build_label_map(train_df)
    num_classes = len(label2idx)

    # Save label mapping + hyperparams for reproducibility
    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(hp) | {"num_classes": num_classes}, f, ensure_ascii=False, indent=2)

    # --------------------------
    # (D) Device selection (GPU if available)
    # --------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon Macs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[Device] {device}")
    print(f"[Classes] {num_classes}: {list(label2idx.keys())}")

    # --------------------------
    # (E) Build datasets + dataloaders
    # --------------------------
    train_ds = AudioCSVDataset(
        train_df,
        args.audio_root,
        label2idx,
        timeseries_length=hp.timeseries_length,
        hop_length=hp.hop_length,
        n_mels=hp.n_mels,
        target_sr=hp.target_sr,
        cache_features=args.cache_features,
        fail_on_missing=args.fail_on_missing,
        is_train=True
    )

    # shuffle=True for training (important!)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=hp.num_workers,
        collate_fn=collate_train,
        drop_last=True  # drop last incomplete batch (keeps batch size consistent)
    )

    val_loader = None
    if len(val_df) > 0:
        val_ds = AudioCSVDataset(
            val_df,
            args.audio_root,
            label2idx,
            timeseries_length=hp.timeseries_length,
            hop_length=hp.hop_length,
            n_mels=hp.n_mels,
            target_sr=hp.target_sr,
            cache_features=args.cache_features,
            fail_on_missing=args.fail_on_missing,
            is_train=False
        )

        # shuffle=False for validation
        val_loader = DataLoader(
            val_ds,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers,
            collate_fn=collate_train,
            drop_last=True
        )

    # --------------------------
    # (F) Build model, loss, optimizer
    # --------------------------
    # model = LSTMClassifier(
    #     input_dim=hp.input_dim,
    #     hidden_dim=hp.hidden_dim,
    #     output_dim=num_classes,
    #     num_layers=hp.num_layers,
    #     dropout=hp.dropout,
    # ).to(device)

    # In main():
    model = TransformerGenreClassifier(
        input_dim=hp.input_dim,
        timeseries_length=hp.timeseries_length,
        num_heads=hp.num_heads,
        num_layers=hp.num_layers,
        output_dim=num_classes,
        dropout=hp.dropout,
        d_model=hp.d_model,
    ).to(device)

    # CrossEntropyLoss expects:
    # - logits (B, C) (no softmax needed)
    # - targets (B,) class indices
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Adam optimizer is a strong default for many problems
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs)


    # Track best validation accuracy and save best checkpoint
    best_val_acc = -1.0
    best_path = os.path.join(args.out_dir, "checkpoint_best.pt")

    # --------------------------
    # (G) Training loop (epoch-based)
    # --------------------------
    for epoch in range(1, hp.epochs + 1):
        model.train()  # enable training behaviors like dropout
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        # Iterate batches
        for _, x, y in train_loader:
            x = x.to(device)  # (128, B, 33)
            y = y.to(device)  # (B,)

            optimizer.zero_grad()

            logits, _ = model(x)        # logits: (B, num_classes)
            loss = loss_fn(logits, y)   # scalar loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += TransformerGenreClassifier.accuracy(logits, y)
            n_batches += 1

        # Average metrics over batches
        train_loss = running_loss / max(n_batches, 1)
        train_acc = running_acc / max(n_batches, 1)
        msg = f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.2f}"

        # --------------------------
        # (H) Validation (optional)
        # --------------------------
        do_val = (val_loader is not None) and (epoch % hp.validate_every == 0)
        if do_val:
            val_loss, val_acc = run_eval(model, val_loader, device)
            msg += f" | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}"

            # If this epoch achieves better val accuracy, save it as "best"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "label2idx": label2idx,
                        "idx2label": idx2label,
                        "hparams": asdict(hp) | {"num_classes": num_classes},
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                    },
                    best_path,
                )
                msg += f"\n[BEST] -> saved {os.path.basename(best_path)}"

        scheduler.step()
        print(msg)

        # --------------------------
        # (I) Periodic checkpoint saving (every 10 epochs)
        # --------------------------
        # Useful when:
        # - training crashes midway
        # - you want to compare intermediate models
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

    # --------------------------
    # (J) Training summary
    # --------------------------
    print(f"[Done] Best val acc: {best_val_acc:.2f} | ckpt: {best_path}")
    print(f"[Note] test rows in CSV = {len(test_df)} (test is not used during training).")


if __name__ == "__main__":
    # Entry point: running "python train.py ..." will call main()
    main()
