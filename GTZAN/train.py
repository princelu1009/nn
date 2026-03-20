#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a 2-layer LSTM genre classifier from a CSV split file.

CSV format:
ID,label,set
00776.au,disco,train
xxxxx.au,jazz,val
yyyyy.au,rock,test

- Extract features on-the-fly with librosa:
  MFCC(13) + centroid(1) + chroma(12) + contrast(7) = 33 dims
- Sequence length fixed to timeseries_length (default 128).
- Output:
  - checkpoint_best.pt (best by val accuracy)
  - label_map.json (string label <-> index)
  - training log printed to stdout

---------------------------
High-level idea (for students):
1) Read CSV -> split into train/val/test rows.
2) For each audio file, extract a fixed-length time-series feature matrix (128 x 33).
3) Feed sequences into an LSTM -> get the last timestep output -> classify genre.
4) Train on train set, validate on val set, save the best checkpoint.
"""

import os
import json
import math
import argparse
import copy
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) Model definition
# ============================================================
def infer_input_dim(n_mfcc: int) -> int:
    """
    Input dim after feature stacking:
      MFCC + delta + delta2 + centroid + RMS + chroma + contrast
    """
    return (n_mfcc * 3) + 1 + 1 + 12 + 7


class LSTMClassifier(nn.Module):
    """
    A simple sequence classifier:
      - LSTM reads an input sequence of shape (seq_len, batch, input_dim)
      - We take the output at the LAST timestep (out[-1]) as the summary
      - Then a Linear layer maps it to class logits

    This is a common baseline for time-series classification.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # nn.LSTM:
        # input_dim  = feature dimension per timestep (33 by default)
        # hidden_dim = hidden size of LSTM state (128 by default)
        # num_layers = stacked LSTM layers (2 by default)
        #
        # Note: PyTorch applies LSTM dropout ONLY when num_layers > 1.
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )

        # self.rnn=nn.RNN(
        #     input_dim,
        #     hidden_dim,
        #     num_layers,
        #     dropout=(dropout if num_layers > 1 else 0.0),
        #     bidirectional=bidirectional,
        # )
    



        # Pool sequence outputs instead of using only the last step.
        self.norm = nn.LayerNorm(lstm_out_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(lstm_out_dim * 2, output_dim)

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Input:
          x: Tensor of shape (seq_len, batch, input_dim)
          hidden: optional initial hidden state for LSTM

        Output:
          logits: (batch, output_dim)  (unnormalized scores for each class)
          hidden: final hidden state tuple (h_n, c_n)
        """
        # out, hidden = self.lstm(x, hidden)
        out, hidden = self.rnn(x, hidden)

        mean_pool = out.mean(dim=0)
        max_pool = out.max(dim=0).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        pooled = self.dropout(self.norm(pooled))
        logits = self.linear(pooled)
        return logits, hidden

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
    timeseries_length: int = 128,
    hop_length: int = 512,
    n_mfcc: int = 13,
    target_sr: int | None = None,
) -> np.ndarray:
    """
    Extract a time-series feature matrix from an audio file.

    We compute:
      - MFCC:               (13, T)
      - Spectral centroid:  (1,  T)
      - Chroma:             (12, T)
      - Spectral contrast:  (7,  T)

    Concatenate along feature axis -> (33, T)

    Then we force a FIXED time length:
      - If T < timeseries_length: zero-pad to the right
      - If T > timeseries_length: truncate

    Finally return shape:
      (timeseries_length, 33)  e.g. (128, 33)

    Why fixed length?
    - Batching in neural nets is easiest when every example has the same shape.
    """
    # librosa.load:
    # y: waveform (1D array)
    # sr: sampling rate
    #
    # If target_sr is None, librosa uses default 22050 (unless file has a different sr and you set sr=None).
    # For ML training we usually want consistency: either always resample to a fixed sr, or always keep original.
    y, sr = librosa.load(file_path, sr=target_sr)

    # Feature extraction per frame (hop_length controls frame step size)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    feats = np.concatenate(
        [mfcc, mfcc_delta, mfcc_delta2, centroid, rms, chroma, contrast],
        axis=0,
    ).astype(np.float32)

    # Force fixed time length to timeseries_length (default 128)
    T = feats.shape[1]
    if T < timeseries_length:
        # pad time axis with zeros on the right
        pad_width = timeseries_length - T
        feats = np.pad(feats, ((0, 0), (0, pad_width)), mode="constant")
    else:
        # truncate time axis
        feats = feats[:, :timeseries_length]

    # Normalize each feature channel across time to stabilize optimization.
    feat_mean = feats.mean(axis=1, keepdims=True)
    feat_std = feats.std(axis=1, keepdims=True)
    feats = (feats - feat_mean) / np.clip(feat_std, a_min=1e-6, a_max=None)

    # Convert (F, 128) -> (128, F) so each row is a timestep feature vector
    return feats.T


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
        n_mfcc: int,
        target_sr: int | None,
        cache_features: bool = False,
        fail_on_missing: bool = False,
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
        self.n_mfcc = n_mfcc
        self.target_sr = target_sr
        self.cache_features = cache_features
        self.fail_on_missing = fail_on_missing

        # simple in-memory cache: idx -> numpy array (128, 33)
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
            feats = np.zeros((self.timeseries_length, infer_input_dim(self.n_mfcc)), dtype=np.float32)
        else:
            # Use cached features if enabled and already computed
            if self.cache_features and idx in self._cache:
                feats = self._cache[idx]
            else:
                feats = extract_audio_features(
                    path,
                    timeseries_length=self.timeseries_length,
                    hop_length=self.hop_length,
                    n_mfcc=self.n_mfcc,
                    target_sr=self.target_sr,
                )
                if self.cache_features:
                    self._cache[idx] = feats

        # x: torch tensor of shape (seq_len, input_dim)
        x = torch.from_numpy(feats)

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
      x: (128, 33)

    Output:
      ids: list of strings length B
      xs: tensor (seq_len=128, B, input_dim=33)  <- LSTM expects seq-first format
      ys: tensor (B,)
    """
    ids = [b[0] for b in batch]

    # stack along batch dimension: (B, 128, 33)
    xs = torch.stack([b[1] for b in batch], dim=0)

    # labels: (B,)
    ys = torch.stack([b[2] for b in batch], dim=0)

    # LSTM in this code uses (seq_len, batch, input_dim)
    xs = xs.permute(1, 0, 2)  # (128, B, 33)
    return ids, xs, ys


def collate_test(batch):
    """
    Batch builder for test/inference.

    Input items: List[(id, x)]
    Output:
      ids: list[str]
      xs:  (128, B, 33)
    """
    ids = [b[0] for b in batch]
    xs = torch.stack([b[1] for b in batch], dim=0)  # (B, 128, 33)
    xs = xs.permute(1, 0, 2)                        # (128, B, 33)
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
    hidden_dim: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    validate_every: int
    timeseries_length: int
    hop_length: int
    n_mfcc: int
    target_sr: int | None
    num_workers: int
    seed: int
    bidirectional: bool
    grad_clip: float
    label_smoothing: float


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


def run_eval(model, loader, device, loss_fn):
    """
    Evaluate on validation set.

    Steps:
      - model.eval() disables dropout etc.
      - torch.no_grad() disables gradient tracking (faster + less memory)
      - compute average loss and average accuracy across batches
    """
    model.eval()
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
            total_acc += LSTMClassifier.accuracy(logits, y)
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
    ap.add_argument("--input_dim", type=int, default=33)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=100)

    # optimization hyperparams
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--validate_every", type=int, default=1, help="Validate every N epochs")

    # feature hyperparams (external adjustable)
    ap.add_argument("--timeseries_length", type=int, default=128)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--n_mfcc", type=int, default=13)

    # For target_sr:
    # - If 0: use librosa default behavior (sr=None in our code becomes None => default 22050)
    # - Else: resample audio to this sampling rate
    ap.add_argument("--target_sr", type=int, default=0, help="0 means librosa default; else resample to this SR")

    # dataloader settings
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--cache_features", action="store_true", help="Cache extracted features in RAM (faster, more RAM)")
    ap.add_argument("--fail_on_missing", action="store_true", help="Raise error if any audio file missing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_bidirectional", action="store_true", help="Use a one-directional LSTM")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    ap.add_argument("--label_smoothing", type=float, default=0.05, help="Cross-entropy label smoothing")

    args = ap.parse_args()

    # Convert target_sr from int argument to either None or actual sr
    target_sr = None if args.target_sr == 0 else int(args.target_sr)

    inferred_input_dim = infer_input_dim(args.n_mfcc)
    if args.input_dim not in (0, inferred_input_dim):
        print(f"[WARN] overriding input_dim={args.input_dim} with inferred feature dim {inferred_input_dim}")

    # Store all hyperparams in a single object (easy to save)
    hp = HParams(
        input_dim=inferred_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        validate_every=args.validate_every,
        timeseries_length=args.timeseries_length,
        hop_length=args.hop_length,
        n_mfcc=args.n_mfcc,
        target_sr=target_sr,
        num_workers=args.num_workers,
        seed=args.seed,
        bidirectional=not args.disable_bidirectional,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
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
        n_mfcc=hp.n_mfcc,
        target_sr=hp.target_sr,
        cache_features=args.cache_features,
        fail_on_missing=args.fail_on_missing
    )

    # shuffle=True for training (important!)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=hp.num_workers,
        collate_fn=collate_train,
        drop_last=False,
    )

    val_loader = None
    if len(val_df) > 0:
        val_ds = AudioCSVDataset(
            val_df,
            args.audio_root,
            label2idx,
            timeseries_length=hp.timeseries_length,
            hop_length=hp.hop_length,
            n_mfcc=hp.n_mfcc,
            target_sr=hp.target_sr,
            cache_features=args.cache_features,
            fail_on_missing=args.fail_on_missing
        )

        # shuffle=False for validation
        val_loader = DataLoader(
            val_ds,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers,
            collate_fn=collate_train,
            drop_last=False
        )

    # --------------------------
    # (F) Build model, loss, optimizer
    # --------------------------
    model = LSTMClassifier(
        input_dim=hp.input_dim,
        hidden_dim=hp.hidden_dim,
        output_dim=num_classes,
        num_layers=hp.num_layers,
        dropout=hp.dropout,
        bidirectional=hp.bidirectional,
    ).to(device)

    # CrossEntropyLoss expects:
    # - logits (B, C) (no softmax needed)
    # - targets (B,) class indices
    loss_fn = nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)

    # Adam optimizer is a strong default for many problems
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if val_loader is not None else "min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    # Track best validation accuracy and save best checkpoint
    best_val_acc = -1.0
    best_state = None
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp.grad_clip)
            optimizer.step()

            running_loss += loss.item()
            running_acc += LSTMClassifier.accuracy(logits, y)
            n_batches += 1

        # Average metrics over batches
        train_loss = running_loss / max(n_batches, 1)
        train_acc = running_acc / max(n_batches, 1)

        msg = f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | train_acc={train_acc:.2f}"
        current_score = train_loss

        # --------------------------
        # (H) Validation (optional)
        # --------------------------
        do_val = (val_loader is not None) and (epoch % hp.validate_every == 0)
        if do_val:
            val_loss, val_acc = run_eval(model, val_loader, device, loss_fn)
            msg += f" | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}"
            current_score = val_acc

            # If this epoch achieves better val accuracy, save it as "best"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                msg += f"\n[BEST] -> saved {os.path.basename(best_path)}"

        scheduler.step(current_score)
        msg += f" | lr={optimizer.param_groups[0]['lr']:.6f}"

        print(msg)

        # --------------------------
        # (I) Periodic checkpoint saving (every 10 epochs)
        # --------------------------
        # Useful when:
        # - training crashes midway
        # - you want to compare intermediate models
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2idx": label2idx,
                    "idx2label": idx2label,
                    "hparams": asdict(hp) | {"num_classes": num_classes},
                    "epoch": epoch,
                },
                ckpt_path,
            )

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = math.nan

    torch.save(
        {
            "model_state_dict": best_state,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "hparams": asdict(hp) | {"num_classes": num_classes},
            "best_val_acc": best_val_acc,
            "epoch": hp.epochs,
        },
        best_path,
    )

    # --------------------------
    # (J) Training summary
    # --------------------------
    print(f"[Done] Best val acc: {best_val_acc:.2f} | ckpt: {best_path}")
    print(f"[Note] test rows in CSV = {len(test_df)} (test is not used during training).")


if __name__ == "__main__":
    # Entry point: running "python train.py ..." will call main()
    main()
