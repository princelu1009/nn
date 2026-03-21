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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


csv_path="/Users/princelu/Desktop/NN/GTZAN/gtzan.csv"


class TransformerGenreClassifier(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, num_layers: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Linear Projection: Map 33 features to a model dimension (e.g., 64 or 128)
        # Transformers work best when the "d_model" is divisible by num_heads
        self.d_model = 64 
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        # 2. Positional Encoding: Tells the model which frame comes first/last
        self.pos_encoder = nn.Parameter(torch.zeros(1, 128, self.d_model)) 
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=num_heads, 
            dim_feedforward=self.d_model * 4, 
            dropout=dropout,
            batch_first=True  # Keeps shape (Batch, Time, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.fc = nn.Linear(self.d_model, output_dim)

    def forward(self, x):
        # x input: (Time, Batch, Feats) from your collate_train -> (128, B, 33)
        # 1. Switch to (Batch, Time, Feats) for Transformer batch_first=True
        x = x.permute(1, 0, 2) 
        
        # 2. Project features to d_model space
        x = self.input_projection(x) # (B, 128, 64)
        
        # 3. Add Positional Encoding
        x = x + self.pos_encoder
        
        # 4. Pass through Transformer
        # out shape: (B, 128, 64)
        out = self.transformer_encoder(x)
        
        # 5. Global Average Pooling (or take the first token)
        # Instead of out[-1], we average across the time dimension for a global summary
        out = out.mean(dim=1) 
        
        logits = self.fc(out)
        return logits, None
    

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


df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

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

label2idx, idx2label = build_label_map(train_df)
print(label2idx)
print(idx2label)
num_classes = len(label2idx)


# Save label mapping + hyperparams for reproducibility
# with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
#     json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, ensure_ascii=False, indent=2)

# with open(os.path.join(args.out_dir, "hparams.json"), "w", encoding="utf-8") as f:
#     json.dump(asdict(hp) | {"num_classes": num_classes}, f, ensure_ascii=False, indent=2)


#device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available(): # For Apple Silicon Macs
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[Device] {device}")
print(f"[Classes] {num_classes}: {list(label2idx.keys())}")

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
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)          # (13, T)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)         # (1,  T)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)                 # (12, T)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)         # (7,  T)

    # Concatenate along feature dimension -> (33, T)
    feats = np.concatenate([mfcc, centroid, chroma, contrast], axis=0).astype(np.float32)

    # Force fixed time length to timeseries_length (default 128)
    T = feats.shape[1]
    if T < timeseries_length:
        # pad time axis with zeros on the right
        pad_width = timeseries_length - T
        feats = np.pad(feats, ((0, 0), (0, pad_width)), mode="constant")
    else:
        # truncate time axis
        feats = feats[:, :timeseries_length]

    # Convert (33, 128) -> (128, 33) so each row is a timestep feature vector
    return feats.T


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
            feats = np.zeros((self.timeseries_length, 33), dtype=np.float32)
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

        # x: torch tensor of shape (128, 33)
        x = torch.from_numpy(feats)

        # If label2idx is None, we are in inference/test mode.
        if self.label2idx is None:
            return audio_id, x

        # Convert label string -> numeric class index
        if label_str not in self.label2idx:
            raise ValueError(f"Label '{label_str}' not in label2idx. Check CSV label normalization.")
        y = torch.tensor(self.label2idx[label_str], dtype=torch.long)

        return audio_id, x, y


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


ap = argparse.ArgumentParser()

    # data paths
ap.add_argument("--csv_path", type=str, required=False, help="CSV with columns: ID,label,set")
ap.add_argument("--audio_root", type=str, required=False, help="Root folder containing audio files")
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

args = ap.parse_args()

# Convert target_sr from int argument to either None or actual sr
target_sr = None if args.target_sr == 0 else int(args.target_sr)

args = ap.parse_args()

hp = HParams(
        input_dim=args.input_dim,
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
    )


train_ds = AudioCSVDataset(
        train_df,
        "/Users/princelu/Desktop/NN/GTZAN/genres",
        label2idx,
        timeseries_length=hp.timeseries_length,
        hop_length=hp.hop_length,
        n_mfcc=hp.n_mfcc,
        target_sr=hp.target_sr,
        cache_features=args.cache_features,
        fail_on_missing=args.fail_on_missing
    )
# shuffle=True for training (important!)
# train_loader = DataLoader(
#     train_ds,
#     batch_size=hp.batch_size,
#     shuffle=True,
#     num_workers=hp.num_workers,
#     collate_fn=collate_train,
#     drop_last=True  # drop last incomplete batch (keeps batch size consistent)
# )


one_sample = train_ds[1]
audio_id, x, y = one_sample

print("audio_id:", audio_id)
print("x shape:", x.shape)
print("y:", y)

x_batch = x.unsqueeze(1)

model = TransformerGenreClassifier(
    input_dim=33,
    num_heads=4,
    num_layers=2,
    output_dim=num_classes,
)

logits, _ = model(x_batch)

print("batched input shape:", x_batch.shape)
print("logits shape:", logits.shape)
print("logits:", logits)
print("pred class idx:", logits.argmax(dim=1))
