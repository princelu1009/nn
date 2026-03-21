import pandas as pd
from typing import Dict, List, Tuple
import json 
import os
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa


csv_path="/Users/princelu/Desktop/NN/GTZAN/gtzan.csv"

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



train_ds = AudioCSVDataset(
    train_df,
    "/Users/princelu/Desktop/NN/GTZAN/genres",
    label2idx,
    timeseries_length=128,
    hop_length=512,
    n_mfcc=13,
    target_sr=0,
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
    drop_last=True  # drop last incomplete batch (keeps batch size consistent)
)
