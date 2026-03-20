#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predict test-set labels from CSV (set=test) and export pred.csv

Output pred.csv default columns:
ID,label

This script loads checkpoint_best.pt produced by train_from_csv.py.

--------------------------------
High-level idea (for students):
1) Read the same CSV split file, but only keep rows where set == "test".
2) For each test audio file:
   - Extract the SAME features used in training (128 x 33).
   - Feed into the trained LSTM model.
   - Take argmax over class logits to get predicted class index.
   - Map class index -> string label using idx2label.
3) Save predictions to pred.csv in Kaggle-friendly format.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import librosa

import torch
import torch.nn as nn


# ============================================================
# 1) Model definition (must match training architecture)
# ============================================================
class LSTMClassifier(nn.Module):
    """
    Same model structure as training:
      - LSTM reads (seq_len, batch, input_dim)
      - last timestep output -> Linear -> logits (batch, num_classes)

    IMPORTANT:
    - The model hyperparameters (input_dim, hidden_dim, num_layers, dropout)
      must match what was used during training, otherwise shape mismatch occurs
      when loading weights.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        # dropout in nn.LSTM only applies when num_layers > 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        """
        x: (seq_len, batch, input_dim)
        returns logits: (batch, output_dim)
        """
        out, hidden = self.lstm(x, hidden)
        logits = self.linear(out[-1])  # last timestep: (batch, hidden_dim) -> (batch, num_classes)
        return logits, hidden


# ============================================================
# 2) Feature extraction (must match training feature pipeline)
# ============================================================
def extract_audio_features(file_path: str, timeseries_length=128, hop_length=512, n_mfcc=13, target_sr=None):
    """
    Extract features from audio file and force fixed sequence length.

    Output shape:
      (timeseries_length, 33) e.g. (128, 33)

    33 dims come from:
      MFCC(13) + centroid(1) + chroma(12) + contrast(7) = 33
    """
    # Load audio waveform
    # y: waveform, sr: sampling rate
    # If target_sr is not None, librosa will resample audio to target_sr
    y, sr = librosa.load(file_path, sr=target_sr)

    # Extract frame-level features (all share the same time dimension T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)              # (13, T)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)             # (1,  T)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)                     # (12, T)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)             # (7,  T)

    # Concatenate into (33, T)
    feats = np.concatenate([mfcc, centroid, chroma, contrast], axis=0).astype(np.float32)

    # Pad / truncate time axis to timeseries_length
    T = feats.shape[1]
    if T < timeseries_length:
        feats = np.pad(feats, ((0, 0), (0, timeseries_length - T)), mode="constant")
    else:
        feats = feats[:, :timeseries_length]

    # Convert to (128, 33) where each row is a timestep feature vector
    return feats.T


# ============================================================
# 3) Main inference flow
# ============================================================
def main():
    # --------------------------
    # (A) Parse command line args
    # --------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True, help="e.g. checkpoint_csv/checkpoint_best.pt")
    ap.add_argument("--out_csv", type=str, default="pred.csv")

    # Feature parameters:
    # MUST match training configuration, otherwise model sees different input distribution/shape.
    ap.add_argument("--timeseries_length", type=int, default=128)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--n_mfcc", type=int, default=13)
    ap.add_argument("--target_sr", type=int, default=0)  # 0 means librosa default (we convert to None)

    # Model parameters:
    # MUST match training architecture, otherwise model weights cannot be loaded correctly.
    ap.add_argument("--input_dim", type=int, default=33)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)

    # If True, missing audio files will raise an error and stop inference.
    # If False, we will provide a fallback prediction.
    ap.add_argument("--fail_on_missing", action="store_true")
    args = ap.parse_args()

    # Convert target_sr argument:
    # - args.target_sr == 0 => None (librosa default behavior)
    # - else => resample to that sampling rate
    target_sr = None if args.target_sr == 0 else int(args.target_sr)

    # --------------------------
    # (B) Load checkpoint and label mapping
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # map_location ensures checkpoint can be loaded on CPU/GPU safely
    ckpt = torch.load(args.ckpt_path, map_location=device)

    # The training script saves idx2label inside checkpoint_best.pt.
    # We use it to translate predicted class index back to human-readable label.
    if isinstance(ckpt, dict) and "idx2label" in ckpt:
        # JSON-like dict sometimes stores keys as strings, so we force int keys here
        idx2label = {int(k): v for k, v in ckpt["idx2label"].items()}
        num_classes = len(idx2label)
    else:
        raise ValueError("Checkpoint missing idx2label. Please use checkpoint produced by train_from_csv.py")

    # --------------------------
    # (C) Rebuild model and load weights
    # --------------------------
    model = LSTMClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Training checkpoint format:
    # - either a dict with key "model_state_dict"
    # - or directly a state_dict
    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    model.load_state_dict(state_dict)

    # model.eval() disables dropout and sets layers to inference mode
    model.eval()

    # --------------------------
    # (D) Read CSV and select test rows
    # --------------------------
    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]

    # normalize strings to avoid issues like extra spaces
    df["ID"] = df["ID"].astype(str).str.strip()
    df["set"] = df["set"].astype(str).str.strip().str.lower()

    # Only predict for set == "test"
    test_df = df[df["set"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test rows found (set=test).")

    # --------------------------
    # (E) Predict each test file
    # --------------------------
    preds = []

    # We loop through each test ID and run feature extraction + model inference
    for audio_id in test_df["ID"].tolist():
        path = os.path.join(args.audio_root, audio_id)

        # Handle missing files
        if not os.path.isfile(path):
            msg = f"Missing audio file: {path}"
            if args.fail_on_missing:
                raise FileNotFoundError(msg)

            # Fallback strategy:
            # If audio is missing and we do not fail, we assign class 0.
            # (Alternative: skip this sample, or output "unknown".)
            pred_label = idx2label[0]
            preds.append((audio_id, pred_label))
            continue

        # 1) Extract feature matrix: (128, 33)
        feats = extract_audio_features(
            path,
            timeseries_length=args.timeseries_length,
            hop_length=args.hop_length,
            n_mfcc=args.n_mfcc,
            target_sr=target_sr,
        )

        # 2) Convert to torch tensor and add batch dimension
        # feats: (128, 33)
        # unsqueeze(0) -> (1, 128, 33) where batch=1
        x = torch.from_numpy(feats).unsqueeze(0)

        # 3) Convert to LSTM expected shape: (seq_len, batch, input_dim)
        # (1, 128, 33) -> permute -> (128, 1, 33)
        x = x.permute(1, 0, 2).to(device)

        # 4) Inference: disable gradient for speed + memory
        with torch.no_grad():
            logits, _ = model(x)  # logits: (1, num_classes)

            # argmax gives predicted class index
            pred_idx = int(torch.argmax(logits, dim=1).item())

            # map numeric index -> string label
            pred_label = idx2label[pred_idx]

        preds.append((audio_id, pred_label))

    # --------------------------
    # (F) Save prediction CSV
    # --------------------------
    out = pd.DataFrame(preds, columns=["ID", "label"])

    # Kaggle usually expects: ID,label (no index column)
    out.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {args.out_csv}  rows={len(out)}")


if __name__ == "__main__":
    # Script entry point
    main()
