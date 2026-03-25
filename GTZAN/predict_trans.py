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
   - Feed into the trained Transformer model.
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
import torch.nn.functional as F


# ============================================================
# 1) Model definition (must match training architecture)
# ============================================================

class TransformerGenreClassifier(nn.Module):
    def __init__(self, timeseries_length, input_dim, num_heads, num_layers, output_dim, dropout=0.2, d_model=64):
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

    def _get_pos_embed(self, seq_len: int) -> torch.Tensor:
        """
        Resize the learnable positional embedding if the active sequence length changes.
        """
        if self.pos_embed.shape[1] == seq_len:
            return self.pos_embed

        pos_embed = self.pos_embed.transpose(1, 2)
        pos_embed = F.interpolate(pos_embed, size=seq_len, mode="linear", align_corners=False)
        return pos_embed.transpose(1, 2)

    def forward(self, x):
        # x: (Time, Batch, Mel, Channel) -> flatten mel/channel into Conv1d channels
        x = x.permute(1, 2, 3, 0).reshape(x.shape[1], -1, x.shape[0])

        # Apply a per-timestep projection before the temporal convolution stack.
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        # Extract features
        x = self.conv_extractor(x) # (B, 128, 128)
        x = x.transpose(1, 2)      # (B, 128, 128) (Time back to middle)
        
        # Prepend CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 129, 128)
        
        # Add Positional Embedding
        x = x + self._get_pos_embed(x.shape[1])
        
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
# 2) Feature extraction (must match training feature pipeline)
# ============================================================
def extract_audio_features(file_path: str, timeseries_length=256, hop_length=256, n_mels=33, target_sr=None):
    """
    Extract a fixed-length 3-channel log-mel tensor.

    Output shape:
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
        feats = np.pad(feats, ((0, 0), (0, 0), (0, timeseries_length - T)), mode="constant")
    else:
        feats = feats[:, :, :timeseries_length]

    return feats.transpose(2, 1, 0)


def resize_checkpoint_pos_embed(state_dict, target_seq_len):
    """
    Resize checkpoint positional embeddings when sequence length changes.
    """
    pos_key = "pos_embed"
    if pos_key not in state_dict:
        return state_dict

    pos_embed = state_dict[pos_key]
    if pos_embed.ndim != 3 or pos_embed.shape[1] == target_seq_len:
        return state_dict

    resized = F.interpolate(
        pos_embed.transpose(1, 2),
        size=target_seq_len,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)

    updated_state_dict = dict(state_dict)
    updated_state_dict[pos_key] = resized
    return updated_state_dict


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
    ap.add_argument("--timeseries_length", type=int, default=512)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--n_mels", type=int, default=33)
    ap.add_argument("--target_sr", type=int, default=0)  # 0 means librosa default (we convert to None)

    # Model parameters:
    # MUST match training architecture, otherwise model weights cannot be loaded correctly.
    ap.add_argument("--input_dim", type=int, default=99)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=8)
    # ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)

    # If True, missing audio files will raise an error and stop inference.
    # If False, we will provide a fallback prediction.
    ap.add_argument("--fail_on_missing", action="store_true")
    args = ap.parse_args()

    # Convert target_sr argument:
    # - args.target_sr == 0 => None (librosa default behavior)
    # - else => resample to that sampling rate
    cli_target_sr = None if args.target_sr == 0 else int(args.target_sr)

    # --------------------------
    # (B) Load checkpoint and label mapping
    # --------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon Macs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


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

    ckpt_hparams = ckpt.get("hparams", {}) if isinstance(ckpt, dict) else {}
    model_input_dim = int(ckpt_hparams.get("input_dim", args.input_dim))
    timeseries_length = int(ckpt_hparams.get("timeseries_length", args.timeseries_length))
    hop_length = int(ckpt_hparams.get("hop_length", args.hop_length))
    n_mels = int(ckpt_hparams.get("n_mels", max(1, model_input_dim // 3)))
    d_model = int(ckpt_hparams.get("d_model", args.d_model))
    num_heads = int(ckpt_hparams.get("num_heads", args.num_heads))
    num_layers = int(ckpt_hparams.get("num_layers", args.num_layers))
    dropout = float(ckpt_hparams.get("dropout", args.dropout))
    target_sr = ckpt_hparams.get("target_sr", cli_target_sr)

    # --------------------------
    # (C) Rebuild model and load weights
    # --------------------------
    model = TransformerGenreClassifier(
        input_dim=model_input_dim,
        timeseries_length=timeseries_length,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=num_classes,
        dropout=dropout,
        d_model=d_model,
    ).to(device)

    # Training checkpoint format:
    # - either a dict with key "model_state_dict"
    # - or directly a state_dict
    state_dict = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
    state_dict = resize_checkpoint_pos_embed(state_dict, timeseries_length + 1)
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

        # 1) Extract feature tensor: (256, 33, 3)
        feats = extract_audio_features(
            path,
            timeseries_length=timeseries_length,
            hop_length=hop_length,
            n_mels=n_mels,
            target_sr=target_sr,
        )

        # 2) Convert to torch tensor and add batch dimension
        # feats: (256, 33, 3)
        # unsqueeze(0) -> (1, 256, 33, 3) where batch=1
        x = torch.from_numpy(feats).unsqueeze(0)

        # 3) Convert to model expected shape: (seq_len, batch, n_mels, channels)
        # (1, 256, 33, 3) -> permute -> (256, 1, 33, 3)
        x = x.permute(1, 0, 2, 3).to(device)

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
