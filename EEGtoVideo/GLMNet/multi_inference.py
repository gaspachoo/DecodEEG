# -*- coding: utf-8 -*-
"""Run multiple GLMNet models on a single EEG example.

This script loads several GLMNet checkpoints together with their
associated class mappings and predicts text labels for a provided EEG
window. The five textual predictions are concatenated to form a phrase.
"""

import argparse
import json
import os
from typing import List, Dict

import numpy as np
import torch

from EEGtoVideo.GLMNet.modules.utils_glmnet import (
    GLMNet,
    load_scaler,
    load_raw_stats,
    normalize_raw,
    standard_scale_features,
)
from EEGtoVideo.GLMNet.modules.models_paper import mlpnet


OCCIPITAL_IDX = list(range(50, 62))


def load_mapping(path: str) -> Dict[int, str]:
    """Load a class mapping from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): str(v) for k, v in data.items()}


def prepare_input(eeg: np.ndarray, stats, scaler) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize raw EEG and compute scaled features."""
    raw = eeg[np.newaxis, ...].astype(np.float32)
    feat = mlpnet.compute_features(raw)
    raw = normalize_raw(raw, stats[0], stats[1])
    feat = standard_scale_features(feat, scaler=scaler)
    x_raw = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
    x_feat = torch.tensor(feat, dtype=torch.float32)
    return x_raw, x_feat


def load_model(ckpt_dir: str, channels: int, time_len: int, device: str) -> tuple[GLMNet, any, tuple[np.ndarray, np.ndarray]]:
    """Load GLMNet model with its scaler and raw statistics."""
    scaler = load_scaler(os.path.join(ckpt_dir, "scaler.pkl"))
    stats = load_raw_stats(os.path.join(ckpt_dir, "raw_stats.npz"))
    model_path = os.path.join(ckpt_dir, "glmnet_best.pt")
    model = GLMNet.load_from_checkpoint(
        model_path, OCCIPITAL_IDX, C=channels, T=time_len, device=device
    )
    return model, scaler, stats


def predict_text(eeg: np.ndarray, models: List[GLMNet], scalers, stats_list, mappings, device: str) -> str:
    """Generate a phrase by applying each model to the EEG sample."""
    parts = []
    for mdl, scaler, stats, mapping in zip(models, scalers, stats_list, mappings):
        x_raw, x_feat = prepare_input(eeg, stats, scaler)
        x_raw = x_raw.to(device)
        x_feat = x_feat.to(device)
        with torch.no_grad():
            logits = mdl(x_raw, x_feat)
            pred = int(logits.argmax(dim=-1).item())
        parts.append(mapping.get(pred, str(pred)))
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Run multiple GLMNet models on one EEG window")
    p.add_argument("--eeg", required=True, help="Path to EEG numpy file (C,T)")
    p.add_argument("--checkpoint_dirs", nargs=5, required=True, help="Five GLMNet checkpoint directories")
    p.add_argument("--mapping_files", nargs=5, required=True, help="JSON files mapping class indices to text")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    eeg = np.load(args.eeg)
    channels, time_len = eeg.shape

    models = []
    scalers = []
    stats_list = []
    for ckpt in args.checkpoint_dirs:
        model, scaler, stats = load_model(ckpt, channels, time_len, args.device)
        models.append(model)
        scalers.append(scaler)
        stats_list.append(stats)

    mappings = [load_mapping(p) for p in args.mapping_files]

    phrase = predict_text(eeg, models, scalers, stats_list, mappings, args.device)
    print(phrase)


if __name__ == "__main__":
    main()
