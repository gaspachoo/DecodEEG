# -*- coding: utf-8 -*-
"""Run multiple GLMNet models on a single EEG example.

This script loads five GLMNet checkpoints and converts their
predictions into text using ``label_mappings.json``.  The textual
outputs are concatenated to form a descriptive phrase.
"""

import argparse
import json
import os
from typing import Dict, List

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

def load_label_mappings(path: str) -> Dict[str, Dict[int, str]]:
    """Load textual descriptions for every label category."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mappings: Dict[str, Dict[int, str]] = {}
    for cat, mapping in data.items():
        mappings[cat] = {int(k): str(v) for k, v in mapping.items()}
    return mappings


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


def get_category(path: str) -> str:
    """Infer the label category from a checkpoint directory."""
    name = os.path.basename(os.path.normpath(path))
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer category from '{name}'")
    return "_".join(parts[1:])


def index_to_text(category: str, idx: int, label_map: Dict[str, Dict[int, str]]) -> str:
    """Convert predicted class index to textual description."""
    if category in {"label", "obj_number"}:
        idx += 1
    return label_map.get(category, {}).get(idx, str(idx))


def predict_text(
    eeg: np.ndarray,
    models: List[GLMNet],
    scalers,
    stats_list,
    categories: List[str],
    label_map: Dict[str, Dict[int, str]],
    device: str,
) -> str:
    """Generate a phrase by applying each model to the EEG sample."""
    parts = []
    for mdl, scaler, stats, cat in zip(models, scalers, stats_list, categories):
        x_raw, x_feat = prepare_input(eeg, stats, scaler)
        x_raw = x_raw.to(device)
        x_feat = x_feat.to(device)
        with torch.no_grad():
            logits = mdl(x_raw, x_feat)
            pred = int(logits.argmax(dim=-1).item())
        parts.append(index_to_text(cat, pred, label_map))
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="Run multiple GLMNet models on one EEG window")
    p.add_argument("--eeg", required=True, help="Path to EEG numpy file (C,T)")
    p.add_argument(
        "--checkpoint_dirs", nargs=5, required=True,
        help="Five GLMNet checkpoint directories"
    )
    p.add_argument(
        "--mapping_path",
        default=os.path.join(os.path.dirname(__file__), "label_mappings.json"),
        help="Path to label_mappings.json"
    )
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

    label_map = load_label_mappings(args.mapping_path)
    categories = [get_category(c) for c in args.checkpoint_dirs]

    phrase = predict_text(
        eeg,
        models,
        scalers,
        stats_list,
        categories,
        label_map,
        args.device,
    )
    print(phrase)


if __name__ == "__main__":
    main()
