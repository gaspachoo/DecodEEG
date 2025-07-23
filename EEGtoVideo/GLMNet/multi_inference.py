# -*- coding: utf-8 -*-
"""Run multiple GLMNet models on a single EEG example.

This script loads several GLMNet checkpoints and converts their
predictions into text using ``label_mappings.json``.  For each model we
evaluate all seven windows of the EEG sample and keep the label that
appears most often.  The textual outputs are concatenated to form a
descriptive phrase and a confidence score is reported for each label.
"""

import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

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


def majority_vote(
    eeg: np.ndarray,
    model: GLMNet,
    scaler,
    stats,
    device: str,
) -> Tuple[int, float]:
    """Return the most common prediction index and its confidence."""
    preds = []
    for win in eeg:
        x_raw, x_feat = prepare_input(win, stats, scaler)
        x_raw = x_raw.to(device)
        x_feat = x_feat.to(device)
        with torch.no_grad():
            logits = model(x_raw, x_feat)
            preds.append(int(logits.argmax(dim=-1).item()))
    values, counts = np.unique(preds, return_counts=True)
    best_idx = counts.argmax()
    majority = int(values[best_idx])
    conf = counts[best_idx] / len(preds)
    return majority, conf


def main() -> None:
    p = argparse.ArgumentParser(description="Run multiple GLMNet models on EEG windows")
    p.add_argument("--eeg", required=True, help="Path to EEG numpy file (concept, repetition, window, C, T)")
    p.add_argument("--block", type=int, default=0, help="Block index to load")
    p.add_argument("--concept", type=int, default=0, help="Concept index to load")
    p.add_argument("--repetition", type=int, default=0, help="Repetition index to load")
    p.add_argument(
        "--checkpoint_root",
        required=True,
        help="Directory containing all GLMNet checkpoints"
    )
    p.add_argument(
        "--mapping_path",
        default=os.path.join(os.path.dirname(__file__), "label_mappings.json"),
        help="Path to label_mappings.json"
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    eeg_all = np.load(args.eeg)
    print(f"Loaded EEG data with shape {eeg_all.shape}")
    eeg = eeg_all[args.block, args.concept, args.repetition]
    channels, time_len = eeg.shape[-2], eeg.shape[-1]

    ckpt_dirs = {
        get_category(d): os.path.join(args.checkpoint_root, d)
        for d in os.listdir(args.checkpoint_root)
        if os.path.isdir(os.path.join(args.checkpoint_root, d))
    }

    required = ["label_cluster"] + [f"label_cluster{i}" for i in range(9)]
    for cat in required:
        if cat not in ckpt_dirs:
            raise FileNotFoundError(
                f"Required checkpoint '{cat}' not found in {args.checkpoint_root}"
            )

    optional = [
        "color_binary",
        "color",
        "face_appearance",
        "human_appearance",
        "obj_number",
        "optical_flow_score",
    ]
    for cat in optional:
        if cat not in ckpt_dirs and cat != "color":
            warnings.warn(f"Checkpoint for {cat} not found - skipping")

    label_map = load_label_mappings(args.mapping_path)

    def load(cat: str):
        path = ckpt_dirs[cat]
        return load_model(path, channels, time_len, args.device)

    models = {}
    scalers = {}
    stats = {}
    for cat in required + [c for c in optional if c in ckpt_dirs]:
        mdl, sc, st = load(cat)
        models[cat] = mdl
        scalers[cat] = sc
        stats[cat] = st

    phrase_parts = []
    confidences = []

    if "color_binary" in models:
        idx, conf = majority_vote(eeg, models["color_binary"], scalers["color_binary"], stats["color_binary"], args.device)
        confidences.append(conf)
        if idx == 1:
            if "color" not in models:
                raise FileNotFoundError("Color checkpoint required when dominant color is predicted")
            idx_col, conf_col = majority_vote(
                eeg,
                models["color"],
                scalers["color"],
                stats["color"],
                args.device,
            )
            phrase_parts.append(index_to_text("color", idx_col, label_map))
            confidences.append(conf_col)
        else:
            phrase_parts.append(index_to_text("color_binary", idx, label_map))

    if "face_appearance" in models:
        idx, conf = majority_vote(eeg, models["face_appearance"], scalers["face_appearance"], stats["face_appearance"], args.device)
        phrase_parts.append(index_to_text("face_appearance", idx, label_map))
        confidences.append(conf)

    if "human_appearance" in models:
        idx, conf = majority_vote(eeg, models["human_appearance"], scalers["human_appearance"], stats["human_appearance"], args.device)
        phrase_parts.append(index_to_text("human_appearance", idx, label_map))
        confidences.append(conf)

    idx_cluster, conf_cluster = majority_vote(
        eeg,
        models["label_cluster"],
        scalers["label_cluster"],
        stats["label_cluster"],
        args.device,
    )
    phrase_parts.append(index_to_text("label_cluster", idx_cluster, label_map))
    confidences.append(conf_cluster)

    label_cat = f"label_cluster{idx_cluster}"
    idx_label, conf_label = majority_vote(
        eeg,
        models[label_cat],
        scalers[label_cat],
        stats[label_cat],
        args.device,
    )
    phrase_parts.append(index_to_text("label", idx_label, label_map))
    confidences.append(conf_label)

    if "obj_number" in models:
        idx, conf = majority_vote(eeg, models["obj_number"], scalers["obj_number"], stats["obj_number"], args.device)
        phrase_parts.append(index_to_text("obj_number", idx, label_map))
        confidences.append(conf)

    if "optical_flow_score" in models:
        idx, conf = majority_vote(eeg, models["optical_flow_score"], scalers["optical_flow_score"], stats["optical_flow_score"], args.device)
        phrase_parts.append(index_to_text("optical_flow_score", idx, label_map))
        confidences.append(conf)

    phrase = " ".join(phrase_parts)
    print(phrase)
    print("Confidences:", [f"{c:.2f}" for c in confidences])


if __name__ == "__main__":
    main()
