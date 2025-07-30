import os, sys
import torch
import numpy as np
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    

from Classifiers.modules.utils import (
    standard_scale_features,
    normalize_raw,
    load_scaler,
    load_raw_stats,
)
from Classifiers.modules.models import mlpnet, glmnet


OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels


def inf_glmnet(model, scaler, raw_sw, stats, device="cuda"):

    # always compute spectral features from the raw windows
    raw_flat = raw_sw.reshape(-1, raw_sw.shape[-2], raw_sw.shape[-1])
    feat_sw = mlpnet.compute_features(raw_flat)
    # reshape back to (runs, videos, trials, windows, channels, features)
    feat_sw = feat_sw.reshape(raw_sw.shape[:-2] + feat_sw.shape[-2:])

    # flatten for batch inference
    raw_flat = raw_sw.reshape(-1, raw_sw.shape[-2], raw_sw.shape[-1])
    feat_flat = feat_sw.reshape(-1, feat_sw.shape[-2], feat_sw.shape[-1])

    raw_flat = normalize_raw(raw_flat, stats[0], stats[1])

    # scale features
    feat_scaled = standard_scale_features(feat_flat, scaler=scaler)

    embeddings = []
    with torch.no_grad():
        for raw_seg, feat_seg in zip(raw_flat, feat_scaled):
            x = np.concatenate([raw_seg, feat_seg], axis=-1)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

            z = model(x, return_features=True)
            embeddings.append(z.squeeze(0).cpu().numpy())

    return np.stack(embeddings)  # shape: (N_segments, emb_dim // 2)

# --- Main generation loop ---
def generate_all_embeddings(
    raw_dir,
    ckpt_path,
    output_dir,
    subject_prefix,
    device="cuda",
):
    """Run GLMNet inference for all subjects matching the prefix."""

    os.makedirs(output_dir, exist_ok=True)

    scaler_path = os.path.join(ckpt_path, "scaler.pkl")
    stats_path = os.path.join(ckpt_path, "raw_stats.npz")
    model_path = os.path.join(ckpt_path, "glmnet_best.pt")
    
    scaler = load_scaler(scaler_path)
    stats = load_raw_stats(stats_path)

    for fname in os.listdir(raw_dir):
        if not (fname.endswith('.npy') and fname.startswith(subject_prefix)):
            continue
        print(f"Processing {fname}...")
        subj = os.path.splitext(fname)[0]

        # load pre-segmented windows
        RAW_SW = np.load(os.path.join(raw_dir, fname))
        # expect shape: (7, 40, 5, 7, 62, T)
        time_len = RAW_SW.shape[-1]
        num_channels = RAW_SW.shape[-2]
        state = torch.load(model_path, map_location=device)
        out_dim = glmnet.infer_out_dim(state)
        model = glmnet(OCCIPITAL_IDX, C=num_channels, T=time_len, out_dim=out_dim)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        embeddings = inf_glmnet(model, scaler, RAW_SW, stats, device)
        
        out_path = os.path.join(output_dir, f"{subj}.npy")
        np.save(out_path, embeddings)
        print(f"Saved embeddings for {subj}, shape {embeddings.shape}")

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dir', default="./data/Preprocessing/Segmented_500ms_sw", help='directory of pre-windowed raw EEG .npy files')
    parser.add_argument('--subject_prefix', default='sub3', help='prefix of subject files to process')
    parser.add_argument('--checkpoint_path', help='path to GLMNet checkpoint')
    parser.add_argument('--train_mode', choices=['ordered', 'shuffle'], default='ordered', help='training mode for mono model')
    parser.add_argument('--seed', type=int, default=0, help='Training seed')
    parser.add_argument('--output_dir', default="./data/eeg_segments", help='where to save projected embeddings')
    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(
            "./Classifiers/checkpoints",
            "mono",
            args.subject_prefix,
            args.train_mode,
            str(args.seed),
            "glmnet",
            "label_cluster",
        )

    generate_all_embeddings(
        args.raw_dir,
        args.checkpoint_path,
        args.output_dir,
        args.subject_prefix,
    )
