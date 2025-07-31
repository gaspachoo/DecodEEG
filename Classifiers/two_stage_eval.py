import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

from Classifiers.modules.utils import block_split
from Classifiers.multi_inference import load_model, majority_vote, CLUSTER_RANGES


def two_stage_predict(eeg, models, scalers, stats, device, model_type):
    """Return predicted global label using cluster and cluster label models."""
    idx_cluster, _ = majority_vote(
        eeg,
        models["label_cluster"],
        scalers["label_cluster"],
        stats["label_cluster"],
        device,
        model_type,
    )
    cat = f"label_cluster{idx_cluster}"
    idx_label, _ = majority_vote(
        eeg,
        models[cat],
        scalers[cat],
        stats[cat],
        device,
        model_type,
    )
    start, _ = CLUSTER_RANGES[idx_cluster]
    return start + idx_label - 1


def evaluate_subject(eeg_path, label_dir, ckpt_root, seed, model_type="glmnet", device="cuda"):
    """Evaluate label classification for one subject."""
    raw = np.load(eeg_path)
    n_blocks, n_concepts, n_rep, n_win, C, T = raw.shape

    ckpt_seed_dir = os.path.dirname(ckpt_root)
    _, test_block = block_split(seed, n_blocks, ckpt_seed_dir)

    labels_all = np.load(os.path.join(label_dir, "All_video_label.npy"))
    if labels_all.shape[1] == n_concepts:
        labels_all = np.repeat(labels_all[:, :, None], n_rep, axis=2)
    labels_all = labels_all.reshape(n_blocks, n_concepts * n_rep)
    labels_test = labels_all[test_block] - 1

    categories = ["label", "label_cluster"] + [f"label_cluster{i}" for i in range(len(CLUSTER_RANGES))]
    models, scalers, stats = {}, {}, {}
    for cat in categories:
        ckpt_dir = os.path.join(ckpt_root, cat)
        model, scaler, st = load_model(ckpt_dir, C, T, device, model_type)
        models[cat] = model
        scalers[cat] = scaler
        stats[cat] = st

    preds_label = []
    preds_two = []
    labels_true = []

    for c in range(n_concepts):
        for r in range(n_rep):
            eeg = raw[test_block, c, r]
            lbl = labels_test[c * n_rep + r]
            labels_true.append(lbl)

            idx_one, _ = majority_vote(eeg, models["label"], scalers["label"], stats["label"], device, model_type)
            preds_label.append(idx_one)

            preds_two.append(two_stage_predict(eeg, models, scalers, stats, device, model_type))

    labels_true = np.array(labels_true)
    preds_label = np.array(preds_label)
    preds_two = np.array(preds_two)

    acc_one = (preds_label == labels_true).mean()
    acc_two = (preds_two == labels_true).mean()

    cm_one = confusion_matrix(labels_true, preds_label, labels=list(range(40)))
    cm_two = confusion_matrix(labels_true, preds_two, labels=list(range(40)))

    return acc_one, cm_one, acc_two, cm_two


def main():
    p = argparse.ArgumentParser(description="Evaluate label accuracy with single and two-stage models")
    p.add_argument("--eeg_path", required=True, help="Path to subject EEG numpy file")
    p.add_argument("--label_dir", required=True, help="Directory with label numpy files")
    p.add_argument("--checkpoint_root", required=True, help="Root directory of checkpoints (seed/model)")
    p.add_argument("--seed", type=int, default=0, help="Training seed")
    p.add_argument("--model", choices=["glmnet", "eegnet", "deepnet"], default="glmnet")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    acc_one, cm_one, acc_two, cm_two = evaluate_subject(
        args.eeg_path, args.label_dir, args.checkpoint_root, args.seed, args.model, args.device
    )

    print(f"One-step label accuracy: {acc_one:.3f}")
    print("Confusion matrix (label):\n", cm_one)
    print(f"Two-step label accuracy: {acc_two:.3f}")
    print("Confusion matrix (two-stage):\n", cm_two)


if __name__ == "__main__":
    main()
