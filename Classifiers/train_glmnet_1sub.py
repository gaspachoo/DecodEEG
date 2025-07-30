import os, time, argparse, sys, re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import pickle
from sklearn.metrics import confusion_matrix

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from Classifiers.modules.utils import (
    standard_scale_features,
    compute_raw_stats,
    normalize_raw,
)
from Classifiers.modules.models import mlpnet, glmnet


# -------- W&B -------------------------------------------------------------
PROJECT_NAME = "eeg2video-classifiersv4.1-1sub"  # <‑‑ change if you need another project

# ------------------------------ constants ---------------------------------
OCCIPITAL_IDX = list(range(50, 62))  # 12 occipital channels


# ------------------------------ utils -------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="./data/Preprocessing/Segmented_1000ms_sw", help="directory with .npy files")
    p.add_argument("--label_dir", default="./data/meta_info", help="Label file")
    p.add_argument(
        "--category",
        default="label_cluster",
        choices=[
            "color",
            "color_binary",
            "face_appearance",
            "human_appearance",
            "label_cluster",
            "label",
            "obj_number",
            "optical_flow_score",
        ],
        help="Label file",
    )
    p.add_argument("--save_dir", default="./Checkpoints")
    p.add_argument(
        "--cluster",
        type=int,
        help="Cluster index to filter labels (only valid when --category label)",
    )
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for the scheduler")
    p.add_argument(
        "--scheduler",
        type=str,
        choices=["steplr", "reducelronplateau", "cosine"],
        default="reducelronplateau",
        help="Type of learning rate scheduler",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--subj_name", default="sub3", help="Subject name to process")
    p.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed controlling the block split",
    )
    return p.parse_args()

def format_labels(labels: np.ndarray, category: str) -> np.ndarray:
    match category:
        case "color":
            return labels.astype(np.int64)
        case "face_appearance" | "human_appearance" | "label_cluster":
            return labels.astype(np.int64)
        case "color_binary":
            # Collapse all non-zero colors into the dominant color class
            return (labels != 0).astype(np.int64)
        case "label" | "obj_number":
            labels = labels - 1
            return labels.astype(np.int64)
        case "optical_flow_score":
            threshold = 1.799
            return (labels > threshold).astype(np.int64)
        case _:
            raise ValueError(
                f"Unknown category: {category}. Must be one of: color, color_binary, face_appearance, human_appearance, object, label_cluster, label, obj_number, optical_flow_score."
            )


# ------------------------------ main -------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define saving paths
    ckpt_name = args.category
    if args.cluster is not None:
        ckpt_name += f"_cluster{args.cluster}"
    ckpt_dir = os.path.join(
        args.save_dir,
        "mono",
        args.subj_name,
        "ordered",
        str(args.split_seed),
        "glmnet",
        ckpt_name,
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    shallownet_path = os.path.join(ckpt_dir, "shallownet.pt")
    mlpnet_path = os.path.join(ckpt_dir, "mlpnet.pt")
    stats_path = os.path.join(ckpt_dir, "raw_stats.npz")
    scaler_path = os.path.join(ckpt_dir, "scaler.pkl")
    glmnet_path = os.path.join(ckpt_dir, "glmnet_best.pt")

    raw = np.load(os.path.join(args.raw_dir, f"{args.subj_name}.npy"))
    # compute DE features from raw EEG windows
    duration_ms = int(re.search(r'_(\d+)ms_', os.path.basename(args.raw_dir)).group(1)) / 1000
    feat = mlpnet.compute_features(raw.reshape(-1, raw.shape[-2], raw.shape[-1]), win_sec=duration_ms).reshape(
        *raw.shape[:4], raw.shape[-2], -1
    )

    n_blocks, n_concepts, n_rep, n_win, C, T = raw.shape  # 7, 40, 5, ...

    raw = raw.reshape(n_blocks, n_concepts * n_rep, n_win, C, T)  # (7, 200, …)
    feat = feat.reshape(n_blocks, n_concepts * n_rep, n_win, C, -1)  # (7, 200, …)

    label_path = os.path.join(args.label_dir, f"All_video_{args.category}.npy")
    if args.category == "color_binary" and not os.path.exists(label_path):
        label_path = os.path.join(args.label_dir, "All_video_color.npy")

    labels_raw = np.load(label_path)  # (7, 40) ou (7, 200)
    if labels_raw.shape[1] == n_concepts:  # encore au niveau concept
        labels_raw = np.repeat(labels_raw[:, :, None], n_rep, axis=2).reshape(
            n_blocks, n_concepts * n_rep
        )  # → (7, 200)

    if args.category == "color":
        mask_2d = labels_raw != 0  # (7, 200) bool
    else:
        mask_2d = np.ones_like(labels_raw, dtype=bool)

    if args.cluster is not None:
        cluster_path = os.path.join(args.label_dir, "All_video_label_cluster.npy")
        clusters = np.load(cluster_path)
        if clusters.shape[1] == n_concepts:
            clusters = np.repeat(clusters[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)
        mask_2d &= clusters == args.cluster

    block_ids = np.repeat(np.arange(n_blocks), n_concepts * n_rep)
    mask_flat = mask_2d.reshape(-1)

    raw = raw.reshape(-1, n_win, C, T)
    feat = feat.reshape(-1, n_win, C, feat.shape[-1])
    labels_flat = labels_raw.reshape(-1)

    block_ids = block_ids[mask_flat]
    raw = raw[mask_flat]
    feat = feat[mask_flat]
    labels_flat = labels_flat[mask_flat] - (
        1 if args.category == "color" else 0
    )

    def expand_labels_flat(labels_1d: np.ndarray, n_win: int) -> np.ndarray:
        """(N_vid,) → (N_vid, n_win)"""
        return np.repeat(labels_1d[:, None], n_win, axis=1)

    labels = format_labels(expand_labels_flat(labels_flat, n_win), args.category)
    # raw, feat : (N_vid, n_win, C, …)   labels : (N_vid, n_win)

    if args.cluster is not None and args.category == "label":
        uniq = np.sort(np.unique(labels))
        mapping = {v: i for i, v in enumerate(uniq)}
        labels = np.vectorize(mapping.get)(labels)
        print(f"Cluster {args.cluster}: mapping original labels {uniq.tolist()} -> {list(mapping.values())}")

    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    num_unique_labels = len(unique_labels)
    label_final_distribution = {int(u): int(c) for u, c in zip(unique_labels, counts_labels)}
    print("Label distribution after formating:", label_final_distribution)

    num_channels = raw.shape[-2]  # C
    time_len = raw.shape[-1]  # T
    feat_dim = feat.shape[-1]

    X_all = raw.reshape(-1, num_channels, time_len)
    F_all = feat.reshape(-1, num_channels, feat_dim)
    y_all = labels.reshape(-1)

    block_ids_win = np.repeat(block_ids, n_win)

    rng = np.random.RandomState(args.split_seed)
    val_block, test_block = rng.choice(np.arange(n_blocks), size=2, replace=False)
    print(f"Validation block: {val_block}, Test block: {test_block}")

    train_mask = (block_ids_win != val_block) & (block_ids_win != test_block)
    val_mask = block_ids_win == val_block
    test_mask = block_ids_win == test_block

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train, F_train, y_train = X_all[train_idx], F_all[train_idx], y_all[train_idx]
    X_val, F_val, y_val = X_all[val_idx], F_all[val_idx], y_all[val_idx]
    X_test, F_test, y_test = X_all[test_idx], F_all[test_idx], y_all[test_idx]

    # Normalization parameters from training data
    raw_mean, raw_std = compute_raw_stats(X_train)
    X_train = normalize_raw(X_train, raw_mean, raw_std)
    X_val = normalize_raw(X_val, raw_mean, raw_std)
    X_test = normalize_raw(X_test, raw_mean, raw_std)

    # Feature scaling
    F_train_scaled, scaler = standard_scale_features(F_train, return_scaler=True)
    F_val_scaled = standard_scale_features(F_val, scaler=scaler)
    F_test_scaled = standard_scale_features(F_test, scaler=scaler)

    # Save preprocessing objects
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    np.savez(stats_path, mean=raw_mean, std=raw_std)

    # DataLoaders
    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(F_train_scaled, dtype=torch.float32),
        torch.tensor(y_train),
    )
    ds_val = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(F_val_scaled, dtype=torch.float32),
        torch.tensor(y_val),
    )
    ds_test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(F_test_scaled, dtype=torch.float32),
        torch.tensor(y_test),
    )

    dl_train = DataLoader(ds_train, args.bs, shuffle=True)
    dl_val = DataLoader(ds_val, args.bs)
    dl_test = DataLoader(ds_test, args.bs)

    model = glmnet(OCCIPITAL_IDX, C=num_channels, T=time_len, out_dim=num_unique_labels).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == "reducelronplateau":
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.8, patience=10, verbose=False, min_lr=args.min_lr)
    elif args.scheduler == "steplr":
        scheduler = StepLR(opt, step_size=10, gamma=0.5)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=args.epochs // 2, eta_min=args.min_lr)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        run_name = f"{args.subj_name}_{ckpt_name}_val{val_block}_test{test_block}"
        wandb.init(project=PROJECT_NAME, name=run_name, config=vars(args))
        wandb.watch(model, log="all")

    best_val = 0.0
    for ep in tqdm(range(1, args.epochs + 1)):
        model.train()
        tl = ta = 0
        for xb, xf, yb in dl_train:
            xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, xf)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            ta += (pred.argmax(1) == yb).sum().item()
        train_acc = ta / len(ds_train)

        model.eval()
        vl = va = 0
        with torch.no_grad():
            for xb, xf, yb in dl_val:
                xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
                pred = model(xb, xf)
                vloss = criterion(pred, yb)
                vl += vloss.item() * len(yb)
                va += (pred.argmax(1) == yb).sum().item()
        val_acc = va / len(ds_val)
        val_loss = vl / len(ds_val)
        if scheduler is not None:
            old_lr = opt.param_groups[0]["lr"]
            if args.scheduler == "reducelronplateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
            for pg in opt.param_groups:
                if pg["lr"] < args.min_lr:
                    pg["lr"] = args.min_lr
            new_lr = opt.param_groups[0]["lr"]
            if new_lr < old_lr:
                tqdm.write(f"Epoch {ep:05d}: reducing learning rate of group 0 to {new_lr:.4e}.")
        current_lr = opt.param_groups[0]["lr"]

        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), glmnet_path)
            torch.save(model.raw_global.state_dict(), shallownet_path)
            torch.save(model.freq_local.state_dict(), mlpnet_path)
            tqdm.write(f"New best model saved at epoch {ep} with val_acc={val_acc:.3f}")

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": ep,
                    "train/acc": train_acc,
                    "val/acc": val_acc,
                    "train/loss": tl / len(ds_train),
                    "val/loss": val_loss,
                    "lr": current_lr,
                }
            )

    state = torch.load(glmnet_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_acc = 0
    preds, labels_test = [], []
    with torch.no_grad():
        for xb, xf, yb in dl_test:
            xb, xf, yb = xb.to(device), xf.to(device), yb.to(device)
            out = model(xb, xf)
            pred_labels = out.argmax(1)
            test_acc += (pred_labels == yb).sum().item()
            preds.append(pred_labels.cpu())
            labels_test.append(yb.cpu())
    preds = torch.cat(preds).numpy()
    labels_test = torch.cat(labels_test).numpy()
    cm = confusion_matrix(labels_test, preds)
    cm_percent = confusion_matrix(labels_test, preds, normalize="true") * 100
    test_acc /= len(ds_test)
    print(f"Test accuracy = {test_acc:.3f}")
    print("Confusion matrix:\n", cm)
    print("Weighted confusion matrix (%)\n", cm_percent)
    if args.use_wandb:
        class_names = [str(c) for c in np.unique(labels_test)]
        cm_plot = wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels_test,
            preds=preds,
            class_names=class_names,
        )
        table_columns = ["true"] + class_names
        cm_table = wandb.Table(columns=table_columns)
        for idx, row in enumerate(cm_percent):
            cm_table.add_data(class_names[idx], *row.tolist())
        wandb.log(
            {
                "test/acc": test_acc,
                "test/confusion_matrix": cm_plot,
                "test/confusion_matrix_percent": cm_table,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
