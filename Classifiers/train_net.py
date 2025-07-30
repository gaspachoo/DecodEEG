import os, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from sklearn.metrics import confusion_matrix

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Classifiers.modules.utils import compute_raw_stats, normalize_raw
from Classifiers.modules.models import deepnet, eegnet


# -------- W&B -------------------------------------------------------------
PROJECT_NAME = "eeg2video-GLMNetv4"  # <‑‑ change if you need another project



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
    p.add_argument("--save_dir", default="./GLMNet/checkpoints_net/")
    p.add_argument(
        "--cluster",
        type=int,
        help="Cluster index to filter labels (only valid when --category label)",
    )
    p.add_argument("--model", choices=["deepnet", "eegnet"], default="deepnet")
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
    p.add_argument("--n_subj", type=int, default=15, help="Number of subjects to sample for train/val when not provided")
    p.add_argument("--seed", type=int, default=0, help="Random seed for subject sampling")
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
                f"Unknown category: {category}. Must be one of: color, color_binary, face_appearance, human_appearance, label_cluster, label, obj_number, optical_flow_score."
            )


# ------------------------------ main -------------------------------------
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    rng = np.random.default_rng(args.seed)
    all_subj = sorted(
        f[:-4] for f in os.listdir(args.raw_dir) if f.startswith("sub") and f.endswith(".npy")
    )

    if args.n_subj > len(all_subj):
        raise ValueError("Not enough subject files in raw_dir")

    rng.shuffle(all_subj)
    selected = all_subj[: args.n_subj]
    # first subjects are used for training and validation, the rest form the test set
    train_subj = selected[:13]
    val_subj = selected[13:15]
    test_subj = [s for s in all_subj if s not in train_subj and s not in val_subj]

    print("Training subjects:", train_subj)
    print("Validation subjects:", val_subj)
    print("Test subjects:", test_subj)

    name_ids = "_".join(s.replace("sub", "") for s in train_subj)

    ckpt_name = args.category
    if args.cluster is not None:
        ckpt_name += f"_cluster{args.cluster}"
    ckpt_dir = os.path.join(args.save_dir, f"sub_{name_ids}", ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, f"{args.model}_best.pt")
    stats_path = os.path.join(ckpt_dir, "raw_stats.npz")

    sample_raw = np.load(os.path.join(args.raw_dir, f"{train_subj[0]}.npy"))
    n_blocks, n_concepts, n_rep, n_win, C, T = sample_raw.shape
    sample_raw = sample_raw.reshape(n_blocks, n_concepts * n_rep, n_win, C, T)

    label_path = os.path.join(args.label_dir, f"All_video_{args.category}.npy")
    if args.category == "color_binary" and not os.path.exists(label_path):
        label_path = os.path.join(args.label_dir, "All_video_color.npy")
    labels_raw = np.load(label_path)
    if labels_raw.shape[1] == n_concepts:
        labels_raw = np.repeat(labels_raw[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)

    if args.category == "color":
        mask_2d = labels_raw != 0
    else:
        mask_2d = np.ones_like(labels_raw, dtype=bool)

    if args.cluster is not None:
        cluster_path = os.path.join(args.label_dir, "All_video_label_cluster.npy")
        clusters = np.load(cluster_path)
        if clusters.shape[1] == n_concepts:
            clusters = np.repeat(clusters[:, :, None], n_rep, axis=2).reshape(n_blocks, n_concepts * n_rep)
        mask_2d &= clusters == args.cluster

    mask_flat = mask_2d.reshape(-1)
    labels_flat = labels_raw.reshape(-1)[mask_flat] - (1 if args.category == "color" else 0)

    def expand_labels_flat(labels_1d: np.ndarray, n_win: int) -> np.ndarray:
        return np.repeat(labels_1d[:, None], n_win, axis=1)

    base_labels = format_labels(expand_labels_flat(labels_flat, n_win), args.category)

    if args.cluster is not None and args.category == "label":
        uniq = np.sort(np.unique(base_labels))
        mapping = {v: i for i, v in enumerate(uniq)}
        base_labels = np.vectorize(mapping.get)(base_labels)
        print(f"Cluster {args.cluster}: mapping original labels {uniq.tolist()} -> {list(mapping.values())}")

    unique_labels, counts_labels = np.unique(base_labels, return_counts=True)
    num_unique_labels = len(unique_labels)
    label_final_distribution = {int(u): int(c) for u, c in zip(unique_labels, counts_labels)}
    print("Label distribution after formating:", label_final_distribution)

    def load_subject(name: str) -> tuple[np.ndarray, np.ndarray]:
        subj_raw = np.load(os.path.join(args.raw_dir, f"{name}.npy"))
        subj_raw = subj_raw.reshape(n_blocks, n_concepts * n_rep, n_win, C, T)
        subj_raw = subj_raw.reshape(-1, n_win, C, T)[mask_flat]
        subj_labels = base_labels.copy()
        return subj_raw, subj_labels

    def concat_subjects(names: list[str]):
        X_list, y_list = [], []
        for n in names:
            print("Processing subject:", n)
            xr, yl = load_subject(n)
            X_list.append(xr)
            y_list.append(yl)
        if not X_list:
            return np.empty((0, n_win, C, T)), np.empty((0, n_win), dtype=np.int64)
        return np.concatenate(X_list), np.concatenate(y_list)

    print("Concatenating subjects...")
    X_train, y_train = concat_subjects(train_subj)
    X_val, y_val = concat_subjects(val_subj)
    X_test, y_test = concat_subjects(test_subj)

    X_train = X_train.reshape(-1, C, T)
    y_train = y_train.reshape(-1)
    X_val = X_val.reshape(-1, C, T)
    y_val = y_val.reshape(-1)
    X_test = X_test.reshape(-1, C, T)
    y_test = y_test.reshape(-1)

    raw_mean, raw_std = compute_raw_stats(X_train)
    X_train = normalize_raw(X_train, raw_mean, raw_std)
    X_val = normalize_raw(X_val, raw_mean, raw_std)
    X_test = normalize_raw(X_test, raw_mean, raw_std)

    np.savez(stats_path, mean=raw_mean, std=raw_std)

    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train),
    )
    ds_val = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val),
    )
    ds_test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test),
    )

    dl_train = DataLoader(ds_train, args.bs, shuffle=True)
    dl_val = DataLoader(ds_val, args.bs)
    dl_test = DataLoader(ds_test, args.bs)

    model_cls = deepnet if args.model == "deepnet" else eegnet
    model = model_cls(out_dim=num_unique_labels, C=C, T=T).to(device)
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
        wandb.init(project=PROJECT_NAME, name=f"sub_{name_ids}_{ckpt_name}_{args.model}", config=vars(args))
        wandb.watch(model, log="all")

    best_val = 0.0
    for ep in tqdm(range(1, args.epochs + 1)):
        model.train()
        tl = ta = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            ta += (pred.argmax(1) == yb).sum().item()
        train_acc = ta / len(ds_train)

        model.eval()
        vl = va = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
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
            torch.save(model.state_dict(), model_path)
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

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_acc = 0
    preds, labels_test = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
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
