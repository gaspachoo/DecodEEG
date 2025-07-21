import csv, argparse, pathlib, re, math, os, sys

project_root = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", default="data/Video_mp4")
    ap.add_argument("--eeg_root", default="data/gaspardnew/eeg_firstw/sub3")
    ap.add_argument("--out_csv",   default="eeg_video_dataset.csv")
    ap.add_argument("--val_block", type=int, default=6,
                    help="block id reserved for validation")
    return ap.parse_args()

def main():
    args = parse_args()
    video_root = pathlib.Path(args.video_root)
    eeg_root   = pathlib.Path(args.eeg_root)

    rows = []
    for video_path in video_root.glob("Block*/*.mp4"):
        # BlockX/1.mp4  →  X,   1
        m = re.match(r"Block(\d+)/(\d+)\.mp4", video_path.relative_to(video_root).as_posix())
        if not m:
            continue
        block, clip_id = map(int, m.groups())
        eeg_path = eeg_root / f"Block{block}/{clip_id}.npy"
        if not eeg_path.exists():
            print(f"[WARN] EEG manquant pour {video_path}")
            continue
        split = "val" if block == args.val_block else "train"
        rows.append({
            "split": split,
            "video_path": video_path.as_posix(),
            "eeg_path":  eeg_path.as_posix(),
            "block": block,
            "clip_id": clip_id
        })

    rows.sort(key=lambda r: (r["block"], r["clip_id"]))
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✔︎  {len(rows)} lignes écrites dans {args.out_csv}")

if __name__ == "__main__":
    main()
