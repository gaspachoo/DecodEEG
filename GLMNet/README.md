# GLMNet Utilities

This directory contains scripts used to train and run the GLMNet EEG encoder.

## Label mappings

`label_mappings.json` stores textual descriptions for the integer labels used
by various training categories. The file is a dictionary where each key is a
category name (e.g. `color` or `label_cluster`) and each value is another
dictionary mapping label IDs to their string description.

Example structure:

```json
{
  "color": {"0": "black and white", "1": "color"},
  "label": {"1": "example label 1", "2": "example label 2"}
}
```

`multi_inference.py` loads this JSON and automatically maps the
predicted class indices from each checkpoint to their textual
descriptions.  The script infers the label category from the checkpoint
directory name (everything after the first underscore).

Example usage:

```bash
python multi_inference.py \
  --eeg example.npy \
  --concept 0 --repetition 0 \
  --checkpoint_root ./EEGtoVideo/checkpoints/glmnet/sub3
```

The script now evaluates all seven windows corresponding to the selected
`concept` and `repetition`.  For each model the label occurring most
often across the windows is kept and a confidence score is reported.

## Binary color category

`color_binary` is a simplified version of the original `color` labels. It maps
any label other than `0` to the value `1`, indicating that one color dominates
the image. Label `0` still represents videos with many colors. Use
`--category color_binary` when training to enable this behaviour. If a file
named `All_video_color_binary.npy` is not present in `--label_dir`, the training
script falls back to `All_video_color.npy` and performs the conversion at
runtime.

When training with `--category color`, samples tagged as `0` are discarded and
the remaining color IDs (1-6) are shifted down to 0-5.  `label_mappings.json`
lists the updated names for these six classes.

## Cluster-specific label training

For the `label` category you can focus on a single label cluster by passing
`--cluster <idx>` to `train_glmnet.py`. The script loads
`All_video_label_cluster.npy` from `--label_dir` and filters the dataset to the
selected cluster before training.  Within that subset, the original label IDs
are remapped to a contiguous range starting at zero.

