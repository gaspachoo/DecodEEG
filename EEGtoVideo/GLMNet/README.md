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
  --checkpoint_dirs ckpt_color ckpt_face ckpt_human ckpt_label ckpt_obj_number
```
