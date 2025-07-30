# EEG2Video Overview

This repository collects utilities to classify EEG recordings and
convert those predictions into natural language prompts for text-to-video models.
Models such as **EEGNet**, **DeepNet** and **GLMNet** can be trained or loaded from
checkpoint directories.

## Data preparation

Raw recordings are first divided into 2‑second segments using
`segment_raw_signals_200Hz.py`.  Each segment has shape `(block, concept,
repetition, channels, time)` and is sampled at 200 Hz.  To obtain more training
samples we apply a sliding‑window strategy with
`segment_sliding_window.py`: every 2‑second segment is sliced into seven
windows of 500 ms with an overlap of 250 ms.
The resulting tensor has shape `(block, concept, repetition, window, channels,
time)`.

## Training

Any of the supported encoders can be trained on these windows.  Models predict
class labels for categories such as *color*, *label cluster* or *object number*.
Checkpoints are stored under `Checkpoints/<mode>/<seed>/<model>/<category>`.
For single-subject runs the hierarchy becomes
`Checkpoints/mono/<subject>/<ordered|shuffle>/<seed>/<model>/<category>`.

## Classification and text generation

During inference each window of a 2‑second segment is passed through the
selected model.  Predictions from the seven windows are combined using a
majority vote to obtain one label per category.  The script
`Classifiers/multi_inference.py` loads multiple checkpoints, performs this voting
procedure and maps the predicted indices to text via
`label_mappings.json`.  The individual pieces are merged into a single English
phrase like:

```
A <cluster> which is more specifically <label>, with dominant color <color>, ...
```

This description can be written to `prompts.txt` and used directly as input to a
text-to-video system.

### Example usage

```
python Classifiers/multi_inference.py \
  --eeg path/to/segmented_eeg.npy \
  --blocks 0 1 \
  --concepts 0 1 \
  --repetitions 0 1 \
  --checkpoint_root ./Checkpoints/multi/0/glmnet
```

The script evaluates all windows for every selected segment and prints the
resulting prompt along with a confidence score for each label.
