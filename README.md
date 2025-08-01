# DecodEEG Overview

This repository collects utilities to classify EEG recordings and
convert those predictions into natural language prompts for text-to-video models.
Models such as **EEGNet**, **DeepNet** and **GLMNet** can be trained or loaded from
checkpoint directories.

## Datasets
- This repository is ready to use on SEED-DV dataset, https://bcmi.sjtu.edu.cn/home/eeg2video/
- You can apply for downloading it, or use it on your own dataset, editing `segment_raw_signals_200Hz.py` input processing must be sufficient.
- Put the dataset in the following folder : `data/`, folders `meta_info` and `EEG` must be here. 

## Data preparation

- Raw recordings are first divided into 2‑second segments using
`segment_raw_signals_200Hz.py`.  Each segment has shape `(block, concept,
repetition, channels, time)` and is sampled at 200 Hz.
- To obtain more training samples we apply a sliding‑window strategy with `segment_sliding_window.py`: every 2‑second segment is sliced into a certain amount of sliding overlapping windows. We mostly used two different cuttings : 7 windows of 500 ms with an overlap of 250 ms, and 3 windows of 250 ms with an overlap of 500 ms.
The resulting tensor has shape `(block, concept, repetition, window, channels, time)`.
- You must also generate the `data/meta_info/All_video_label_cluster.npy` file using `EEG_preprocessing/create_label_cluster.py`.

## Training

Any of the supported encoders can be trained on these windows.  Models predict
class labels for categories such as *color*, *label cluster* or *object number*.
Checkpoints are stored under `Classifiers/checkpoints/<mode>/<seed>/<model>/<category>`.
For single-subject runs the hierarchy becomes
`Classifiers/checkpoints/mono/<subject>/<ordered|shuffle>/seed<seed>/<model>/<category>`.

Training across all categories can be automated with the Makefile:

```bash
# Multi-subject
make checkpoints_multi SEED=0 MODEL=glmnet use_wandb=1

# Mono-subject (ordered or shuffle)
make checkpoints_mono SUBJECT=sub3 SEED=0 MODEL=glmnet shuffle=1
```

These two commands will run respectively `train_classifier_multi` and `train_classifier_mono`

For mono, 2 splitting methods for the dataset have been tested :
- Shuffle : Out of the 1400 EEGs from all blocks (segmented in sliding windows), we take 80% for training, 10% for validation, 10% for testing.
- Ordered : 5 blocks for training, 1 block for validation, 1 block for testing.

For multi, 13 subject's EEGs are used for training, 2 for validation, 5 for testing.

You can also classify the category `label` for a classic 40-classes classification.

## Classification and text generation

 - During inference you must use a EEG preprocessing which has the same number of windows of the one which have been used to generate the checkpoints .  
 - The script `Classifiers/multi_inference.py` loads multiple checkpoints, performs this voting procedure and maps the predicted indices to text via `label_mappings.json`.  Any of the supported models (``glmnet``, ``eegnet`` or ``deepnet``) can be selected with ``--model``. 
- The individual pieces are merged into a single English
phrase like:

```
A <cluster> which is more specifically <label>, with dominant color <color>, ...
```

This description can be written to `prompts.txt` and used directly as input to a text-to-video system. We recommend the usage of VideoTuna.
https://github.com/VideoVerses/VideoTuna

### Example usage

```
python Classifiers/multi_inference.py \
  --eeg path/to/segmented_eeg.npy \
  --blocks [0, 7] \
  --concepts [0, 10] \
  --repetitions [0] \
  --checkpoint_root ./Classifiers/checkpoints/multi/0/eegnet \
  --model eegnet
```

The script evaluates all windows for every selected segment and prints the resulting prompt along with a confidence score for each label.
