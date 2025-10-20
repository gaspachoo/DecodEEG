# EEG_preprocessing

This folder gathers all the utilities to prepare the EEG signals before model training.
Below is a short description of each script.

| Script | Description |
| ------ | ----------- |
| `__init__.py` | Marks this folder as a Python package.
| `create_label_cluster.py` | Creates the file `data/meta_info/All_video_label_cluster.npy` which is mandatory for training on the corresponding category.
| `segment_raw_signals_200Hz.py` | Splits raw SEED-DV recordings (62 channels, 200 Hz) into 2-second windows organized as `(block, concept, repetition, channel, time)`.
| `segment_sliding_window.py` | Further divides the 2-second windows into smaller windows with overlap, producing a `(block, concept, repetition, window, channel, time)` tensor.

