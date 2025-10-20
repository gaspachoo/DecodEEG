from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
import math
from scipy.fftpack import fft

def standard_scale_features(X, scaler=None, return_scaler=False):
    """Scale features with ``StandardScaler``.

    Parameters
    ----------
    X : np.ndarray
        Array of shape ``(N, ...)`` to scale.
    scaler : sklearn.preprocessing.StandardScaler or None
        If ``None`` a new scaler is fitted on ``X``. Otherwise ``X`` is
        transformed using the provided scaler.
    return_scaler : bool, optional
        Whether to return the fitted scaler.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as ``X``.
    sklearn.preprocessing.StandardScaler, optional
        Returned only if ``return_scaler`` is ``True``.
    """

    orig_shape = X.shape[1:]
    X_2d = X.reshape(len(X), -1)

    if scaler is None:
        scaler = StandardScaler().fit(X_2d)

    X_scaled = scaler.transform(X_2d).reshape((len(X),) + orig_shape)

    if return_scaler:
        return X_scaled, scaler
    return X_scaled


def compute_raw_stats(X: np.ndarray):
    """Compute per-channel mean and std from training data."""
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2)) + 1e-6
    return mean, std


def normalize_raw(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize raw EEG with provided statistics."""
    return (X - mean[None, :, None]) / std[None, :, None]


def load_scaler(path: str) -> StandardScaler:
    """Load a ``StandardScaler`` object from ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_raw_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load raw EEG normalization statistics from a ``.npz`` file."""
    data = np.load(path)
    return data["mean"], data["std"]


def subject_split(
    seed: int,
    subjects: list[str],
    ckpt_seed_dir: str,
    n_select: int | None = None,
    n_train: int = 13,
    n_val: int = 2,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically split subjects for multi-subject training.

    Parameters
    ----------
    seed : int
        Seed controlling the shuffle.
    subjects : list[str]
        Available subject names.
    ckpt_seed_dir : str
        Directory where ``subjects.txt`` will be written.
    n_select : int, optional
        Number of subjects drawn from ``subjects`` before splitting.
        If ``None`` all subjects are considered.
    n_train : int, optional
        Number of subjects used for training.
    n_val : int, optional
        Number of subjects used for validation.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Lists of train, validation and test subjects.
    """

    rng = np.random.default_rng(seed)
    subj_sorted = sorted(subjects)
    rng.shuffle(subj_sorted)

    if n_select is not None:
        if n_select > len(subj_sorted):
            raise ValueError("Not enough subjects to select")
        selected = subj_sorted[:n_select]
    else:
        selected = subj_sorted

    if n_train + n_val > len(selected):
        raise ValueError("Not enough subjects to split")

    train_subj = selected[:n_train]
    val_subj = selected[n_train : n_train + n_val]
    test_subj = [s for s in subj_sorted if s not in train_subj and s not in val_subj]

    os.makedirs(ckpt_seed_dir, exist_ok=True)
    txt_path = os.path.join(ckpt_seed_dir, "subjects.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write("train:" + ",".join(train_subj) + "\n")
            f.write("val:" + ",".join(val_subj) + "\n")
            f.write("test:" + ",".join(test_subj) + "\n")

    return train_subj, val_subj, test_subj


def block_split(seed: int, n_blocks: int, ckpt_seed_dir: str) -> tuple[int, int]:
    """Choose validation and test blocks for single-subject training.

    Parameters
    ----------
    seed : int
        Seed controlling the selection.
    n_blocks : int
        Total number of blocks in the data.
    ckpt_seed_dir : str
        Directory where ``blocks.txt`` will be written.

    Returns
    -------
    tuple[int, int]
        Selected validation and test block indices.
    """

    rng = np.random.RandomState(seed)
    val_block, test_block = rng.choice(np.arange(n_blocks), size=2, replace=False)

    os.makedirs(ckpt_seed_dir, exist_ok=True)
    txt_path = os.path.join(ckpt_seed_dir, "blocks.txt")
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write(f"val:{int(val_block)}\n")
            f.write(f"test:{int(test_block)}\n")

    return int(val_block), int(test_block)





def DE_PSD(data, fre, time_window, which="both"):
    """Compute Differential Entropy (DE) and/or Power Spectral Density (PSD).

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(n_channels, n_samples)`` containing the EEG segment.
    fre : int
        Sampling frequency of ``data``.
    time_window : float
        Window length in seconds used for the STFT.
    which : {"both", "de", "psd"}
        Selects which features to compute.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Depending on ``which`` either DE, PSD or both.
    """
    # initialize the parameters
    # STFTN=stft_para['stftn']
    # fStart=stft_para['fStart']
    # fEnd=stft_para['fEnd']
    # fs=stft_para['fs']
    # window=stft_para['window']

    STFTN = 200
    fStart = [1, 4, 8, 14, 31]
    fEnd = [4, 8, 14, 31, 99]  # bands : delta, theta, alpha, beta, gamma
    window = time_window
    fs = fre

    WindowPoints = fs * window

    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    for i in range(0, len(fStart)):
        fStartNum[i] = int(fStart[i] / fs * STFTN)
        fEndNum[i] = int(fEnd[i] / fs * STFTN)

    # print(fStartNum[0],fEndNum[0])
    n = data.shape[0]

    # print(m,n,l)
    if which in ("both", "psd"):
        psd = np.zeros((n, len(fStart)), dtype=float)
    else:
        psd = None

    if which in ("both", "de"):
        de = np.zeros((n, len(fStart)), dtype=float)
    else:
        de = None
    # Hanning window
    Hlength = int(window * fs)  # added int()
    # Hwindow=hanning(Hlength)
    Hwindow = np.array(
        [
            0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1))
            for n in range(1, Hlength + 1)
        ]
    )

    dataNow = data[0:n]
    for j in range(n):
        temp = dataNow[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = abs(FFTdata[0 : int(STFTN / 2)])
        for p in range(len(fStart)):
            E = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E += magFFTdata[p0] * magFFTdata[p0]
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            if psd is not None:
                psd[j][p] = E
            if de is not None:
                de[j][p] = math.log(100 * E, 2)

    if which == "de":
        return de
    if which == "psd":
        return psd
    return de, psd
