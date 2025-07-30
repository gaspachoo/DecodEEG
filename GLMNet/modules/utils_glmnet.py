import torch
import torch.nn as nn
from GLMNet.modules.models_paper import shallownet, mlpnet
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

class GLMNet(nn.Module):
    """ShallowNet (raw) + MLP (freq) → concat → FC."""

    def __init__(self, occipital_idx, C: int, T: int, out_dim: int = 40, emb_dim: int = 512):
        """Construct the GLMNet model.

        Parameters
        ----------
        occipital_idx : iterable
            Indexes of occipital channels used for the local branch.
        out_dim : int
            Dimension of the classification output.
        emb_dim : int
            Dimension of the intermediate embeddings (each branch outputs
            ``emb_dim`` features).
        T : int
            Number of temporal samples of the raw EEG. This value can vary
            depending on the dataset.
        """
        super().__init__()
        self.occipital_idx = list(occipital_idx)

        # Global branch processing raw EEG
        self.raw_global = shallownet(emb_dim, C, T)
        # Local branch processing spectral features
        self.freq_local = mlpnet(emb_dim, len(self.occipital_idx) * 5)

        # Projection of concatenated features followed by classifier
        self.projection = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim // 2),
        )
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim // 2, out_dim),
        )

    @staticmethod
    def infer_out_dim(state: dict) -> int:
        """Infer ``out_dim`` from a checkpoint state dict."""
        if "classifier.1.weight" in state:
            return state["classifier.1.weight"].shape[0]
        if "fc.2.weight" in state:
            return state["fc.2.weight"].shape[0]
        if "fc.weight" in state:
            return state["fc.weight"].shape[0]
        raise KeyError("Cannot infer output dimension from checkpoint")

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str, device: str = "cpu") -> dict:
        """Simple wrapper around ``torch.load``."""
        return torch.load(ckpt_path, map_location=device)


    def forward(self, x_raw, x_feat, return_features: bool = False):
        """Forward pass of the network.

        Parameters
        ----------
        x_raw : torch.Tensor
            Raw EEG of shape ``(B, 1, 62, T)``.
        x_feat : torch.Tensor
            Spectral features of shape ``(B, 62, 5)``.
        return_features : bool, optional
            If ``True`` returns the projected features before the
            classification layer. Defaults to ``False``.
        """

        g_raw = self.raw_global(x_raw)
        l_freq = self.freq_local(x_feat[:, self.occipital_idx, :])

        features = torch.cat([g_raw, l_freq], dim=1)
        projected = self.projection(features)

        if return_features:
            return projected

        return self.classifier(projected)


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
