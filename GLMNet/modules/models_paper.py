"""
Different EEG encoders for comparison

SA GA

shallownet, deepnet, eegnet
"""

import math
import numpy as np
from EEG_preprocessing.DE_PSD import DE_PSD

import torch
import torch.nn as nn

class shallownet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(shallownet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            #nn.AdaptiveAvgPool2d((1, 26)),
            nn.Dropout(0.5),
        )
        n_samples = math.floor((T - 75) / 5 + 1)
        self.out = nn.Linear(40 * n_samples, out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
class deepnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(deepnet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )

        # compute output dimension using a dummy input tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            out_features = self.net(dummy).view(1, -1).shape[1]
        self.out = nn.Linear(out_features, out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
    
class eegnet(nn.Module):
    def __init__(self, out_dim, C, T):
        super(eegnet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16), 
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )

        # compute output dimension using a dummy input tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            out_features = self.net(dummy).view(1, -1).shape[1]
        self.out = nn.Linear(out_features, out_dim)
    
    def forward(self, x):               #input:(batch,1,C,T)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class mlpnet(nn.Module):
    def __init__(self, out_dim, input_dim):
        super(mlpnet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim)
        )

    @staticmethod
    def compute_features(raw: np.ndarray, fs: int = 200, win_sec: float = 0.5) -> np.ndarray:
        """Compute DE features from raw EEG."""
        feats = np.zeros((raw.shape[0], raw.shape[1], 5), dtype=np.float32)
        for i, seg in enumerate(raw):
            de = DE_PSD(seg, fs, win_sec, which="de")
            feats[i] = de
        return feats
        
    def forward(self, x):               #input:(batch,C,5)
        out = self.net(x)
        return out