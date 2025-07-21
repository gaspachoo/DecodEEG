#!/usr/bin/env python
"""
Extract first-window EEG segments and save one file per vidéo.

Entrées
-------
- data/Preprocessing/Segmented_500ms_sw/sub3.npy
    shape = (block, concept, repet, window, C, T)

Sorties
-------
- data/gaspardnew/eeg_firstw/Block{b}/{id}.npy
    où id = concept * 5 + repet + 1      (1‥200)
    contenu = ndarray shape (C, T), dtype d’origine
"""
import os,sys
import numpy as np
from tqdm import trange  # barre de progression optionnelle

project_root = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# ----------- chemins -----------
IN_FILE   = "data/Preprocessing/Segmented_500ms_sw/sub3.npy"
OUT_ROOT  = "data/gaspardnew/eeg_firstw/sub3"

# ----------- chargement -----------
eeg = np.load(IN_FILE)                      # (B, 40, 5, W, C, T)
B, C, R, W, CH, T = eeg.shape               # R = 5  (répétitions), W >= 1
assert W >= 1, "Pas de fenêtre 0 !"

# ----------- sauvegarde -----------
for b in trange(B, desc="Bloc"):
    block_dir = os.path.join(OUT_ROOT, f"Block{b}")
    os.makedirs(block_dir, exist_ok=True)

    for c in range(40):            # concepts
        for r in range(5):         # répétitions
            id_ = c * 5 + r + 1    # 1 … 200
            seg = eeg[b, c, r, 0]  # (C, T)  fenêtre 0

            out_path = os.path.join(block_dir, f"{id_}.npy")
            np.save(out_path, seg.astype(eeg.dtype))

print(f"✅ Terminé : fichiers enregistrés dans {OUT_ROOT}")
