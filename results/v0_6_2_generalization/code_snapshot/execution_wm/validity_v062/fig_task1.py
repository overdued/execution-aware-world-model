"""Task 1 图：action_space_pca.png（train-only PCA，PC1-PC2 / PC1-PC3）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from execution_wm.validity_v062.common import OUT

nov = pd.read_csv(os.path.join(OUT, "metrics", "command_novelty.csv"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
groups = [("train", "0.75", 6, "o"), ("A_unseen_anchor_seen_family", "C0", 14, "o"),
          ("B_seen_anchor_unseen_family", "C3", 14, "s"),
          ("C_unseen_anchor_unseen_family", "C1", 16, "^")]
for name, col, s, mk in groups:
    d = nov[nov.split == name]
    ax[0].scatter(d.pca_pc1, d.pca_pc2, s=s, c=col, marker=mk, alpha=.6,
                  label=name.split("_")[0], edgecolors="none")
    ax[1].scatter(d.pca_pc1, d.pca_pc3, s=s, c=col, marker=mk, alpha=.6, edgecolors="none")
for a, yl in ((ax[0], "PC2"), (ax[1], "PC3")):
    a.set_xlabel("PC1 (train-fitted)"); a.set_ylabel(yl)
    a.grid(alpha=.3); a.axhline(0, color="k", lw=.4); a.axvline(0, color="k", lw=.4)
ax[0].legend(fontsize=7, markerscale=1.6)
ax[0].set_title("Command PCA: A lies on train support; B/C separate (multi-axis co-activation)", fontsize=8)
ax[1].set_title("PC1 vs PC3 (same fit)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "action_space_pca.png"), dpi=140)
print("-> figures/action_space_pca.png")
