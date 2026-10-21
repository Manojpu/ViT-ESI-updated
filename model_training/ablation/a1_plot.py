"""
A1 figure: 6 x 8 heatmaps of the paired change vs the intact model for each metric, from
a1_head_ablation.py's a1_summary.json. Cells whose 95% CI excludes 0 are marked with *.
A side column shows the whole-layer ablation.

Example:
    python model_training/ablation/a1_plot.py -summary <run>/a1_summary.json -out <run>/a1_heads.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [("LE_mm", "ΔLE (mm)"), ("AUC", "ΔAUC"), ("nMSE", "ΔnMSE"), ("PSNR", "ΔPSNR (dB)"), ("time_err_ms", "Δtime error (ms)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-summary", required=True)
    p.add_argument("-out", required=True)
    args = p.parse_args()
    with open(args.summary) as f:
        s = json.load(f)
    rows = s["deltas"]
    n_layers = 1 + max(r["layer"] for r in rows)
    n_heads = 1 + max(max(r["heads"]) for r in rows)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.6))
    for ax, (m, name) in zip(axes, PANELS):
        grid = np.full((n_layers, n_heads + 2), np.nan)  # 8 heads, a gap, whole layer
        sig = np.zeros_like(grid, dtype=bool)
        for r in rows:
            col = r["heads"][0] if len(r["heads"]) == 1 else n_heads + 1
            grid[r["layer"], col] = r[f"d_{m}"]
            lo, hi = r[f"d_{m}_ci"]
            sig[r["layer"], col] = lo > 0 or hi < 0
        heads_only = grid[:, :n_heads]
        lim = np.nanmax(np.abs(heads_only)) or 1.0
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for (i, j), v in np.ndenumerate(grid):
            if np.isnan(v):
                continue
            txt = f"{v:.2g}" + ("*" if sig[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5)
        ax.set_xticks(list(range(n_heads)) + [n_heads + 1])
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)] + ["all"], fontsize=7)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=7)
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.035)
    intact = s["intact"]
    fig.suptitle(
        f"A1: zero-ablate one head (or a whole layer) vs intact ViTESI on {s['n_samples']} samples "
        f"(intact LE {intact['LE_mm']:.2f} mm). * = 95% CI excludes 0. Colour scale fits single heads; "
        "'all' can exceed it.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
