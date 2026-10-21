"""
A3 figure: per-bin mean and 95% CI of each metric vs patch peak time, from a3_peak_position.py
summary.json files. Training peak range is shaded.

Example:
    python model_training/ablation/a3_plot.py -summary widepeak=<run>/widepeak2k_s2/VIT/summary.json \
        -summary test2k=<run>/test2k_s1/VIT/summary.json -out <run>/a3_vit.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [("LE_mm", "LE (mm)"), ("AUC", "AUC"), ("nMSE", "nMSE"), ("time_err_ms", "Time error (ms)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-summary", action="append", required=True, help="LABEL=path/to/summary.json (repeatable)")
    p.add_argument("-out", required=True)
    p.add_argument("-min_patches", type=int, default=20, help="hide bins with fewer patches")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for spec in args.summary:
        label, path = spec.split("=", 1)
        with open(path) as f:
            s = json.load(f)
        bins = [b for b in s["bins"] if b["n_patches"] >= args.min_patches]
        x = [(b["bin_lo_ms"] + b["bin_hi_ms"]) / 2 for b in bins]
        for ax, (m, name) in zip(axes.flat, PANELS):
            y = [b[m] for b in bins]
            lo = [b[f"{m}_ci"][0] for b in bins]
            hi = [b[f"{m}_ci"][1] for b in bins]
            ax.plot(x, y, marker="o", ms=3, label=f"{label} ({s['model']})")
            ax.fill_between(x, lo, hi, alpha=0.2)
            ax.set_ylabel(name)
    lo_ms, hi_ms = s["train_peak_range_ms"]
    for ax in axes.flat:
        ax.axvspan(lo_ms, hi_ms, color="grey", alpha=0.12, lw=0)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Patch peak time in window (ms)")
    axes[0, 0].legend(fontsize=8, title=f"grey = training peak range {lo_ms:.0f}-{hi_ms:.0f} ms", title_fontsize=8)
    fig.suptitle(f"A3: error vs peak position (50 ms bins, mean + 95% CI, bins with < {args.min_patches} patches hidden)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
