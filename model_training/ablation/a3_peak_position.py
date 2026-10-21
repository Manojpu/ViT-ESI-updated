"""
A3 (issue #5): localisation error vs. where in the window the activity peaks. No retraining.

Scores one or more trained models on a SEREEGA set with the shared harness, but writes one
row per *patch* instead of per sample: the harness's per-sample metrics are the mean over
patches, and each patch has its own peak time. Per patch we call `sample_metrics` with that
single seed, which gives exactly the term the harness averages (same eval zone, same
t_eval_gt, same metric code). The harness's overlap rule (keep only the stronger of
overlapping patches 1 and 2) is applied first, so the kept patches match the harness.

Peak time of a patch = argmax_t |j[seed, t]| (the time the LE is scored at). It is
cross-checked against the generator's ERP centre for that patch (md JSON `erp[k].center`).

Per model it writes per_patch.csv, per_sample.csv (harness-identical), and
summary.json with 50 ms bins (mean + 95% CI, bootstrap over samples).
Plot with `a3_plot.py`.

Example (from repo root):
    python model_training/ablation/a3_peak_position.py widepeak2k_s2 -root_simu . -to_load 2000 \
        -run VIT=<ckpt.pt> -out model_training/results/ablation/a3/<run>
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MT_DIR = Path(__file__).resolve().parents[1]
if str(MT_DIR) not in sys.path:
    sys.path.insert(0, str(MT_DIR))

from ablation.evaluate import (  # noqa: E402
    _git_info,
    _sha256,
    build_model,
    load_context,
    load_module_weights,
    rescale_prediction,
    sample_metrics,
)
from ablation.seed import set_seed  # noqa: E402

METRICS = ["LE_mm", "AUC", "nMSE", "time_err_ms"]
# Patch peak centres of the training set train8k_s0 (data_generation/FROZEN_DATASETS.md).
TRAIN_PEAK_RANGE_MS = (89.0, 487.0)


def kept_patches(j, seeds, patches):
    """The harness's overlap rule (utl_eval.sample_metrics), returning the kept patch indices."""
    if len(patches) >= 2 and set(patches[0]) & set(patches[1]):
        to_keep = int(torch.argmax(torch.Tensor([j[seeds[0], :].abs().max(), j[seeds[1], :].abs().max()])))
        return [to_keep]
    return list(range(len(seeds)))


def score(model, ctx, *, loss="cosine", seed=0, device="cuda"):
    model = model.to(device).eval()
    ds = ctx.dataset
    patch_rows, sample_rows = [], []
    for k in range(len(ds)):
        set_seed(seed + k)  # same per-sample noise as the harness
        M, j = ds[k]
        M, j = M.float(), j.float()
        M_unscaled = M * ds.max_eeg[k]
        j_unscaled = j * ds.max_src[k]

        md = ds.md_dict[ctx.md_keys[k]]
        seeds = md["seeds"]
        if type(seeds) is int:
            seeds = [seeds]
        patches = [md["act_src"][f"patch_{kk+1}"] for kk in range(len(seeds))]
        erp = md.get("erp")
        if isinstance(erp, dict):
            erp = [erp]

        with torch.no_grad():
            j_hat = model(M.unsqueeze(0).to(device)).squeeze().cpu()
        j_hat = rescale_prediction(j_hat, M_unscaled, ctx.fwd, loss, ds.max_src[k])

        full = sample_metrics(j, j_unscaled, j_hat, seeds, patches, ctx.spos, ctx.neighbors, ctx.t_vec)
        sample_rows.append(
            {
                "index": k,
                "id": md.get("id", ctx.md_keys[k]),
                "LE_mm": float(full["le"]) * 1e3,
                "AUC": float(full["auc"]),
                "nMSE": float(full["nmse"]),
                "PSNR": float(full["psnr"]),
                "time_err_ms": float(full["te"]) * 1e3,
                "n_patch": len(seeds),
                "overlap": bool(full["overlap"]),
            }
        )

        for kk in kept_patches(j, seeds, patches):
            s = seeds[kk]
            r = sample_metrics(j, j_unscaled, j_hat, [s], [patches[kk]], ctx.spos, ctx.neighbors, ctx.t_vec)
            t_gt = int(torch.argmax(j[s, :].abs()))
            patch_rows.append(
                {
                    "index": k,
                    "patch": kk + 1,
                    "seed_region": int(s),
                    "peak_time_ms": float(ctx.t_vec[t_gt]) * 1e3,
                    "erp_center_ms": float(erp[kk]["center"]) if erp else np.nan,
                    "n_patch": len(seeds),
                    "overlap": bool(full["overlap"]),
                    "LE_mm": float(r["le"]) * 1e3,
                    "AUC": float(r["auc"]),
                    "nMSE": float(r["nmse"]),
                    "time_err_ms": float(r["te"]) * 1e3,
                }
            )
        if (k + 1) % 200 == 0:
            print(f"  {k + 1} samples done")
    return pd.DataFrame(patch_rows), pd.DataFrame(sample_rows)


def bootstrap_ci(df, metric, n_boot=2000, rng=None):
    """95% CI of the patch-level mean, resampling whole samples (patches of one sample are correlated)."""
    groups = [g[metric].to_numpy() for _, g in df.groupby("index")]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bin_table(df, width_ms=50, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    edges = np.arange(0, 1000 + width_ms, width_ms)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = df[(df.peak_time_ms >= lo) & (df.peak_time_ms < hi)]
        row = {"bin_lo_ms": float(lo), "bin_hi_ms": float(hi), "n_patches": int(len(b))}
        for m in METRICS:
            if len(b):
                row[m] = float(b[m].mean())
                row[f"{m}_ci"] = bootstrap_ci(b, m, n_boot, rng) if b["index"].nunique() > 1 else None
            else:
                row[m], row[f"{m}_ci"] = None, None
        rows.append(row)
    return rows


def range_summary(df):
    lo, hi = TRAIN_PEAK_RANGE_MS
    inside = df[(df.peak_time_ms >= lo) & (df.peak_time_ms <= hi)]
    outside = df[(df.peak_time_ms < lo) | (df.peak_time_ms > hi)]
    return {
        name: {"n_patches": int(len(d)), **{m: float(d[m].mean()) if len(d) else None for m in METRICS}}
        for name, d in [("inside_train_range", inside), ("outside_train_range", outside), ("all", df)]
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("simu_name")
    p.add_argument("-root_simu", required=True)
    p.add_argument("-run", action="append", required=True, help="MODEL=path/to/weights.pt (repeatable)")
    p.add_argument("-out", required=True)
    p.add_argument("-to_load", type=int, default=2000)
    p.add_argument("-eeg_snr", type=int, default=5)
    p.add_argument("-loss", default="cosine")
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("-bin_ms", type=int, default=50)
    p.add_argument("-n_boot", type=int, default=2000)
    p.add_argument("-leadfield_mat", default=str(MT_DIR.parent / "anatomy" / "leadfield_75_20k.mat"))
    p.add_argument("-vit_embed_dim", type=int, default=256)
    p.add_argument("-vit_depth", type=int, default=6)
    p.add_argument("-vit_heads", type=int, default=8)
    p.add_argument("-vit_mlp_dim", type=int, default=512)
    p.add_argument("-vit_dropout", type=float, default=0.1)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("HARD STOP: -device cuda requested but CUDA is not available (no silent CPU fallback).")

    ctx = load_context(args.root_simu, args.simu_name, args.to_load, args.eeg_snr, args.leadfield_mat)
    n_times = ctx.config["rec_info"]["n_times"]
    commit, dirty = _git_info()

    for spec in args.run:
        name, ckpt = spec.split("=", 1)
        model = build_model(name.split(":")[0], args, ctx.fwd.shape[0], ctx.fwd.shape[1], n_times)
        load_module_weights(model, ckpt)
        print(f"A3: {name} on {ctx.simu_path} ({len(ctx.dataset)} samples, {args.device})")
        patches_df, samples_df = score(model, ctx, loss=args.loss, seed=args.seed, device=args.device)

        # Consistency: the mean of a sample's patch LEs must equal the harness's sample LE.
        chk = patches_df.groupby("index").LE_mm.mean().reindex(samples_df["index"]).to_numpy()
        max_diff = float(np.max(np.abs(chk - samples_df.LE_mm.to_numpy())))
        if max_diff > 1e-6:
            raise SystemExit(f"per-patch LE does not decompose the harness LE (max diff {max_diff} mm)")
        peak_vs_erp = (patches_df.peak_time_ms - patches_df.erp_center_ms).abs()

        out = Path(args.out) / name.replace(":", "_")
        out.mkdir(parents=True, exist_ok=True)
        patches_df.to_csv(out / "per_patch.csv", index=False)
        samples_df.to_csv(out / "per_sample.csv", index=False)
        summary = {
            "model": name,
            "ckpt": os.path.abspath(ckpt),
            "ckpt_sha256": _sha256(ckpt),
            "dataset": ctx.simu_path,
            "n_samples": int(len(samples_df)),
            "n_patches": int(len(patches_df)),
            "sample_means": {m: float(samples_df[m].mean()) for m in METRICS + ["PSNR"]},
            "decomposition_max_abs_diff_mm": max_diff,
            "peak_vs_erp_center_ms": {
                "median": float(peak_vs_erp.median()),
                "p95": float(peak_vs_erp.quantile(0.95)),
                "frac_within_10ms": float((peak_vs_erp <= 10).mean()),
            },
            "train_peak_range_ms": TRAIN_PEAK_RANGE_MS,
            "ranges": range_summary(patches_df),
            "bins": bin_table(patches_df, args.bin_ms, args.n_boot, args.seed),
            "eeg_snr": args.eeg_snr,
            "seed": args.seed,
            "device": args.device,
            "torch": torch.__version__,
            "git_commit": commit,
            "git_dirty": dirty,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        r = summary["ranges"]
        print(
            f"  {name}: sample LE {summary['sample_means']['LE_mm']:.2f} mm | patch LE inside "
            f"{r['inside_train_range']['LE_mm']} / outside {r['outside_train_range']['LE_mm']} mm -> {out}"
        )


if __name__ == "__main__":
    main()
