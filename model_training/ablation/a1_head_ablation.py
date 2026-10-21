"""
A1 (issue #6): attention head ablation. No retraining.

Zero-ablates one attention head at a time (6 layers x 8 heads = 48 runs), plus every
whole layer (all 8 heads, the required sanity check), and scores each with the shared
harness on the same samples and noise. Paired per-sample deltas vs the intact model give
a mean change and a 95% bootstrap CI per head.

How a head is switched off: in nn.MultiheadAttention the heads' outputs are concatenated
and multiplied by out_proj.weight, so head h only ever touches columns h*d:(h+1)*d of
that weight (d = embed_dim / num_heads). Zeroing those columns removes exactly that
head's contribution (the out_proj bias stays). It's a weight edit, so it also works on
PyTorch's fused fast path, where Python hooks are skipped. `verify_head_mask` checks
it against a hand-written attention forward.

Runs are resumable: a mask whose summary.json exists is skipped.

Example (from repo root):
    python model_training/ablation/a1_head_ablation.py test2k_s1 -root_simu . -to_load 2000 \
        -ckpt <VIT_model.pt> -out model_training/results/ablation/a1/<run>
"""

import argparse
import contextlib
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
    evaluate,
    load_context,
    load_module_weights,
    summarize,
)

METRICS = ["LE_mm", "AUC", "nMSE", "PSNR", "time_err_ms"]


def attn_layers(model):
    return [layer.self_attn for layer in model.model.encoder.layers]


@contextlib.contextmanager
def head_mask(model, layer, heads):
    """Zero the out_proj columns of `heads` in encoder layer `layer`; restore on exit."""
    attn = attn_layers(model)[layer]
    d = attn.embed_dim // attn.num_heads
    w = attn.out_proj.weight
    saved = w.data.clone()
    for h in heads:
        w.data[:, h * d : (h + 1) * d] = 0
    try:
        yield
    finally:
        w.data.copy_(saved)


def manual_attention(attn, x, drop_heads=()):
    """Hand-written multi-head self-attention (eval mode, no masks) with given heads zeroed."""
    B, T, D = x.shape
    H, d = attn.num_heads, D // attn.num_heads
    qkv = torch.nn.functional.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = (t.reshape(B, T, H, d).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
    a = torch.softmax(q @ k.transpose(-2, -1) / d**0.5, dim=-1)
    heads = a @ v  # (B, H, T, d)
    for h in drop_heads:
        heads[:, h] = 0
    return attn.out_proj(heads.transpose(1, 2).reshape(B, T, D))


def verify_head_mask(model, x, device):
    """The weight-edit mask must match a hand-written attention with that head removed."""
    model.eval()
    worst = 0.0

    def rel(got, ref):  # float32 rounding scales with the output size, so compare relatively
        return float((got - ref).abs().max() / ref.abs().max())

    with torch.no_grad():
        for li, attn in enumerate(attn_layers(model)):
            xin = torch.randn(1, model.model.n_times, attn.embed_dim, device=device)
            for heads in ([0], [attn.num_heads - 1], list(range(attn.num_heads))):
                ref = manual_attention(attn, xin, heads)
                with head_mask(model, li, heads):
                    got = attn(xin, xin, xin, need_weights=False)[0]
                worst = max(worst, rel(got, ref))
            # No mask: the module must also agree with the manual forward.
            worst = max(worst, rel(attn(xin, xin, xin, need_weights=False)[0], manual_attention(attn, xin)))
        # Masking must change the whole-model output, and restoring must undo it exactly.
        base = model(x)
        with head_mask(model, 0, list(range(attn_layers(model)[0].num_heads))):
            changed = float((model(x) - base).abs().max())
        restored = float((model(x) - base).abs().max())
    return {"max_rel_diff_vs_manual": worst, "full_layer0_output_change": changed, "restore_diff": restored}


def paired_delta(base, run, metric, n_boot, rng):
    d = run[metric].to_numpy() - base[metric].to_numpy()
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("simu_name")
    p.add_argument("-root_simu", required=True)
    p.add_argument("-ckpt", required=True)
    p.add_argument("-out", required=True)
    p.add_argument("-to_load", type=int, default=2000)
    p.add_argument("-eeg_snr", type=int, default=5)
    p.add_argument("-loss", default="cosine")
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("-n_boot", type=int, default=2000)
    p.add_argument("-only", default=None, help="comma list of run names to do, e.g. intact,L0_all,L0_H3 (smoke)")
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
    model = build_model("VIT", args, ctx.fwd.shape[0], ctx.fwd.shape[1], ctx.config["rec_info"]["n_times"])
    load_module_weights(model, args.ckpt)
    model = model.to(args.device).eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x0 = ctx.dataset[0][0].float().unsqueeze(0).to(args.device)
    check = verify_head_mask(model, x0, args.device)
    print(f"mask check: {check}")
    if check["max_rel_diff_vs_manual"] > 1e-4 or check["full_layer0_output_change"] < 1e-3 or check["restore_diff"] != 0:
        raise SystemExit(f"head mask failed its sanity check: {check}")

    n_layers, n_heads = args.vit_depth, args.vit_heads
    runs = [("intact", None, [])]
    runs += [(f"L{l}_all", l, list(range(n_heads))) for l in range(n_layers)]
    runs += [(f"L{l}_H{h}", l, [h]) for l in range(n_layers) for h in range(n_heads)]
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r[0] in keep]

    for name, layer, heads in runs:
        rdir = out / name
        if (rdir / "summary.json").exists():
            continue
        print(f"A1 {name}: layer {layer} heads {heads}")
        ctxm = head_mask(model, layer, heads) if heads else contextlib.nullcontext()
        with ctxm:
            df = evaluate(model, ctx, loss=args.loss, seed=args.seed, device=args.device)
        rdir.mkdir(exist_ok=True)
        df.to_csv(rdir / "per_sample.csv", index=False)
        with open(rdir / "summary.json", "w") as f:
            json.dump({"run": name, "layer": layer, "heads": heads, "metrics": summarize(df)}, f, indent=2)

    # Aggregate: paired deltas vs intact.
    base = pd.read_csv(out / "intact" / "per_sample.csv")
    rng = np.random.default_rng(args.seed)
    table = []
    for name, layer, heads in runs:
        if name == "intact" or not (out / name / "per_sample.csv").exists():
            continue
        run = pd.read_csv(out / name / "per_sample.csv")
        if not (run["index"].to_numpy() == base["index"].to_numpy()).all():
            raise SystemExit(f"{name}: sample order differs from intact")
        row = {"run": name, "layer": layer, "heads": heads}
        for m in METRICS:
            row[f"d_{m}"], row[f"d_{m}_ci"] = paired_delta(base, run, m, args.n_boot, rng)
        table.append(row)
    commit, dirty = _git_info()
    agg = {
        "dataset": ctx.simu_path,
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_sha256": _sha256(args.ckpt),
        "n_samples": int(len(base)),
        "intact": {m: float(base[m].mean()) for m in METRICS},
        "mask_check": check,
        "ablation": "zero (out_proj columns of the head set to 0)",
        "deltas": table,
        "eeg_snr": args.eeg_snr,
        "seed": args.seed,
        "device": args.device,
        "torch": torch.__version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
    }
    with open(out / "a1_summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"intact LE {agg['intact']['LE_mm']:.3f} mm; {len(table)} ablations aggregated -> {out / 'a1_summary.json'}")


if __name__ == "__main__":
    main()
