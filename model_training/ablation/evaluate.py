"""
Shared evaluation harness for the Objective 2 ablations (issue #2).

Evaluates any `nn.Module` mapping EEG (B, 75, T) -> sources (B, 994, T) on a SEREEGA
simulation folder and writes one CSV row per sample with the 5 thesis metrics, using the
exact metric code of eval.py (`utils/utl_eval.py`).

Noise: the dataset adds fresh random noise on every load. By default each sample is
re-seeded with `seed + index` before loading, so every run (any model, any ablation) sees
the identical noisy EEG -> paired comparisons are valid.
`-legacy_order` instead reproduces eval.py's RNG order (seed_everything(0), random_split,
sequential noise) - only used to check parity with eval.py.

Example (from repo root):
    python model_training/ablation/evaluate.py mes_debug_python -root_simu . \
        -ckpt <run_dir>/trained_models/VIT_model.pt -model VIT -out model_training/results/ablation/smoke/vit
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

MT_DIR = Path(__file__).resolve().parents[1]  # model_training/
if str(MT_DIR) not in sys.path:
    sys.path.insert(0, str(MT_DIR))

from ablation.seed import set_seed  # noqa: E402
from load_data import HeadModel  # noqa: E402
from load_data.FolderStructure import FolderStructure  # noqa: E402
from loaders import EsiDatasetds_new  # noqa: E402
from utils.utl_eval import (  # noqa: E402
    load_leadfield_mat,
    load_module_weights,
    rescale_prediction,
    sample_metrics,
)


@dataclass
class EvalContext:
    dataset: EsiDatasetds_new
    md_keys: list
    fwd: np.ndarray
    spos: torch.Tensor
    neighbors: np.ndarray
    t_vec: np.ndarray
    simu_path: str
    config: dict


def load_context(
    root_simu,
    simu_name,
    to_load,
    eeg_snr,
    leadfield_mat,
    source_space="fsav_994",
    electrode_montage="standard_1020",
    orientation="constrained",
    subject_name="fsaverage",
):
    """Load a SEREEGA test set plus the head-model pieces, the same way eval.py does."""
    root_simu_path = Path(root_simu)
    if (root_simu_path / "simulation" / subject_name).is_dir():
        root_base = root_simu_path / "simulation" / subject_name
    else:
        root_base = root_simu_path
    simu_path = str(root_base / orientation / electrode_montage / source_space / "simu" / simu_name)
    config_file = f"{simu_path}/{simu_name}{source_space}_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    config["eeg_snr"] = eeg_snr
    config["simu_name"] = simu_name

    folders = FolderStructure(str(root_base), config)
    source_space_obj = HeadModel.SourceSpace(folders, config)

    fwd = np.asarray(load_leadfield_mat(leadfield_mat), dtype=np.float32)

    nb_file = f"{folders.model_folder}/fs_cortex_neighbors_994.mat"
    if not os.path.isfile(nb_file):
        raise FileNotFoundError(
            f"{nb_file} missing - run `python model_training/ablation/build_model_folder.py -root .` first"
        )
    neighbors = loadmat(nb_file)["nbs"] - 1

    fs = config["rec_info"]["fs"]
    n_times = config["rec_info"]["n_times"]
    t_vec = np.arange(0, n_times / fs, 1 / fs)
    spos = torch.from_numpy(source_space_obj.positions)  # metres

    dataset = EsiDatasetds_new(
        str(root_base),
        config_file,
        simu_name,
        source_space,
        config["electrode_space"]["electrode_montage"],
        to_load,
        eeg_snr,
        noise_type={"white": 1.0, "pink": 0.0},
    )
    md_keys = [k for k, _ in dataset.md_dict.items()]
    return EvalContext(dataset, md_keys, fwd, spos, neighbors, t_vec, simu_path, config)


def heldout_indices(n, per_valid):
    """
    Indices main_train.py holds out (and eval.py scores) for -to_load n -per_valid p:
    both call seed_everything(0) then random_split, so the held-out set depends only on (n, p).
    Thesis protocol: train8k_s0 with n=8000, p=0.25 -> 6000 train / 2000 held out.
    """
    from pytorch_lightning import seed_everything
    from torch.utils.data import random_split

    seed_everything(0, verbose=False)
    n_train = int(n * (1 - per_valid))
    _, val = random_split(range(n), [n_train, n - n_train])
    return list(val.indices)


def evaluate(model, ctx, *, loss="cosine", seed=0, device="cpu", indices=None, reseed=True):
    """
    Run `model` over the samples in `indices` (default: all) and return a per-sample DataFrame.
    reseed=True: seed+index before each load (identical noise across runs).
    """
    model = model.to(device).eval()
    ds = ctx.dataset
    if indices is None:
        indices = range(len(ds))

    rows = []
    for n, k in enumerate(indices):
        if reseed:
            set_seed(seed + int(k))
        M, j = ds[k]
        M, j = M.float(), j.float()
        M_unscaled = M * ds.max_eeg[k]
        j_unscaled = j * ds.max_src[k]

        md = ds.md_dict[ctx.md_keys[k]]
        seeds = md["seeds"]
        if type(seeds) is int:
            seeds = [seeds]
        patches = [md["act_src"][f"patch_{kk+1}"] for kk in range(len(seeds))]

        with torch.no_grad():
            j_hat = model(M.unsqueeze(0).to(device)).squeeze().cpu()
        j_hat = rescale_prediction(j_hat, M_unscaled, ctx.fwd, loss, ds.max_src[k])

        r = sample_metrics(j, j_unscaled, j_hat, seeds, patches, ctx.spos, ctx.neighbors, ctx.t_vec)

        t_peak = int(torch.argmax(j_unscaled.abs().max(dim=0).values))
        rows.append(
            {
                "index": int(k),
                "id": md.get("id", ctx.md_keys[k]),
                "LE_mm": float(r["le"]) * 1e3,
                "AUC": float(r["auc"]),
                "nMSE": float(r["nmse"]),
                "PSNR": float(r["psnr"]),
                "time_err_ms": float(r["te"]) * 1e3,
                "peak_time_ms": float(ctx.t_vec[t_peak]) * 1e3,
                "erp_center_ms": float(md["base_center"]) if "base_center" in md else np.nan,
                "n_patch": int(md.get("n_patch", len(seeds))),
                "overlap": bool(r["overlap"]),
            }
        )
        if (n + 1) % 100 == 0:
            print(f"  {n + 1} samples done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model registry
def build_model(name, args, n_sensors, n_sources, n_times):
    name = name.upper()
    if name == "VIT":
        from models.vit import EEGViTpl

        return EEGViTpl(
            num_sensor=n_sensors,
            num_source=n_sources,
            n_times=n_times,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
            dropout=args.vit_dropout,
            optimizer=None,
            lr=1e-3,
            criterion=None,
        )
    # The three baselines are built exactly like main_train.py builds them (same constructor args).
    if name == "1DCNN":
        from models.cnn_1d import CNN1Dpl

        return CNN1Dpl(
            channels=[n_sensors, args.inter_layer, n_sources],
            kernel_size=args.kernel_size,
            bias=False,
            optimizer=None,
            lr=1e-3,
            criterion=None,
        )
    if name == "LSTM":
        from models.lstm import HeckerLSTMpl

        return HeckerLSTMpl(
            n_electrodes=n_sensors,
            hidden_size=85,
            n_sources=n_sources,
            bias=False,
            optimizer=None,
            lr=1e-3,
            criterion=None,
            mc_dropout_rate=0,
        )
    if name == "DEEPSIF":
        from models.deepsif import DeepSIFpl

        return DeepSIFpl(
            num_sensor=n_sensors,
            num_source=n_sources,
            temporal_input_size=args.deepsif_temporal_input_size,
            optimizer=None,
            lr=1e-3,
            criterion=None,
        )
    raise SystemExit(f"unknown model '{name}' (VIT, 1DCNN, LSTM, DEEPSIF)")


def _git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MT_DIR, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=MT_DIR, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(df):
    out = {"n": int(len(df))}
    for c in ["LE_mm", "AUC", "nMSE", "PSNR", "time_err_ms"]:
        out[c] = {"mean": float(df[c].mean()), "std": float(df[c].std())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("simu_name")
    p.add_argument("-root_simu", required=True, help="repo root (with simulation/<subject>/...) or the subject folder")
    p.add_argument("-ckpt", required=True, help="model weights (.pt or Lightning .ckpt)")
    p.add_argument("-model", default="VIT")
    p.add_argument("-out", required=True, help="output folder, e.g. model_training/results/ablation/<test>/<run>")
    p.add_argument("-leadfield_mat", default=str(MT_DIR.parent / "anatomy" / "leadfield_75_20k.mat"))
    p.add_argument("-source_space", default="fsav_994")
    p.add_argument("-electrode_montage", default="standard_1020")
    p.add_argument("-orientation", default="constrained")
    p.add_argument("-subject_name", default="fsaverage")
    p.add_argument("-to_load", type=int, default=100)
    p.add_argument("-eeg_snr", type=int, default=5)
    p.add_argument("-loss", default="cosine", help="training loss of the model (selects the rescaling)")
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("-device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("-inter_layer", type=int, default=4096, help="1DCNN hidden channels (main_train.py default)")
    p.add_argument("-kernel_size", type=int, default=5, help="1DCNN kernel size (main_train.py default)")
    p.add_argument("-deepsif_temporal_input_size", type=int, default=500, help="DeepSIF hidden size (main_train.py default)")
    p.add_argument("-vit_embed_dim", type=int, default=256)
    p.add_argument("-vit_depth", type=int, default=6)
    p.add_argument("-vit_heads", type=int, default=8)
    p.add_argument("-vit_mlp_dim", type=int, default=512)
    p.add_argument("-vit_dropout", type=float, default=0.1)
    p.add_argument("-legacy_order", action="store_true", help="mimic eval.py's RNG order (parity check only)")
    p.add_argument(
        "-heldout", action="store_true",
        help="score only the samples main_train.py held out for -to_load/-per_valid (thesis: train8k_s0, 8000, 0.25)",
    )
    p.add_argument("-per_valid", type=float, default=1.0, help="validation fraction for -heldout / -legacy_order")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("HARD STOP: -device cuda requested but CUDA is not available (no silent CPU fallback).")

    if args.legacy_order:
        from pytorch_lightning import seed_everything

        seed_everything(0)

    ctx = load_context(
        args.root_simu, args.simu_name, args.to_load, args.eeg_snr, args.leadfield_mat,
        args.source_space, args.electrode_montage, args.orientation, args.subject_name,
    )

    if args.heldout and args.legacy_order:
        raise SystemExit("use either -heldout or -legacy_order, not both")
    indices = None
    if args.heldout:
        if not 0 < args.per_valid < 1:
            raise SystemExit("-heldout needs the training -per_valid (e.g. 0.25), not 1.0")
        indices = heldout_indices(len(ctx.dataset), args.per_valid)
    if args.legacy_order:
        from torch.utils.data import random_split

        _, val_ds = random_split(ctx.dataset, [1 - args.per_valid, args.per_valid])
        indices = list(val_ds.indices)

    n_times = ctx.config["rec_info"]["n_times"]
    model = build_model(args.model, args, ctx.fwd.shape[0], ctx.fwd.shape[1], n_times)
    load_module_weights(model, args.ckpt)

    n_eval = len(indices) if indices is not None else len(ctx.dataset)
    print(f"Evaluating {args.model} on {ctx.simu_path} ({n_eval} of {len(ctx.dataset)} samples, device={args.device})")
    df = evaluate(
        model, ctx, loss=args.loss, seed=args.seed, device=args.device,
        indices=indices, reseed=not args.legacy_order,
    )

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "per_sample.csv"), index=False)
    commit, dirty = _git_info()
    summary = {
        "metrics": summarize(df),
        "model": args.model,
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_sha256": _sha256(args.ckpt),
        "dataset": ctx.simu_path,
        "eeg_snr": args.eeg_snr,
        "seed": args.seed,
        "legacy_order": args.legacy_order,
        "heldout": args.heldout,
        "n_evaluated": int(len(df)),
        "device": args.device,
        "torch": torch.__version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    m = summary["metrics"]
    print(
        f"LE {m['LE_mm']['mean']:.2f} mm | AUC {m['AUC']['mean']:.3f} | nMSE {m['nMSE']['mean']:.4f} | "
        f"PSNR {m['PSNR']['mean']:.2f} dB | time err {m['time_err_ms']['mean']:.1f} ms  ->  {args.out}"
    )


if __name__ == "__main__":
    main()
