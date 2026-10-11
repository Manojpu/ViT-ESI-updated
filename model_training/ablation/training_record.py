"""Record training provenance without consuming random numbers or changing training."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_manifest(path, record):
    record["updated_utc"] = datetime.now(timezone.utc).isoformat()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_manifest(args, repo_root, simu_path, train_ds, val_ds, model, lr, device, gc_val):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
    ids = getattr(train_ds.dataset, "ids", list(range(len(train_ds.dataset))))
    return {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args), "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()), "git_commit": commit, "git_dirty": dirty,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "pytorch_lightning": version("pytorch-lightning"),
            "cuda_runtime": torch.version.cuda, "device": device.type,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "requirements_lock_sha256": sha256(repo_root / "model_training/ablation/requirements-lock.txt"),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "dataset": str(Path(simu_path).resolve()), "seed": 0,
        "train_indices": list(train_ds.indices), "heldout_indices": list(val_ds.indices),
        "train_ids": [ids[i] for i in train_ds.indices],
        "heldout_ids": [ids[i] for i in val_ds.indices],
        "heldout_sample_ids": [train_ds.dataset.md_dict[ids[i]].get("id", ids[i]) for i in val_ds.indices]
            if hasattr(train_ds.dataset, "md_dict") else None,
        "optimizer": "Adam", "learning_rate": lr,
        "noise_type": {"white": 1.0, "pink": 0.0},
        "early_stopping_enabled": not args.no_early_stop,
        "gradient_clip_val": gc_val,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoints": {},
    }


def export_checkpoint(callback, model_class, parameters, export_path):
    if not callback.best_model_path:
        raise RuntimeError(f"No checkpoint saved for {callback.monitor}")
    # These are locally generated, trusted Lightning checkpoints.
    checkpoint = torch.load(callback.best_model_path, map_location="cpu", weights_only=False)
    model = model_class.load_from_checkpoint(
        checkpoint_path=callback.best_model_path, map_location="cpu", **parameters,
    )
    torch.save(model.state_dict(), export_path)
    return {
        "monitor": callback.monitor, "epoch": int(checkpoint["epoch"]),
        "score": float(callback.best_model_score),
        "checkpoint": str(Path(callback.best_model_path).resolve()),
        "checkpoint_sha256": sha256(callback.best_model_path),
        "weights": str(Path(export_path).resolve()), "weights_sha256": sha256(export_path),
    }
