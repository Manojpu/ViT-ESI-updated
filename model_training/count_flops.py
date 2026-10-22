"""
FLOPs and parameter count for all EEG source-imaging model architectures.

Models covered
--------------
  1D CNN   -- simple_1dCNN_v2  (wrapped by CNN1Dpl)
  LSTM     -- HeckerLSTM       (wrapped by HeckerLSTMpl)
  DeepSIF  -- TemporalInverseNet (wrapped by DeepSIFpl)
  EEGViT   -- EEGViT           (wrapped by EEGViTpl)

FLOPs backend preference (first found wins)
-------------------------------------------
  1. thop        -- pip install thop
  2. torchinfo   -- pip install torchinfo
  3. fvcore      -- pip install fvcore
  (parameter count is always reported regardless of backend)

All counts are for a single sample forward pass (batch_size = 1).
FLOPs ≈ 2 × MACs (Multiply-Accumulate Operations).

Usage
-----
  cd model_training
  python count_flops.py

  # Override default problem dimensions:
  python count_flops.py --n_electrodes 75 --n_sources 994 --n_times 500
"""

from __future__ import annotations

import argparse
import importlib
import sys
import os

# Make model_training importable regardless of where the script is called from
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def human_readable(n: int | float) -> str:
    """Return a human-readable string like '12.34 M'."""
    for unit, threshold in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= threshold:
            return f"{n / threshold:.2f} {unit}"
    return str(int(n))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# FLOPs backends
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend() -> str | None:
    for name in ("thop", "torchinfo", "fvcore"):
        try:
            importlib.import_module(name)
            return name
        except ImportError:
            continue
    return None


def _flops_thop(model: nn.Module, dummy: torch.Tensor) -> tuple[int, int]:
    """Returns (flops, macs) using thop."""
    from thop import profile  # type: ignore
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    return int(macs * 2), int(macs)


def _flops_torchinfo(model: nn.Module, dummy: torch.Tensor) -> tuple[int, int]:
    """Returns (flops, macs) using torchinfo."""
    from torchinfo import summary  # type: ignore
    result = summary(model, input_data=dummy, verbose=0)
    macs = int(result.total_mult_adds)
    return macs * 2, macs


def _flops_fvcore(model: nn.Module, dummy: torch.Tensor) -> tuple[int, int]:
    """Returns (flops, macs) using fvcore."""
    from fvcore.nn import FlopCountAnalysis  # type: ignore
    flops_analysis = FlopCountAnalysis(model, dummy)
    flops_analysis.unsupported_ops_warnings(False)
    flops_analysis.uncalled_modules_warnings(False)
    flops = int(flops_analysis.total())
    return flops, flops // 2


_BACKEND_FN = {
    "thop":      _flops_thop,
    "torchinfo": _flops_torchinfo,
    "fvcore":    _flops_fvcore,
}


def compute_flops(
    model: nn.Module,
    dummy: torch.Tensor,
    backend: str | None,
) -> tuple[int | None, int | None]:
    """Return (flops, macs) or (None, None) if no backend is available."""
    if backend is None:
        return None, None
    return _BACKEND_FN[backend](model, dummy.clone())


# ─────────────────────────────────────────────────────────────────────────────
# DeepSIF wrapper — TemporalInverseNet.forward returns a dict, which can
# confuse some FLOPs counters.  This thin wrapper exposes a plain tensor output.
# ─────────────────────────────────────────────────────────────────────────────

class _DeepSIFWrapper(nn.Module):
    def __init__(self, pl_module):
        super().__init__()
        self.inner = pl_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)  # DeepSIFpl.forward already returns a tensor


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def build_model_configs(n_electrodes: int, n_sources: int, n_times: int,
                        inter_layer: int = 4096, lstm_hidden: int = 85) -> dict:
    return {
        "1D CNN": {
            "module": "models.cnn_1d",
            "class":  "CNN1Dpl",
            "kwargs": dict(
                channels=[n_electrodes, inter_layer, n_sources],
                kernel_size=5,
                bias=False,
            ),
            "wrapper": None,
        },
        "LSTM (Hecker)": {
            "module": "models.lstm",
            "class":  "HeckerLSTMpl",
            "kwargs": dict(
                n_electrodes=n_electrodes,
                hidden_size=lstm_hidden,
                n_sources=n_sources,
                bias=False,
                mc_dropout_rate=0,
            ),
            "wrapper": None,
        },
        "DeepSIF": {
            "module": "models.deepsif",
            "class":  "DeepSIFpl",
            "kwargs": dict(
                num_sensor=n_electrodes,
                num_source=n_sources,
                temporal_input_size=n_times,
                rnn_layer=3,
            ),
            "wrapper": _DeepSIFWrapper,
        },
        "EEGViT (Transformer)": {
            "module": "models.vit",
            "class":  "EEGViTpl",
            "kwargs": dict(
                num_sensor=n_electrodes,
                num_source=n_sources,
                n_times=n_times,
                embed_dim=256,
                depth=6,
                num_heads=8,
                mlp_dim=512,
                dropout=0.1,
            ),
            "wrapper": None,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLOPs/parameter counter for all ESI models.")
    p.add_argument("--n_electrodes", type=int, default=75,
                   help="Number of EEG sensors  (default: 75)")
    p.add_argument("--n_sources",    type=int, default=994,
                   help="Number of source dipoles (default: 994)")
    p.add_argument("--n_times",      type=int, default=500,
                   help="Time-series length in samples (default: 500)")
    p.add_argument("--inter_layer",  type=int, default=4096,
                   help="1D CNN hidden channels (default: 4096)")
    p.add_argument("--lstm_hidden",  type=int, default=85,
                   help="LSTM hidden size per direction (default: 85)")
    p.add_argument("--backend", type=str, default=None,
                   choices=["thop", "torchinfo", "fvcore"],
                   help="Force a specific FLOPs backend (auto-detected if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    n_electrodes = args.n_electrodes
    n_sources    = args.n_sources
    n_times      = args.n_times

    backend = args.backend or _detect_backend()

    dummy = torch.randn(1, n_electrodes, n_times)  # (B=1, E, T)

    print()
    print("=" * 72)
    print("  EEG Source Imaging — Model FLOPs & Parameter Count")
    print("=" * 72)
    print(f"  Input shape  : (B=1, E={n_electrodes}, T={n_times})")
    print(f"  Target shape : (B=1, S={n_sources}, T={n_times})")
    print(f"  FLOPs backend: {backend or 'none — install thop / torchinfo / fvcore'}")
    print("=" * 72)

    col_w = [26, 14, 14, 14]
    header = (
        f"{'Model':<{col_w[0]}}"
        f"{'Parameters':>{col_w[1]}}"
        f"{'FLOPs':>{col_w[2]}}"
        f"{'MACs':>{col_w[3]}}"
    )
    print(header)
    print("-" * 72)

    configs = build_model_configs(n_electrodes, n_sources, n_times,
                                  args.inter_layer, args.lstm_hidden)

    rows: list[dict] = []

    for name, cfg in configs.items():
        mod = importlib.import_module(cfg["module"])
        cls = getattr(mod, cfg["class"])
        pl_model: nn.Module = cls(**cfg["kwargs"])
        pl_model.eval()

        # Optionally wrap (e.g. for DeepSIF dict-output issue)
        model = cfg["wrapper"](pl_model) if cfg["wrapper"] else pl_model

        n_params = count_parameters(pl_model)

        try:
            flops, macs = compute_flops(model, dummy, backend)
            err = None
        except Exception as exc:
            flops = macs = None
            err = str(exc)

        rows.append(dict(name=name, params=n_params, flops=flops, macs=macs))

        params_str = human_readable(n_params)
        flops_str  = human_readable(flops) if flops is not None else "N/A"
        macs_str   = human_readable(macs)  if macs  is not None else "N/A"

        print(
            f"{name:<{col_w[0]}}"
            f"{params_str:>{col_w[1]}}"
            f"{flops_str:>{col_w[2]}}"
            f"{macs_str:>{col_w[3]}}"
        )
        if err:
            print(f"  ⚠  FLOPs error: {err}")

    print("=" * 72)
    print()
    print("Notes")
    print("  • FLOPs ≈ 2 × MACs (each MAC = 1 multiply + 1 add).")
    print("  • LSTM/RNN FLOPs may be underestimated by thop/torchinfo;")
    print("    fvcore gives the most accurate count for recurrent layers.")
    print("  • All counts are for a single sample (batch_size = 1).")
    print("  • Dropout and normalisation layers contribute negligible FLOPs")
    print("    and are excluded by most backends.")
    print()

    # ── detailed per-layer breakdown (torchinfo only) ──────────────────────
    if backend == "torchinfo":
        print("─" * 72)
        print("  Per-layer breakdown (torchinfo)  — EEGViT shown as example")
        print("─" * 72)
        from torchinfo import summary  # type: ignore
        mod = importlib.import_module("models.vit")
        vit_cfg = configs["EEGViT (Transformer)"]
        vit_model = getattr(mod, vit_cfg["class"])(**vit_cfg["kwargs"]).eval()
        summary(vit_model, input_data=dummy, col_names=["input_size", "output_size",
                                                        "num_params", "mult_adds"],
                row_settings=["var_names"], depth=4)


if __name__ == "__main__":
    main()
