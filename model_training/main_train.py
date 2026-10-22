import os
import argparse
import time
import sys

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, tensorboard

from pytorch_lightning import seed_everything

from loaders import ModSpikeEEGBuild, EsiDatasetds_new
from scipy.io import loadmat
from utils.utl import CosineSimilarityLoss, logMSE
from load_data.FolderStructure import FolderStructure
from load_data import HeadModel
import json
from pathlib import Path
from ablation.training_record import make_manifest, save_manifest, export_checkpoint

# Training on GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")


# seed
seed_everything(0)

home = os.path.expanduser("~")
# ------------------------ ARGPARSE -----------------------------------------------#

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

#argument to load the data
parser.add_argument("simu_name", type=str, help="name of the simulation")
parser.add_argument("-root_simu", type=str, default=None, help="path to the folder containing data (required if -simu_folder is not set)")
parser.add_argument("-simu_folder", type=str, default=None, help="optional: full path to the simulation folder (e.g. .../constrained/standard_1020/fsav_994/simu/mes_debug_python); when set, used directly and root_simu is ignored for loading data")
parser.add_argument("-results_path", type=str, required=True, help="where to save results")

parser.add_argument("-orientation", type=str, default="constrained", help="constrained or unconstrained, orientation of the sources")
parser.add_argument("-electrode_montage", type=str, default="standard_1020", help="name of the electrode montage to use")
parser.add_argument("-source_space", type=str, default="ico3", help="name of the source space")
parser.add_argument(
    "-subject_name",
    type=str,
    default="fsaverage",
    help="Subject name when using `simulation/<subject>/...` folder layout",
)
parser.add_argument(
    "-data_layout",
    type=str,
    default="auto",
    choices=["auto", "flat", "simulation"],
    help=(
        "Where to load simulations from. "
        "'flat' expects <root>/<ori>/<montage>/<src>/simu/<simu_name>. "
        "'simulation' expects <root>/simulation/<subject>/<ori>/<montage>/<src>/simu/<simu_name>. "
        "'auto' picks 'simulation' if that folder exists, else 'flat'."
    ),
)
parser.add_argument(
    "-simu_type", type=str, help="type of simulation used (NMM or SEREEGA)"
)
parser.add_argument("-spikes_folder", type=str, default="nmm_spikes_nov23", help="folder with spikes for NMM based simulations")

parser.add_argument(
    "-n_times", type=int, default=500, help="number of time samples in the signal"
)
parser.add_argument(
    "-to_load",
    default=100,
    type=int,
    help="number of samples to load in the train+val dataset",
)
parser.add_argument(
    "-per_valid",
    default=0.2,
    type=float,
    help="fraction of the dataset to use for validation",
)
parser.add_argument(
    "-eeg_snr",
    default=5,
    type=int,
    help="SNR of the EEG data (additive white gaussian noise)",
)
parser.add_argument(
    "-model",
    default="1dcnn",
    type=str,
    help="name of the model to use (neural network",
)
parser.add_argument(
    "-inter_layer",
    type=int,
    default=4096,
    help="number of channels of the hidden layer of the 1D CNN",
)
parser.add_argument(
    "-kernel_size", type=int, default=5, help="kernel size of the 1D CNN"
)
parser.add_argument(
    "-lstm_hidden",
    type=int,
    default=85,
    help="hidden size of each LSTM direction (85 = thesis baseline)",
)
parser.add_argument(
    "-deepsif_temporal_input_size",
    type=int,
    default=500,
    help="DeepSIF hidden size (called temporal_input_size in the original implementation)",
)
parser.add_argument("-vit_embed_dim", type=int, default=256, help="EEGViT embedding dimension")
parser.add_argument("-vit_depth", type=int, default=6, help="EEGViT number of Transformer layers")
parser.add_argument("-vit_heads", type=int, default=8, help="EEGViT number of attention heads")
parser.add_argument("-vit_mlp_dim", type=int, default=512, help="EEGViT feedforward dimension")
parser.add_argument("-vit_dropout", type=float, default=0.1, help="EEGViT dropout")
parser.add_argument(
    "-leadfield_mat",
    type=str,
    default=None,
    help="Optional path to a .mat leadfield to use (overrides model leadfield).",
)

parser.add_argument(
    "-n_epochs", "--ep", default=25, type=int, help="number of epochs for training"
)
parser.add_argument(
    "-no_early_stop", action="store_false", help="do not use early stopping"
)
parser.add_argument("-batch_size", "--bs", default=8, type=int, help="batch size")
parser.add_argument(
    "-scaler",
    default="linear",
    type=str,
    help="type of normalisation to use (max or linear)",
)
parser.add_argument(
    "-loss", default="cosine", type=str, help="type of loss function to use"
)
parser.add_argument(
    "-sfolder",
    default="trainings",
    type=str,
    help="name of the subfolder in which to save results",
)
parser.add_argument(
    "-resume",
    type=str,
    default=None,
    metavar="PATH",
    help="Path to a saved checkpoint (.ckpt) to resume training from (optimizer, epoch, and model state are restored).",
)

parser.add_argument(
    "-checkpoint_monitor", choices=["train_loss", "validation_loss"], default="train_loss",
    help="loss minimized when selecting the primary exported model (legacy default: train_loss)",
)
parser.add_argument(
    "-save_comparison_checkpoint", action="store_true",
    help="also retain and export the best checkpoint under the other loss criterion",
)
parser.add_argument("-require_cuda", action="store_true", help="hard stop instead of CPU fallback")
parser.add_argument(
    "-deterministic", action="store_true",
    help="require deterministic operations for verification (not used by the full baseline protocol)",
)
args = parser.parse_args()
if args.require_cuda and device.type != "cuda":
    raise SystemExit("HARD STOP: CUDA is required; no CPU fallback.")
if args.deterministic:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------ where to save results -------- #
results_path = f"{args.results_path}/{args.simu_name}{args.source_space}_/{args.sfolder}"
os.makedirs(f"{results_path}/{args.sfolder}", exist_ok=True)
## -------------------------------------- LOAD DATA ----------------------------------------------- ##
if args.simu_folder:
    # Use simulation folder path directly: .../orientation/montage/source_space/simu/simu_name
    simu_path = str(Path(args.simu_folder).resolve())
    _p = Path(simu_path)
    root_base = _p.parent.parent.parent.parent.parent  # simu_name -> simu -> source_space -> montage -> orientation -> root
    model_path = str(_p.parent.parent / "model")  # source_space/model
else:
    if not args.root_simu:
        parser.error("either -root_simu or -simu_folder must be provided")
    root_simu = args.root_simu
    root_simu_path = Path(root_simu)
    # Support both layouts:
    #   A) <root>/<ori>/<montage>/<src>/simu/<simu_name>
    #   B) <root>/simulation/<subject>/<ori>/<montage>/<src>/simu/<simu_name>
    if args.data_layout == "flat":
        root_base = root_simu_path
    elif args.data_layout == "simulation":
        root_base = root_simu_path / "simulation" / args.subject_name
    else:  # auto
        if (root_simu_path / "simulation" / args.subject_name).is_dir():
            root_base = root_simu_path / "simulation" / args.subject_name
        else:
            root_base = root_simu_path

    simu_path = str(
        root_base
        / args.orientation
        / args.electrode_montage
        / args.source_space
        / "simu"
        / args.simu_name
    )
    model_path = str(
        root_base / args.orientation / args.electrode_montage / args.source_space / "model"
    )


# config_file = f"{simu_path}/{args.simu_name}{args.source_space}_config.json"

# with open(config_file, "r") as f:
#     general_config_dict = json.load(f)

general_config_dict = {
    "eeg_snr": args.eeg_snr,
    "simu_name": args.simu_name,
    "source_space": {
        "constrained_orientation": args.orientation == "constrained",
        "src_sampling": args.source_space,
    },
    "electrode_space": {
        "electrode_montage": args.electrode_montage,
    },
}
folders = FolderStructure(str(root_base), general_config_dict)
source_space_obj = HeadModel.SourceSpace(folders, general_config_dict)

def _load_leadfield_mat(mat_path: str):
    m = loadmat(mat_path)
    if "G" in m:
        return m["G"]
    if "fwd" in m:
        return m["fwd"]
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and getattr(v, "ndim", 0) == 2:
            return v
    raise KeyError(f"No leadfield matrix found in {mat_path}. Keys={list(m.keys())}")


# --------------------------- Leadfield loading --------------------------- #
repo_root = Path(__file__).resolve().parents[1]
repo_default_fsav994_lf = repo_root / "anatomy" / "leadfield_75_20k.mat"

if args.leadfield_mat:
    fwd = _load_leadfield_mat(args.leadfield_mat)
elif args.source_space == "fsav_994" and os.path.isfile(f"{model_path}/LF_fsav_994.mat"):
    fwd = loadmat(f"{model_path}/LF_fsav_994.mat")["G"]
elif args.source_space == "fsav_994" and repo_default_fsav994_lf.is_file():
    # Common lightweight setup: use the repo-provided 75x994 leadfield without requiring a full model folder.
    fwd = _load_leadfield_mat(str(repo_default_fsav994_lf))
else:
    electrode_space_obj = HeadModel.ElectrodeSpace(folders, general_config_dict)
    head_model = HeadModel.HeadModel(
        electrode_space_obj, source_space_obj, folders, subject_name=args.subject_name
    )
    fwd = head_model.fwd["sol"]["data"]

print("fwd.shape:", fwd.shape)

############################### LOAD DATA ################################
if args.simu_type.upper() == "NMM":
    if args.source_space != "fsav_994":
        sys.exit(
            "NMM spike simulations require the 994-region source space. "
            "Please run with `-source_space fsav_994` (and a matching leadfield)."
        )
    spikes_data_path = f"{str(root_base)}/{args.orientation}/{args.electrode_montage}/{args.source_space}/simu/{args.spikes_folder}"
    dataset_meta_path = f"{simu_path}/{args.simu_name}.mat"

    ds_dataset = ModSpikeEEGBuild(
        spike_data_path=spikes_data_path,
        metadata_file=dataset_meta_path,
        fwd=fwd,
        n_times=args.n_times,
        args_params={"dataset_len": args.to_load},
        spos=source_space_obj.positions,
        norm=args.scaler,
    )

elif args.simu_type.upper() == "SEREEGA":
    # simu_data_path = f"{home}/Documents/Data/simulation"
    # config_file = f"{simu_data_path}/{args.simu_name}{args.source_space}_config.json"
    
    config_file = f"{simu_path}/{args.simu_name}{args.source_space}_config.json"

    ds_dataset = EsiDatasetds_new(
        str(root_base),
        config_file,
        args.simu_name,
        args.source_space,
        "standard_1020",
        args.to_load,
        args.eeg_snr,
        noise_type={"white": 1.0, "pink": 0.0},
        norm=args.scaler,
    )

else:
    sys.exit("unknown simulation type (argument simu_type)")


effective_len = len(ds_dataset)
if effective_len <= 0:
    sys.exit("Dataset is empty (no samples found). Check your simulation folder and match JSON.")
n_train = int(effective_len * (1 - args.per_valid))
n_val = effective_len - n_train
train_ds, val_ds = random_split(ds_dataset, [n_train, n_val])
train_dataloader = DataLoader(dataset=train_ds, batch_size=args.bs, shuffle=True)
val_dataloader = DataLoader(dataset=val_ds, batch_size=args.bs, shuffle=False)

n_electrodes = fwd.shape[0]
n_sources = fwd.shape[1]

## ------------------------------------------- NETWORK TO LOAD --------------------------##
# loss function
if args.loss == "cosine":
    crit = CosineSimilarityLoss()
elif args.loss.upper() == "mse":
    crit = F.mse_loss()
elif args.loss.upper() == "logmse":
    crit = logMSE()
else:
    sys.exit("unknown loss function")

# ---------- CNN 1D ---------#
if args.model.upper() == "1DCNN":
    from models.cnn_1d import CNN1Dpl as net

    lr = 1e-3
    net_parameters = {
        "channels": [
            n_electrodes,
            args.inter_layer,
            n_sources,
        ],
        "kernel_size": args.kernel_size,
        "bias": False,
        "optimizer": torch.optim.Adam,
        "lr": lr,
        "criterion": crit,
    }
    model = net(**net_parameters)

##------------- LSTM ---------------------##
elif args.model.upper() == "LSTM":
    from models.lstm import HeckerLSTMpl as net

    lr = 1e-3
    net_parameters = {
        "n_electrodes": n_electrodes,
        "hidden_size": args.lstm_hidden,
        "n_sources": n_sources,
        "bias": False,
        "optimizer": torch.optim.Adam,
        "lr": lr,
        "criterion": crit,  # nn.MSELoss(reduction='sum'), #CosineSimilarityLoss(),
        "mc_dropout_rate": 0,
    }
    model = net(**net_parameters)

##------------------ DEEPSIF ----------------##
elif args.model.upper() == "DEEPSIF":
    from models.deepsif import DeepSIFpl as net

    lr = 1e-3
    net_parameters = {
        "num_sensor": n_electrodes,
        "num_source": n_sources,
        "temporal_input_size": args.deepsif_temporal_input_size,
        "optimizer": torch.optim.Adam,
        "lr": lr,
        "criterion": crit,
    }
    model = net(**net_parameters)

##------------------ EEGViT (Transformer) ----------------##
elif args.model.upper() in ("VIT", "EEGVIT", "TRANSFORMER"):
    from models.vit import EEGViTpl as net

    lr = 1e-3
    net_parameters = {
        "num_sensor": n_electrodes,
        "num_source": n_sources,
        "n_times": args.n_times,
        "embed_dim": args.vit_embed_dim,
        "depth": args.vit_depth,
        "num_heads": args.vit_heads,
        "mlp_dim": args.vit_mlp_dim,
        "dropout": args.vit_dropout,
        "optimizer": torch.optim.Adam,
        "lr": lr,
        "criterion": crit,
    }
    model = net(**net_parameters)

else:
    sys.exit("unknown model")

##------------------- TRAINING ----------------------------##
n_train_samples = len(train_ds)

if args.model.upper() == "1DCNN":
    subfolder = (
        f"simu_{args.simu_type}_"
        f"srcspace_{args.source_space}"
        f"_model_{args.model}"
        f"_interlayer_{args.inter_layer}"
        f"_trainset_{n_train_samples}"
        f"_epochs_{args.ep}"
        f"_loss_{args.loss}"
        f"_norm_{args.scaler}"
    ) 
else:
    subfolder = (
        f"simu_{args.simu_type}_"
        f"srcspace_{args.source_space}"
        f"_model_{args.model}"
        f"_trainset_{n_train_samples}"
        f"_epochs_{args.ep}"
        f"_loss_{args.loss}"
        f"_norm_{args.scaler}"
    )

results_path = f"{results_path}/{subfolder}"
os.makedirs(f"{results_path}/trained_models", exist_ok=True)
best_model_path = f"{results_path}/trained_models/{args.model}_model.pt"

print(f"### best model path: {best_model_path} ###")

# --------------------- TRAINING ------------------------#
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{results_path}/pl_checkpoints/{args.checkpoint_monitor}",
    filename="{epoch}-{" + args.checkpoint_monitor + ":.6f}",
    monitor=args.checkpoint_monitor,
    mode="min",
    save_top_k=1,
)
checkpoint_callbacks = [checkpoint_callback]
if args.save_comparison_checkpoint:
    comparison_monitor = "train_loss" if args.checkpoint_monitor == "validation_loss" else "validation_loss"
    checkpoint_callbacks.append(ModelCheckpoint(
        dirpath=f"{results_path}/pl_checkpoints/{comparison_monitor}",
        filename="{epoch}-{" + comparison_monitor + ":.6f}",
        monitor=comparison_monitor,
        mode="min",
        save_top_k=1,
    ))

try:
    logger = TensorBoardLogger(save_dir=f"{results_path}/logs/")
except ModuleNotFoundError:
    # TensorBoard is optional; fall back to CSV logging to avoid crashing.
    from pytorch_lightning.loggers import CSVLogger

    logger = CSVLogger(save_dir=f"{results_path}/logs/")

# gradient clipping
if args.model.upper() == "LSTM":
    gc_val = 1  # gradient are clipped to a value of 1 for the LSTM
else:
    gc_val = 0

# early stopping strategy
if args.no_early_stop:
    cbs = list(checkpoint_callbacks)
else:
    early_stop_cb = EarlyStopping(monitor="validation_loss", min_delta=0.0, patience=20)
    cbs = [*checkpoint_callbacks, early_stop_cb]

# trainer
trainer = pl.Trainer(
    accelerator=device.type,
    max_epochs=args.ep,
    logger=logger,
    callbacks=cbs,
    log_every_n_steps=1, ## pas ce que je veux non?
    gradient_clip_val=gc_val,
    deterministic=True if args.deterministic else None,
)

print(
    f"<<<<<<<<<<<<<<<<<<<<<<<<<<< training model {args.model.upper()} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
)
if args.resume:
    print(f"Resuming from checkpoint: {args.resume}")
manifest = make_manifest(args, repo_root, simu_path, train_ds, val_ds, model, lr, device, gc_val)
manifest_path = Path(results_path) / "run_manifest.json"
save_manifest(manifest_path, manifest)
start_t = time.time()
try:
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume,
    )
    end_t = time.time()
    manifest["duration_seconds"] = end_t - start_t
    manifest["completed_epochs"] = int(trainer.current_epoch)
    for callback in checkpoint_callbacks:
        export_path = (best_model_path if callback is checkpoint_callback else
                       f"{results_path}/trained_models/{args.model}_model_{callback.monitor}.pt")
        manifest["checkpoints"][callback.monitor] = export_checkpoint(
            callback, net, net_parameters, export_path,
        )
    manifest["status"] = "completed"
    save_manifest(manifest_path, manifest)
except BaseException as error:
    manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    manifest["duration_seconds"] = time.time() - start_t
    manifest["error"] = f"{type(error).__name__}: {error}"
    save_manifest(manifest_path, manifest)
    raise
print(f"Training time : {(end_t-start_t)}")

with open(f'{results_path}/training_times.txt','a') as f :
    f.write(
        f"model : {subfolder} - training time : {(end_t-start_t):0.3f}s\n"
    ) 
    f.write(
        "-------------------------------------------------------------\n"
    )
