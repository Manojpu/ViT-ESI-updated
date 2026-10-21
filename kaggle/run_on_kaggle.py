"""
Run one training job inside a Kaggle notebook, end to end:

  environment check -> regenerate frozen data -> fingerprint check -> train -> evaluate -> package

The notebook clones this repo at a pinned commit and calls this script; everything lands in
/kaggle/working/out (the notebook's saved output). Data and the repo live in /tmp, so the
~11 GB of simulations don't count against Kaggle's 20 GB output limit.

  python kaggle/run_on_kaggle.py --job smoke --model VIT
  python kaggle/run_on_kaggle.py --job full  --model VIT --train-args "-vit_depth 4"

smoke: 100 seed-0 training samples + 100 test2k_s1 samples, 2 epochs (checks the plumbing only).
full : the thesis protocol from #4 (8,000 loaded, 6,000 / 2,000 split, 100 epochs, batch 8, 5 dB,
       validation-loss checkpoint), scored on the held-out 2,000 and on test2k_s1.
"""

import argparse
import datetime
import glob
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Kaggle's "T4 x2" has two GPUs and Lightning's devices="auto" would train with DDP on both,
# doubling the effective batch (2 x 8). Pin every stage to one GPU, as in the local #4 runs.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT = Path(__file__).resolve().parents[1]
SIMU = ROOT / "simulation/fsaverage/constrained/standard_1020/fsav_994/simu"
FINGERPRINTS = ROOT / "data_generation/fingerprints"

# Exactly the generator settings in data_generation/FROZEN_DATASETS.md.
DATASETS = {
    "train8k_s0": {"n": 8000, "smoke_n": 100, "args": ["--seed", "0"]},
    "test2k_s1": {"n": 2000, "smoke_n": 100, "args": ["--seed", "1"]},
    "widepeak2k_s2": {"n": 2000, "smoke_n": 100, "args": ["-c", "425", "-interdev", "0.5", "0.765", "0.02", "--seed", "2"]},
}
# requirements.txt minus torch/torchvision/torchaudio, which come with the Kaggle image (recorded in the manifest).
EXTRA_PACKAGES = ["pytorch-lightning>=2.2,<3", "mne>=1.6,<2", "colorednoise>=2.2,<3"]


class Job:
    def __init__(self, out):
        self.out = Path(out)
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        self.manifest = {"started_utc": _now(), "stages": []}

    def save(self):
        with open(self.out / "job_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

    def run(self, stage, command, cwd=ROOT):
        print(f"\n=== {stage}: {' '.join(shlex.quote(str(c)) for c in command)}", flush=True)
        record = {"stage": stage, "command": [str(c) for c in command], "started_utc": _now()}
        self.manifest["stages"].append(record)
        self.save()
        log_path = self.out / "logs" / f"{stage}.log"
        with open(log_path, "w") as log:
            proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
        record.update(finished_utc=_now(), returncode=proc.returncode)
        self.save()
        if proc.returncode != 0:
            raise SystemExit(f"HARD STOP: stage '{stage}' failed (exit {proc.returncode}); see {log_path}")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def environment(job):
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("HARD STOP: no CUDA GPU in this session (enable a GPU accelerator); no CPU fallback.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    job.manifest["environment"] = {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_used": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout,
    }
    job.save()
    if dirty:
        raise SystemExit("HARD STOP: repo checkout is dirty; runs must come from a clean pinned commit.")


def install(job):
    job.run("install", [sys.executable, "-m", "pip", "install", "-q", *EXTRA_PACKAGES])
    # Every requirements.txt constraint must hold in the final environment.
    from importlib.metadata import version, PackageNotFoundError
    from packaging.requirements import Requirement

    bad = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        req = Requirement(line)
        try:
            v = version(req.name)
        except PackageNotFoundError:
            bad.append(f"{req.name} missing")
            continue
        if req.specifier and v not in req.specifier:
            bad.append(f"{req.name} {v} not in {req.specifier}")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (job.out / "pip_freeze.txt").write_text(freeze)
    job.manifest["requirements_problems"] = bad
    job.save()
    if bad:
        raise SystemExit(f"HARD STOP: requirements not satisfied: {bad}")


def generate(job, name, n):
    spec = DATASETS[name]
    job.run(f"generate_{name}", [
        sys.executable, "data_generation/sereega/simu_extended_source.py", "-sin", name, "-ne", str(n),
        "-mk", "standard_1020", "-ss", "fsav_994", "-o", "constrained", "-sn", "fsaverage", "-rf", str(ROOT),
        "--leadfield_mat", str(ROOT / "anatomy/leadfield_75_20k.mat"), "-fs", "500", "-d", "1000",
        "-af", str(ROOT / "anatomy"), *spec["args"],
    ])
    command = [sys.executable, "data_generation/fingerprint_dataset.py", "check", str(SIMU / name),
               "-ref", str(FINGERPRINTS / f"{name}.json.gz")]
    if n != spec["n"]:
        command += ["-n", str(n)]
    job.run(f"fingerprint_{name}", command)


def train(job, model, n_load, epochs, extra):
    results = ROOT / "model_training/results/kaggle"
    job.run(f"train_{model}", [
        sys.executable, "main_train.py", "train8k_s0", "-results_path", str(results),
        "-simu_type", "sereega", "-source_space", "fsav_994", "-electrode_montage", "standard_1020",
        "-orientation", "constrained", "-model", model, "-loss", "cosine", "-scaler", "linear",
        "-eeg_snr", "5", "-n_times", "500", "-per_valid", "0.25", "-to_load", str(n_load),
        "-n_epochs", str(epochs), "-batch_size", "8", "-require_cuda",
        "-checkpoint_monitor", "validation_loss", "-save_comparison_checkpoint",
        "-leadfield_mat", str(ROOT / "anatomy/leadfield_75_20k.mat"), "-simu_folder", str(SIMU / "train8k_s0"),
        *extra,
    ], cwd=ROOT / "model_training")
    if "Initializing distributed" in (job.out / "logs" / f"train_{model}.log").read_text():
        raise SystemExit("HARD STOP: training ran distributed (more than one GPU); the protocol is single-GPU batch 8.")
    found =glob.glob(str(results / "**" / "trained_models" / f"{model}_model.pt"), recursive=True)
    if len(found) != 1:
        raise SystemExit(f"HARD STOP: expected one exported {model}_model.pt, found {found}")
    return Path(found[0])


def evaluate(job, model, ckpt, name, n, heldout, model_args):
    tag = f"{name}_heldout" if heldout else name
    command = [sys.executable, "model_training/ablation/evaluate.py", name, "-root_simu", str(ROOT),
               "-ckpt", str(ckpt), "-model", model, "-to_load", str(n), "-device", "cuda",
               "-eeg_snr", "5", "-seed", "0", "-out", str(job.out / "eval" / tag), *model_args]
    if heldout:
        command += ["-heldout", "-per_valid", "0.25"]
    job.run(f"eval_{tag}", command)
    with open(job.out / "eval" / tag / "summary.json") as f:
        metrics = json.load(f)["metrics"]
    return {"n": metrics["n"], **{k: v["mean"] for k, v in metrics.items() if isinstance(v, dict)}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", choices=["smoke", "full"], required=True)
    p.add_argument("--model", default="VIT", choices=["VIT", "1DCNN", "LSTM", "DEEPSIF"])
    p.add_argument("--train-args", default="", help="extra main_train.py flags, e.g. '-vit_depth 4'")
    p.add_argument("--eval-args", default="", help="matching evaluate.py flags, e.g. '-vit_depth 4'")
    p.add_argument("--widepeak", action="store_true", help="also regenerate widepeak2k_s2 and score it (A3)")
    p.add_argument("--out", default="/kaggle/working/out")
    args = p.parse_args()

    smoke = args.job == "smoke"
    job = Job(args.out)
    job.manifest["args"] = vars(args)
    job.save()
    environment(job)
    install(job)

    sizes = {name: (spec["smoke_n"] if smoke else spec["n"]) for name, spec in DATASETS.items()}
    names = ["train8k_s0", "test2k_s1"] + (["widepeak2k_s2"] if args.widepeak else [])
    for name in names:
        generate(job, name, sizes[name])
    job.run("build_model_folder", [sys.executable, "model_training/ablation/build_model_folder.py", "-root", "."])

    ckpt = train(job, args.model, sizes["train8k_s0"], 2 if smoke else 100, shlex.split(args.train_args))
    model_args = shlex.split(args.eval_args)
    results = {
        "heldout": evaluate(job, args.model, ckpt, "train8k_s0", sizes["train8k_s0"], True, model_args),
        "test2k_s1": evaluate(job, args.model, ckpt, "test2k_s1", sizes["test2k_s1"], False, model_args),
    }
    if args.widepeak:
        results["widepeak2k_s2"] = evaluate(job, args.model, ckpt, "widepeak2k_s2", sizes["widepeak2k_s2"], False, model_args)

    run_dir = ckpt.parents[1]
    shutil.copytree(run_dir, job.out / "training", dirs_exist_ok=True)
    job.manifest.update(results=results, checkpoint=str(ckpt.relative_to(run_dir)), finished_utc=_now(), status="passed")
    job.save()
    print("\nRESULTS", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
