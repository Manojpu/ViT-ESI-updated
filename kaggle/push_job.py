"""
Build a Kaggle notebook for one job and (optionally) submit it with the Kaggle CLI.

  python kaggle/push_job.py --job smoke --model VIT --push
  python kaggle/push_job.py --job full --model VIT --title "vitesi calib vit"          # build only, hand over
  python kaggle/push_job.py --job full --model VIT --train-args "-vit_depth 4" --eval-args "-vit_depth 4" --title "a4 depth4"

The notebook clones the public repo at --commit (default: this checkout's HEAD, which must be
pushed and clean) and runs kaggle/run_on_kaggle.py. Without --push it only writes the folder
(notebook + kernel-metadata.json); upload that notebook on kaggle.com, or run
`kaggle kernels push -p <folder>` yourself.

Afterwards:
  kaggle kernels status <user>/<slug>
  kaggle kernels output <user>/<slug> -p <download folder>
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Project-Synapse-AI/ViT-ESI-updated.git"
TEMPLATE = ROOT / "kaggle" / "vitesi_job.ipynb"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _username():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", choices=["smoke", "full"], required=True)
    p.add_argument("--model", default="VIT", choices=["VIT", "1DCNN", "LSTM", "DEEPSIF"])
    p.add_argument("--train-args", default="")
    p.add_argument("--eval-args", default="")
    p.add_argument("--widepeak", action="store_true")
    p.add_argument("--commit", default=None, help="commit to run (default: HEAD, must be on origin)")
    p.add_argument("--title", default=None, help="Kaggle notebook title; the slug is derived from it")
    p.add_argument("--out", default=None, help="folder to write (default: kaggle/build/<slug>, gitignored)")
    p.add_argument("--push", action="store_true", help="submit with `kaggle kernels push`")
    args = p.parse_args()

    commit = args.commit or _git("rev-parse", "HEAD")
    if args.commit is None and _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit or discard tracked changes first: the notebook runs the pushed commit, not this working tree")
    if commit not in _git("branch", "-r", "--contains", commit):
        raise SystemExit(f"commit {commit[:7]} is not on origin yet; push it first")

    title = args.title or f"vitesi {args.job} {args.model.lower()}"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    user = _username()
    out = Path(args.out) if args.out else ROOT / "kaggle" / "build" / slug
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "REPO_URL": REPO_URL,
        "COMMIT": commit,
        "JOB": args.job,
        "MODEL": args.model,
        "TRAIN_ARGS": args.train_args,
        "EVAL_ARGS": args.eval_args,
        "WIDEPEAK": args.widepeak,
    }
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if "# CONFIG" in "".join(cell["source"]):
            cell["source"] = ["# CONFIG (written by kaggle/push_job.py)\n"] + [f"{k} = {v!r}\n" for k, v in config.items()]
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    metadata = {
        "id": f"{user}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"built {out} for {user}/{slug} at {commit[:7]}")

    if args.push:
        kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
        subprocess.run([str(kaggle), "kernels", "push", "-p", str(out)], check=True)
        print(f"submitted: https://www.kaggle.com/code/{user}/{slug}")


if __name__ == "__main__":
    main()
