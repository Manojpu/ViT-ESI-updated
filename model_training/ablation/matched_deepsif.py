"""Issue #4: user-approved DeepSIF comparison while thesis reproduction is unresolved.

Reuse the frozen ViTESI protocol and validation-selected reference. No ViTESI
retraining, tuning, automatic retry/resume, dataset regeneration or CPU fallback.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation import baseline_workflow as workflow
from ablation.training_record import save_manifest, sha256

ROOT, MT = workflow.ROOT, workflow.MT
REFERENCE = MT / "results/ablation/reproduction/checkpoint-selection-20261013-restart/full/workflow.json"
REFERENCE_COMMIT = "84d435c79f0859b649805423256d6e8008cefe20"
REFERENCE_HASH = "0385eca30a2a423639cea9ca5bf53993cf9996ffb3a76235eccfdcd1da186a69"
BEGIN, END = "<!-- matched-deepsif-status -->", "<!-- /matched-deepsif-status -->"
SCIENCE_PATHS = [
    "model_training/main_train.py", "model_training/models", "model_training/loaders.py",
    "model_training/load_data", "model_training/utils", "model_training/ablation/evaluate.py",
    "model_training/ablation/seed.py", "model_training/ablation/training_record.py",
    "model_training/ablation/baseline_workflow.py", "data_generation/sereega",
    "model_training/ablation/requirements-lock.txt",
]


def reference(preflight):
    record = json.loads(REFERENCE.read_text())
    if record["git_commit"] != REFERENCE_COMMIT or record["status"] != "gate_failed":
        raise RuntimeError("Unexpected ViTESI reference run")
    changes = subprocess.check_output(
        ["git", "diff", REFERENCE_COMMIT, "HEAD", "--", *SCIENCE_PATHS], cwd=ROOT,
    )
    if changes:
        raise RuntimeError("Scientific implementation differs from the ViTESI reference")
    if preflight["datasets"] != record["preflight"]["datasets"]:
        raise RuntimeError("Frozen dataset config/index hashes differ from the reference")
    vit = record["training"]["VIT"]
    item = vit["checkpoints"]["validation_loss"]
    if item["weights_sha256"] != REFERENCE_HASH or sha256(item["weights"]) != REFERENCE_HASH:
        raise RuntimeError("ViTESI reference weights do not match the approved hash")
    if vit["status"] != "completed" or vit["completed_epochs"] != 100 or vit["git_dirty"]:
        raise RuntimeError("Incomplete/dirty ViTESI reference")
    environment = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "pytorch_lightning": version("pytorch-lightning"), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "requirements_lock_sha256": sha256(MT / "ablation/requirements-lock.txt"),
    }
    if any(vit["environment"][key] != value for key, value in environment.items()):
        raise RuntimeError("Current environment differs from the ViTESI reference")
    for dataset in ["heldout", "test2k"]:
        folder = REFERENCE.parent / f"eval_VIT_validation_loss_{dataset}"
        summary = json.loads((folder / "summary.json").read_text())
        frame = pd.read_csv(folder / "per_sample.csv")
        if (summary["ckpt_sha256"] != REFERENCE_HASH or summary["git_commit"] != REFERENCE_COMMIT
                or summary["git_dirty"] or summary["device"] != "cuda"
                or summary["eeg_snr"] != 5 or summary["seed"] != 0 or summary["legacy_order"]):
            raise RuntimeError("ViTESI evaluation provenance differs")
        expected = vit["heldout_indices"] if dataset == "heldout" else list(range(2000))
        if frame["index"].tolist() != expected or not np.isfinite(frame[workflow.METRICS]).all().all():
            raise RuntimeError("Invalid ViTESI evaluation sample indices/metrics")
        if dataset == "heldout" and frame["id"].tolist() != vit["heldout_sample_ids"]:
            raise RuntimeError("Invalid ViTESI held-out sample IDs")
        for metric in workflow.METRICS:
            if not np.isclose(frame[metric].mean(), summary["metrics"][metric]["mean"], rtol=1e-12, atol=1e-12):
                raise RuntimeError("ViTESI CSV/summary mismatch")
    return record, vit


def replace_block(body, block):
    body = re.sub(r"\r+\n", "\n", body).replace("\r", "\n")
    if BEGIN in body:
        start = body.index(BEGIN)
        stop = body.index(END, start) + len(END)
        return body[:start] + block + body[stop:]
    return body.rstrip() + "\n\n" + block + "\n"


def publish(record, message):
    block = f"""{BEGIN}
## Matched ViTESI vs DeepSIF — user authorized 2026-10-19

Status: **{record['status']}**. Stage: `{record['current_stage']}`. Commit `{record['git_commit']}`.
Record: `{record['output']}/workflow.json`; inspect stage logs and process IDs before acting.

This additional comparison is explicitly authorized despite the failed thesis gate.
Thesis reproduction remains unresolved; the old failed-gate results are preserved.
Train only DeepSIF from scratch: identical frozen 6000/2000 split, seed 0, 100 full epochs,
batch 8, Adam 0.001, cosine loss, linear scaling, white noise 5 dB, 500 Hz / 1 second,
CUDA and validation-loss checkpoint selection. Reuse ViTESI validation-selected weights
`{REFERENCE_HASH}` from `{REFERENCE_COMMIT}` (held-out LE 7.147851 / test LE 7.100556 mm).
Evaluate both 2000-sample datasets with per-sample noise seed 0 + index and all five metrics.
Primary comparison: independent `test2k_s1` mean LE; held-out result and all other metrics also reported.
This is a single-seed comparison under our fixed settings, not confirmation of thesis replication
or reproduction of the original NMM-trained DeepSIF paper.

Next: {record['next_action']}
No CNN/LSTM/ablation runs, automatic retries/resumes, settings tuning, merge or closure authorized by this comparison.
{END}"""
    for kind, number in [("issue", "4"), ("pr", "14")]:
        body = json.loads(workflow.gh(kind, "view", number, "--json", "body"))["body"]
        workflow.gh(kind, "edit", number, body=replace_block(body, block))
    workflow.gh("pr", "comment", "14", body=message)
    vault = ROOT.parent / "Obsidian/05_Project"
    for note in [vault / "ablation_study/Ablation Study — Hub.md", vault / "SynapseAI Objectives → Code.md"]:
        body = note.read_text(encoding="utf-8")
        text = f"matched DeepSIF: {record['status']}; thesis reproduction ⛔"
        if note.name.startswith("Ablation Study"):
            body = re.sub(r"(?m)^\| \[4\].*$",
                f"| [4](https://github.com/{workflow.REPO}/issues/4) | Reproduce baselines | yes | {text} | [#14](https://github.com/{workflow.REPO}/pull/14) |", body)
        else:
            body = re.sub(r"(?m)^- \*\*Status:\*\*.*$",
                f"- **Status:** 🔧 Obj 2 in progress; #2/#3 done; #4 {text}.", body, count=1)
        note.write_text(body, encoding="utf-8")


def verify_exports(manifest):
    events = next(Path(manifest["checkpoints"]["validation_loss"]["weights"]).parents[1].rglob("events.out.tfevents.*"))
    data = EventAccumulator(str(events)).Reload()
    for monitor, item in manifest["checkpoints"].items():
        saved = torch.load(item["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
        exported = torch.load(item["weights"], map_location="cpu", weights_only=True)
        if saved.keys() != exported.keys() or not all(torch.equal(saved[k], exported[k]) for k in saved):
            raise RuntimeError(f"DeepSIF {monitor} export mismatch")
        if item["score"] != min(v.value for v in data.Scalars(monitor)):
            raise RuntimeError(f"DeepSIF {monitor} did not select the logged loss minimum")


def compare(record):
    output = Path(record["output"])
    record["paired_deepsif_minus_vit"] = {}
    for dataset in ["heldout", "test2k"]:
        a = pd.read_csv(output / f"eval_VIT_validation_loss_{dataset}/per_sample.csv")
        b = pd.read_csv(output / f"eval_DEEPSIF_validation_loss_{dataset}/per_sample.csv")
        if not a[["index", "id"]].equals(b[["index", "id"]]):
            raise RuntimeError("ViTESI and DeepSIF evaluation sample identities differ")
        differences = b[workflow.METRICS] - a[workflow.METRICS]
        paired = a[["index", "id"]].join(differences)
        paired.to_csv(output / f"paired_{dataset}.csv", index=False)
        record["paired_deepsif_minus_vit"][dataset] = {
            metric: float(differences[metric].mean()) for metric in workflow.METRICS
        }
        record["paired_deepsif_minus_vit"][dataset]["vit_lower_LE_fraction"] = float((differences["LE_mm"] > 0).mean())
    le_delta = record["paired_deepsif_minus_vit"]["test2k"]["LE_mm"]
    record["independent_test_LE_winner"] = "VIT" if le_delta > 0 else "DEEPSIF" if le_delta < 0 else "tie"
    save_manifest(output / "workflow.json", record)


def package(record):
    output = Path(record["output"])
    folder = output / "handoff"
    folder.mkdir()
    for model, manifest in record["training"].items():
        target = folder / model
        target.mkdir()
        shutil.copy2(manifest["manifest"], target / "run_manifest.json")
        for monitor, item in manifest["checkpoints"].items():
            dest = target / monitor
            dest.mkdir()
            for key in ["checkpoint", "weights"]:
                shutil.copy2(item[key], dest / Path(item[key]).name)
    for name in record["evaluations"]:
        shutil.copytree(output / name, folder / name)
    for name in ["paired_heldout.csv", "paired_test2k.csv"]:
        shutil.copy2(output / name, folder / name)
    shutil.copy2(MT / "ablation/requirements-lock.txt", folder / "requirements-lock.txt")
    (folder / "README.md").write_text(
        "# Matched ViTESI vs DeepSIF (#4 / PR #14)\n\n"
        "User authorized 2026-10-19 while thesis reproduction remains unresolved.\n"
        f"DeepSIF commit {record['git_commit']}; ViTESI commit {REFERENCE_COMMIT}.\n"
        "Both checkpoints selected only by validation loss. Primary endpoint: independent test mean LE.\n"
        "Single seed, frozen 6000/2000 split, 100 epochs, batch 8, Adam 0.001, cosine loss, linear scaling, "
        "white noise 5 dB, 500 Hz / 1 second, CUDA. No thesis-equivalence claim.\n"
        "Exact commands, PIDs, durations, checkpoint hashes and provenance: workflow.json and run manifests.\n"
        "Paired CSVs contain DeepSIF minus ViTESI metric differences on identical samples/noise.\n"
        "Manually upload this package to team Drive and record the link in #4 / PR #14. "
        "Do not merge/close #4 solely because this comparison finished.\n\n" + workflow.result_table(record) + "\n",
        encoding="utf-8",
    )
    record["package"] = str(output / "matched-deepsif-handoff.zip")
    record["manual_upload_pending"] = True
    save_manifest(output / "workflow.json", record)
    shutil.copy2(output / "workflow.json", folder / "workflow.json")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    (folder / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(folder).as_posix()}\n" for p in files), encoding="utf-8",
    )
    with zipfile.ZipFile(record["package"], "w", compression=zipfile.ZIP_STORED) as archive:
        for path in folder.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(folder))
    with zipfile.ZipFile(record["package"]) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Handoff ZIP integrity failure")
    record["package_sha256"] = sha256(record["package"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke-record", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run_name):
        parser.error("Invalid run name")
    output = MT / "results/ablation/matched_deepsif" / args.run_name / args.stage
    if output.exists():
        parser.error(f"Output exists; no automatic retry or overwrite: {output}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        parser.error("Source must be committed and clean")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.stage == "full":
        if not args.smoke_record:
            parser.error("Full run requires --smoke-record")
        smoke = json.loads(args.smoke_record.read_text())
        if smoke["status"] != "smoke_passed" or smoke["git_commit"] != commit:
            parser.error("DeepSIF smoke must pass on this exact commit")
    workflow.gh("issue", "view", "4", "--json", "state")
    workflow.gh("pr", "view", "14", "--json", "state")
    preflight = workflow.preflight(False)
    old, vit = reference(preflight)
    if args.stage == "smoke":
        workflow.preflight(True)
    output.mkdir(parents=True)
    record = {"run_name": args.run_name, "status": "running", "output": str(output),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": commit, "pid": os.getpid(), "current_stage": "preflight", "stages": [],
              "preflight": preflight, "reference_workflow": str(REFERENCE),
              "reference_workflow_sha256": sha256(REFERENCE), "reference_commit": REFERENCE_COMMIT,
              "smoke_record": str(args.smoke_record) if args.smoke_record else None,
              "thesis_reproduction": "unresolved", "primary_endpoint": "test2k_s1 mean LE_mm",
              "next_action": "Complete the CUDA DeepSIF smoke before full training." if args.stage == "smoke"
              else "DeepSIF trains for 100 epochs; evaluate both datasets, compare and package automatically."}
    save_manifest(output / "workflow.json", record)
    try:
        publish(record, "**What:** start the explicitly approved matched DeepSIF comparison (" + args.stage +
                "). **Why:** test relative performance while BrainForge settings are unavailable. "
                "**Why not:** no thesis-success claim, setting changes or extra baselines. "
                f"**How:** `{output}/workflow.json`; frozen protocol, validation selection, existing ViTESI reference.")
        deepsif = workflow.train(record, "DEEPSIF", smoke=args.stage == "smoke")
        verify_exports(deepsif)
        if args.stage == "smoke":
            workflow.evaluate(record, "DEEPSIF", deepsif, "validation_loss", "heldout", smoke=True)
            record["status"] = "smoke_passed"
            record["next_action"] = "Launch the full stage using this same-commit smoke record and a fresh full directory."
        else:
            for field in ["train_indices", "heldout_indices", "train_ids", "heldout_ids"]:
                if deepsif[field] != vit[field]:
                    raise RuntimeError(f"DeepSIF/ViTESI split mismatch: {field}")
            if deepsif["environment"] != vit["environment"]:
                raise RuntimeError("DeepSIF/ViTESI training environments differ")
            record["training"]["VIT"] = vit
            for dataset in ["heldout", "test2k"]:
                name = f"eval_VIT_validation_loss_{dataset}"
                shutil.copytree(REFERENCE.parent / name, output / name)
                record.setdefault("evaluations", {})[name] = old["evaluations"][name]
            publish(record, "**What:** DeepSIF completed all 100 epochs. **Why:** matched budget comparison. "
                    "**Why not:** no test-selected checkpoints. **How:** same split/environment; validation-loss export verified; both evaluations follow.")
            for dataset in ["heldout", "test2k"]:
                workflow.evaluate(record, "DEEPSIF", deepsif, "validation_loss", dataset)
            compare(record)
            record["status"] = "completed_awaiting_manual_upload"
            record["next_action"] = "Review all five metrics and paired results; upload ZIP manually and record Drive link. Thesis settings remain pending; no merge or closure."
            package(record)
            date = datetime.now(ZoneInfo("Asia/Colombo")).strftime("%Y-%m-%d")
            note = ROOT.parent / "Obsidian/05_Project/ablation_study/Ablation Results.md"
            note.write_text(note.read_text(encoding="utf-8") +
                f"\n## {date} · matched ViTESI vs DeepSIF\n\nDeepSIF `{commit}`; ViTESI `{REFERENCE_COMMIT}`. Single seed; same frozen settings, not thesis reproduction. Record `{output}/workflow.json`.\n\n" +
                workflow.result_table(record) + f"\n\nPaired DeepSIF minus ViTESI: `{json.dumps(record['paired_deepsif_minus_vit'])}`. "
                f"Local ZIP `{record['package']}`; SHA-256 `{record['package_sha256']}`; manual upload pending.\n", encoding="utf-8")
            roadmap = json.loads(workflow.gh("issue", "view", "1", "--json", "body"))["body"]
            block = BEGIN + "\n## Matched ViTESI vs DeepSIF (#4)\n\n" + workflow.result_table(record) + \
                f"\n\nSingle seed; DeepSIF `{commit}`, ViTESI `{REFERENCE_COMMIT}`. Thesis reproduction remains unresolved. Details: #4 / PR #14.\n" + END
            workflow.gh("issue", "edit", "1", body=replace_block(roadmap, block))
        save_manifest(output / "workflow.json", record)
        message = f"**What:** {record['status']}. **Why:** assess the user's fixed-protocol relative comparison. " \
            f"**Why not:** no thesis replication claim or settings tuning. **How:** `{output}/workflow.json`. {record['next_action']}\n\n" + workflow.result_table(record)
        publish(record, message)
        workflow.vault_log(record, f"Matched DeepSIF status {record['status']}. {record['next_action']}")
        print(message, flush=True)
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["next_action"] = "Inspect the current stage log; report the failure. No automatic retry, resume, fallback or setting changes."
        save_manifest(output / "workflow.json", record)
        try:
            publish(record, f"**What:** hard stop: {record['error']}. **Why:** preserve the approved comparison. "
                    f"**Why not:** no silent retry/fallback. **How:** `{output}/workflow.json` and current stage log.")
            workflow.vault_log(record, record["error"])
        except Exception as tracking_error:
            print(f"Tracking also failed: {tracking_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
