"""Issue #4: real-data smoke verification and the gated, frozen baseline protocol.

Run from the repo root with its locked venv. No automatic retry, resume, tuning,
regeneration or CPU fallback. A fresh run name is required for each invocation.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
MT = ROOT / "model_training"
sys.path.insert(0, str(MT))
from ablation.training_record import save_manifest, sha256

REPO = "Project-Synapse-AI/ViT-ESI-updated"
BEGIN = "<!-- baseline-reproduction-status -->"
END = "<!-- /baseline-reproduction-status -->"
METRICS = ["LE_mm", "AUC", "nMSE", "PSNR", "time_err_ms"]


def gh(*args, body=None):
    command = ["gh", *args, "--repo", REPO]
    if body is not None:
        command += ["--body-file", "-"]
    # Windows text-mode stdin converts LF to CRLF. Re-sending a fetched JSON
    # body containing CRLF otherwise adds another CR on every status update.
    payload = re.sub(r"\r+\n", "\n", body).replace("\r", "\n").encode("utf-8") if body is not None else None
    result = subprocess.run(command, input=payload, capture_output=True, check=True)
    return result.stdout.decode("utf-8")


def replace_status(body, block):
    body = re.sub(r"\r+\n", "\n", body).replace("\r", "\n")
    if BEGIN in body:
        start = body.index(BEGIN)
        stop = body.index(END, start) + len(END)
        return body[:start] + block + body[stop:]
    return body.rstrip() + "\n\n" + block + "\n"


def status_text(record):
    completed = {s["name"] for s in record.get("stages", []) if s["status"] == "completed"}
    check = lambda value: "x" if value else " "
    run_dir = record["output"]
    return f"""{BEGIN}
## Current handoff — {record['updated_utc']}

Status: **{record['status']}**. Stage: **{record.get('current_stage', 'preflight')}**.
Branch `ablation/baselines`; draft PR #14; commit `{record['git_commit']}`.
Output on the current host: `{run_dir}`. Read `workflow.json`, then the stage log and `run_manifest.json`; do not launch a duplicate run while its process is active.

- [x] Protocol audit: thesis §4.4 selects by validation loss; old run saved epoch 98 by training loss, while validation preferred epoch 96 (weights unavailable).
- [x] Checkpoint controls and manifest implemented.
- [{check(record.get('smoke_verified', False))}] Real-data CUDA smoke verification, identical legacy/dual training trajectories.
- [{check('train_VIT' in completed)}] Fresh 100-epoch ViTESI run, both checkpoint criteria saved.
- [{check('eval_VIT_validation_loss_heldout' in completed and 'eval_VIT_train_loss_test2k' in completed)}] Both checkpoints scored on both datasets, all five metrics and verified IDs.
- [{check('gate' in record)}] Gate decision: held-out validation-selected LE must be 4.57–5.57 mm.
- [{check(all('train_' + m in completed and 'eval_' + m + '_validation_loss_test2k' in completed for m in ['1DCNN', 'LSTM', 'DEEPSIF']))}] Remaining CNN/LSTM/DeepSIF baselines (only after gate passes).
- [{check('package' in record)}] Local artifact package with README and checksums.
- [ ] User's manual Drive upload and recorded link; required before closing #4.

Fixed protocol: train8k_s0, 8000 loaded, split seed 0, 6000/2000, 100 epochs, batch 8, Adam lr 0.001, cosine loss, linear scaler, white noise 5 dB, CUDA, no early stopping. SNR 5 and batch 8 remain unverified thesis settings, explicitly approved defaults. Test set: test2k_s1, 2000; never used to select weights. Frozen datasets and metric math unchanged.

Historical reference: `f7c8f13`, epoch 98, checkpoint weights SHA-256 `9ced236cce62aa7d049cef52d8453da9e55ed72d9b1cbc57fa28e00adacc8e01`; held-out LE 7.0371 mm, unseen LE 7.0563 mm.

Next action: {record.get('next_action', 'Inspect workflow.json and continue the current stage.')}
{END}"""


def publish(record, message):
    block = status_text(record)
    for kind, number in [("issue", "4"), ("pr", "14")]:
        current = json.loads(gh(kind, "view", number, "--json", "body"))["body"]
        gh(kind, "edit", number, body=replace_status(current, block))
    gh("pr", "comment", "14", body=message)


def vault_log(record, message):
    vault = ROOT.parent / "Obsidian/05_Project"
    date = datetime.now(ZoneInfo("Asia/Colombo")).strftime("%Y-%m-%d")
    entry = f"\n- **{date} · #4 · {record['current_stage']}** — {message} Commit `{record['git_commit']}`; records `{record['output']}/workflow.json`; PR [#14](https://github.com/{REPO}/pull/14).\n"
    for note in [vault / "ablation_study/Ablation Progress Log.md", vault / "SynapseAI Objectives → Code.md"]:
        body = note.read_text(encoding="utf-8")
        position = body.rfind("\n## Related")
        body = body[:position] + entry + body[position:] if position != -1 else body + entry
        note.write_text(body, encoding="utf-8")


def preflight(smoke):
    missing = []
    for line in (MT / "ablation/requirements-lock.txt").read_text().splitlines():
        if "==" in line:
            name, expected = line.split("==")
            actual = version(name)
            if actual != expected:
                missing.append(f"{name}: expected {expected}, found {actual}")
    if not torch.cuda.is_available():
        missing.append("CUDA unavailable")
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        missing.append("Less than 5 GiB free for logs, checkpoints and package")
    leadfield = ROOT / "anatomy/leadfield_75_20k.mat"
    if not leadfield.is_file():
        missing.append(str(leadfield))
    elif not any(getattr(v, "shape", None) == (75, 994) for v in loadmat(leadfield).values()):
        missing.append("Leadfield is not 75 x 994")
    model = ROOT / "simulation/fsaverage/constrained/standard_1020/fsav_994/model"
    for name in ["fs_cortex_20k_region_mapping.mat", "fs_cortex_neighbors_994.mat", "ch_fsav_994.mat"]:
        if not (model / name).is_file():
            missing.append(str(model / name))
    if not (ROOT / "anatomy/sources_fsav_994.mat").is_file():
        missing.append(str(ROOT / "anatomy/sources_fsav_994.mat"))
    names = [("mes_debug_python", 100)] if smoke else [("train8k_s0", 8000), ("test2k_s1", 2000)]
    datasets = {}
    for name, expected in names:
        folder = model.parent / "simu" / name
        config = folder / f"{name}fsav_994_config.json"
        index = folder / f"{name}fsav_994_match_json_file.json"
        if not config.is_file() or not index.is_file():
            missing.append(f"{name}: missing config or file index")
            continue
        config_data = json.loads(config.read_text())
        if config_data["rec_info"]["n_times"] != 500 or config_data["rec_info"]["fs"] != 500:
            missing.append(f"{name}: unexpected sampling or epoch length")
        mapping = json.loads(index.read_text())
        if len(mapping) != expected:
            missing.append(f"{name}: expected {expected} samples, found {len(mapping)}")
        for row in mapping.values():
            for file in row.values():
                if not Path(file).is_file():
                    missing.append(file)
        datasets[name] = {"samples": len(mapping), "config_sha256": sha256(config), "index_sha256": sha256(index)}
    if missing:
        raise RuntimeError("HARD STOP: preflight failed:\n" + "\n".join(missing[:30]))
    return {"datasets": datasets, "gpu": torch.cuda.get_device_name(), "free_bytes": shutil.disk_usage(ROOT).free}


def run_stage(record, name, command, cwd):
    record["current_stage"] = name
    entry = {"name": name, "command": command, "cwd": str(cwd), "status": "running"}
    record["stages"].append(entry)
    save_manifest(Path(record["output"]) / "workflow.json", record)
    print(f"START {name}", flush=True)
    log = Path(record["output"]) / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        environment = dict(os.environ)
        if "-deterministic" in command:
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        entry["verification_environment"] = {"CUBLAS_WORKSPACE_CONFIG": environment.get("CUBLAS_WORKSPACE_CONFIG")}
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        entry["pid"] = child.pid
        save_manifest(Path(record["output"]) / "workflow.json", record)
        code = child.wait()
    entry["exit_code"] = code
    entry["status"] = "completed" if code == 0 else "failed"
    save_manifest(Path(record["output"]) / "workflow.json", record)
    if code:
        raise RuntimeError(f"{name} failed, exit {code}; inspect {log}")
    print(f"DONE {name}", flush=True)


def train(record, model, *, smoke=False, dual=True, label=None):
    output = Path(record["output"]) / (label or f"train_{model}")
    name = "mes_debug_python" if smoke else "train8k_s0"
    command = [sys.executable, str(MT / "main_train.py"), name,
        "-results_path", str(output), "-simu_type", "sereega", "-source_space", "fsav_994",
        "-electrode_montage", "standard_1020", "-orientation", "constrained", "-model", model,
        "-loss", "cosine", "-scaler", "linear", "-eeg_snr", "5", "-n_times", "500",
        "-per_valid", "0.25", "-to_load", "100" if smoke else "8000",
        "-n_epochs", "2" if smoke else "100", "-batch_size", "8", "-require_cuda",
        "-leadfield_mat", str(ROOT / "anatomy/leadfield_75_20k.mat"),
        "-simu_folder", str(ROOT / f"simulation/fsaverage/constrained/standard_1020/fsav_994/simu/{name}")]
    if dual:
        command += ["-checkpoint_monitor", "validation_loss", "-save_comparison_checkpoint"]
    if smoke:
        command += ["-deterministic"]
    # Legacy invocation deliberately omits both new checkpoint-selection flags.
    run_stage(record, label or f"train_{model}", command, MT)
    paths = list(output.rglob("run_manifest.json"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one manifest under {output}")
    manifest = json.loads(paths[0].read_text())
    if manifest["status"] != "completed" or manifest["completed_epochs"] != (2 if smoke else 100):
        raise RuntimeError("Training did not complete the planned number of epochs")
    if manifest["early_stopping_enabled"] or manifest["environment"]["device"] != "cuda":
        raise RuntimeError("Training protocol differs from the approved CUDA/full-epoch protocol")
    record.setdefault("training", {})[label or model] = {"manifest": str(paths[0]), **manifest}
    save_manifest(Path(record["output"]) / "workflow.json", record)
    return manifest


def evaluate(record, model, manifest, monitor, dataset, *, smoke=False):
    label = f"eval_{model}_{monitor}_{dataset}"
    output = Path(record["output"]) / label
    name = "mes_debug_python" if smoke else ("train8k_s0" if dataset == "heldout" else "test2k_s1")
    n = "100" if smoke else ("8000" if dataset == "heldout" else "2000")
    checkpoint = manifest["checkpoints"][monitor]
    command = [sys.executable, str(MT / "ablation/evaluate.py"), name,
        "-root_simu", str(ROOT), "-ckpt", checkpoint["weights"], "-model", model,
        "-to_load", n, "-out", str(output), "-device", "cuda", "-eeg_snr", "5", "-seed", "0"]
    if dataset == "heldout":
        command += ["-heldout", "-per_valid", "0.25"]
    run_stage(record, label, command, ROOT)
    summary = json.loads((output / "summary.json").read_text())
    frame = pd.read_csv(output / "per_sample.csv")
    expected_indices = manifest["heldout_indices"] if dataset == "heldout" else list(range(2000))
    if frame["index"].tolist() != expected_indices:
        raise RuntimeError(f"Wrong evaluation indices in {label}")
    if dataset == "heldout" and frame["id"].tolist() != manifest["heldout_sample_ids"]:
        raise RuntimeError(f"Wrong evaluation sample IDs in {label}")
    if dataset != "heldout":
        index = ROOT / f"simulation/fsaverage/constrained/standard_1020/fsav_994/simu/{name}/{name}fsav_994_match_json_file.json"
        mapping = json.loads(index.read_text())
        ids = [json.loads(Path(row["md_json_file_name"]).read_text()).get("id", key)
               for key, row in mapping.items()]
        if frame["id"].tolist() != ids:
            raise RuntimeError(f"Wrong independent test IDs in {label}")
    if set(manifest["train_indices"]) & set(manifest["heldout_indices"]):
        raise RuntimeError("Training/heldout overlap")
    if not np.isfinite(frame[METRICS].to_numpy()).all():
        raise RuntimeError(f"Non-finite or missing metrics in {label}")
    if summary["device"] != "cuda" or summary["ckpt_sha256"] != checkpoint["weights_sha256"]:
        raise RuntimeError(f"Checkpoint/device provenance mismatch in {label}")
    record.setdefault("evaluations", {})[label] = summary
    save_manifest(Path(record["output"]) / "workflow.json", record)
    return summary


def verify_smoke(record):
    legacy = train(record, "VIT", smoke=True, dual=False, label="smoke_legacy")
    dual = train(record, "VIT", smoke=True, label="smoke_dual")
    for field in ["train_indices", "heldout_indices", "train_ids", "heldout_ids"]:
        if legacy[field] != dual[field]:
            raise RuntimeError(f"Smoke split mismatch: {field}")
    def scalars(manifest):
        event = next(Path(manifest["checkpoints"]["train_loss"]["weights"]).parents[1].rglob("events.out.tfevents.*"))
        data = EventAccumulator(str(event)).Reload()
        return {tag: [(v.step, v.value) for v in data.Scalars(tag)] for tag in ["train_loss", "validation_loss"]}
    if scalars(legacy) != scalars(dual):
        raise RuntimeError("Legacy/dual smoke training trajectories differ")
    left = torch.load(legacy["checkpoints"]["train_loss"]["weights"], weights_only=True)
    right = torch.load(dual["checkpoints"]["train_loss"]["weights"], weights_only=True)
    if left.keys() != right.keys():
        raise RuntimeError("Legacy/dual smoke weight keys differ")
    # Verification alone requests deterministic CUDA operations. Full scientific
    # runs retain their inherited runtime settings and record that distinction.
    maximum = max(float((left[k] - right[k]).abs().max()) for k in left)
    record["smoke_weight_comparison"] = {"max_abs_difference": maximum, "atol": 0, "rtol": 0}
    if not all(torch.equal(left[k], right[k]) for k in left):
        raise RuntimeError(f"Deterministic legacy/dual smoke weights differ: {maximum}")
    for monitor, item in dual["checkpoints"].items():
        saved = torch.load(item["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
        exported = torch.load(item["weights"], weights_only=True)
        if saved.keys() != exported.keys() or not all(torch.equal(saved[k], exported[k]) for k in saved):
            raise RuntimeError(f"Export does not exactly match its {monitor} checkpoint")
        events = scalars(dual)[monitor]
        best_step, best_loss = min(events, key=lambda v: v[1])
        if item["score"] != best_loss or item["epoch"] != (best_step + 1) // 10 - 1:
            raise RuntimeError(f"Smoke selected wrong checkpoint for {monitor}")
        evaluate(record, "VIT", dual, monitor, "heldout", smoke=True)
    record["smoke_verified"] = True
    record["status"] = "smoke_passed"
    record["next_action"] = "Run the full workflow with this smoke workflow.json supplied as --smoke-record; no settings changed."


def package(record):
    output = Path(record["output"])
    package_dir = output / "handoff"
    package_dir.mkdir()
    for name, manifest in record.get("training", {}).items():
        folder = package_dir / name
        folder.mkdir()
        shutil.copy2(manifest["manifest"], folder / "run_manifest.json")
        for monitor, checkpoint in manifest["checkpoints"].items():
            target = folder / monitor
            target.mkdir()
            for key in ["checkpoint", "weights"]:
                shutil.copy2(checkpoint[key], target / Path(checkpoint[key]).name)
    for name in record.get("evaluations", {}):
        shutil.copytree(output / name, package_dir / name)
    record["package"] = str(output / "baseline-handoff.zip")
    record["manual_upload_pending"] = True
    save_manifest(output / "workflow.json", record)
    shutil.copy2(output / "workflow.json", package_dir / "workflow.json")
    shutil.copy2(MT / "ablation/requirements-lock.txt", package_dir / "requirements-lock.txt")
    readme = f"""# Baseline reproduction #4 — local handoff

Commit: {record['git_commit']}. Run: {record['run_name']}. Status: {record['status']}.
Gate: {json.dumps(record.get('gate'))}.
All exact argv commands and working directories are in workflow.json; training manifests include seeds, split IDs, environment, parameter counts, selected epochs and file hashes. All evaluations include dataset, checkpoint hash and all five metrics.
Protocol: 8000 seed-0 samples split 6000/2000, 100 full epochs, batch 8, Adam lr 0.001, cosine loss, linear scaling, white noise SNR 5 dB, CUDA. Batch 8 and SNR 5 remain unverified thesis defaults.
Historical reference: f7c8f13, epoch 98, held-out LE 7.0371 mm, unseen LE 7.0563 mm. Its original files are preserved outside this package.
Upload this ZIP manually to SynapseAI-FYP/checkpoints/baseline/, then add its Drive link to issue #4 and PR #14. No dataset upload is required. Do not close #4 if the gate failed or models remain unfinished.
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")
    files = sorted(p for p in package_dir.rglob("*") if p.is_file())
    (package_dir / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(p)}  {p.relative_to(package_dir).as_posix()}" for p in files) + "\n")
    with zipfile.ZipFile(record["package"], "w", compression=zipfile.ZIP_STORED) as archive:
        for path in package_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir))
    record["package_sha256"] = sha256(record["package"])


def result_table(record):
    lines = ["| Model / selection / dataset | LE mm | AUC | nMSE | PSNR dB | Time error ms |", "|---|---:|---:|---:|---:|---:|"]
    for name, summary in record.get("evaluations", {}).items():
        values = [summary["metrics"][key]["mean"] for key in METRICS]
        lines.append(f"| {name.removeprefix('eval_')} | " + " | ".join(f"{v:.6g}" for v in values) + " |")
    return "\n".join(lines)


def finish_notes(record):
    vault = ROOT.parent / "Obsidian/05_Project"
    date = datetime.now(ZoneInfo("Asia/Colombo")).strftime("%Y-%m-%d")
    table = result_table(record)
    result_note = vault / "ablation_study/Ablation Results.md"
    result_note.write_text(result_note.read_text(encoding="utf-8") +
        f"\n## {date} · #4 controlled checkpoint-selection experiment\n\nCommit `{record['git_commit']}`; status **{record['status']}**. Full checkpoint paths and SHA-256 values: `{record['output']}/workflow.json`.\n\n{table}\n\nGate: `{json.dumps(record.get('gate'))}`. Local package: `{record['package']}`; manual Drive upload pending.\n", encoding="utf-8")
    state = "⛔ controlled gate still missed" if not record["gate"]["passed"] else "🔧 baselines scored; Drive upload pending"
    for note in [vault / "ablation_study/Ablation Study — Hub.md", vault / "SynapseAI Objectives → Code.md"]:
        body = note.read_text(encoding="utf-8")
        if note.name.startswith("Ablation Study"):
            body = re.sub(r"(?m)^\| \[4\].*$", f"| [4](https://github.com/{REPO}/issues/4) | Reproduce baselines | yes | {state} | [#14](https://github.com/{REPO}/pull/14) |", body)
        else:
            body = re.sub(r"(?m)^- \*\*Status:\*\*.*$", f"- **Status:** 🔧 Obj 2 in progress; #2/#3 done; #4 {state}.", body, count=1)
        note.write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke-record", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.run_name):
        parser.error("run-name must contain only lowercase letters, digits, underscores and hyphens")
    output = MT / "results/ablation/reproduction" / args.run_name / args.stage
    if output.exists():
        parser.error(f"Output already exists; no overwrite or automatic retry: {output}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        parser.error("Commit source changes before running; clean provenance is required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    smoke_record = None
    if args.stage == "full":
        if not args.smoke_record:
            parser.error("Full run requires --smoke-record")
        smoke_record = json.loads(args.smoke_record.read_text())
        if not smoke_record.get("smoke_verified") or smoke_record["git_commit"] != commit:
            parser.error("Smoke verification must pass on this exact commit")
    # Validate GitHub access before reserving a run directory or starting GPU work.
    gh("issue", "view", "4", "--json", "state")
    gh("pr", "view", "14", "--json", "state")
    output.mkdir(parents=True)
    record = {"run_name": args.run_name, "status": "running", "output": str(output),
        "git_commit": commit, "pid": os.getpid(), "current_stage": "preflight", "stages": [],
        "smoke_verified": bool(smoke_record), "smoke_record": str(args.smoke_record) if args.smoke_record else None}
    save_manifest(output / "workflow.json", record)
    try:
        record["preflight"] = preflight(args.stage == "smoke")
        save_manifest(output / "workflow.json", record)
        publish(record, "**What:** start the approved checkpoint-selection experiment. **Why:** thesis validation selection differs from the inherited training-loss selection. **Why not:** no SNR/batch sweep or extra issues. **How:** frozen data, unchanged optimizer/model/noise, CUDA; exact commands are saved in workflow.json.")
        if args.stage == "smoke":
            verify_smoke(record)
            gh("issue", "edit", "4", "--remove-label", "blocked")
        else:
            vit = train(record, "VIT")
            save_manifest(output / "workflow.json", record)
            publish(record, "**What:** ViTESI training completed; both checkpoint criteria retained. **Why:** compare selection within one trajectory. **Why not:** test LE never selects weights. **How:** 100 epochs; inspect the run manifest for epochs, scores, hashes and duration. Evaluation is next.")
            for monitor in ["validation_loss", "train_loss"]:
                for dataset in ["heldout", "test2k"]:
                    evaluate(record, "VIT", vit, monitor, dataset)
            validation = record["evaluations"]["eval_VIT_validation_loss_heldout"]["metrics"]["LE_mm"]["mean"]
            record["gate"] = {"LE_mm": validation, "lower": 4.57, "upper": 5.57, "passed": 4.57 <= validation <= 5.57}
            a = pd.read_csv(output / "eval_VIT_validation_loss_heldout/per_sample.csv").set_index("index")
            b = pd.read_csv(output / "eval_VIT_train_loss_heldout/per_sample.csv").set_index("index")
            record["paired_validation_minus_train"] = {key: float((a[key] - b[key]).mean()) for key in METRICS}
            save_manifest(output / "workflow.json", record)
            if record["gate"]["passed"]:
                for model in ["1DCNN", "LSTM", "DEEPSIF"]:
                    manifest = train(record, model)
                    for dataset in ["heldout", "test2k"]:
                        evaluate(record, model, manifest, "validation_loss", dataset)
                    save_manifest(output / "workflow.json", record)
                    publish(record, f"**What:** {model} baseline completed. **Why:** finish #4 after the ViTESI gate passed. **Why not:** no extra model issues or settings changes. **How:** identical frozen protocol, validation-selected checkpoint, both datasets and all five metrics.\n\n" + result_table(record))
                    vault_log(record, f"{model} completed and evaluated on both sets.")
                record["status"] = "awaiting_manual_upload"
                record["next_action"] = "User: upload the local ZIP to Drive and provide its link. Agent: record the link, complete roadmap #1, ready/merge PR #14 and close #4; then #5."
            else:
                record["status"] = "gate_failed"
                record["next_action"] = "Obtain BrainForge's exact training/evaluation SNR, batch size and original notebook/configuration. No more trainings, tuning or ablations are authorized under this gate."
                gh("issue", "edit", "4", "--add-label", "blocked")
            package(record)
            finish_notes(record)
            current = json.loads(gh("issue", "view", "1", "--json", "body"))["body"]
            block = BEGIN + "\n## Controlled reproduction record (#4)\n\n" + result_table(record) + f"\n\nCommit `{commit}`; gate `{json.dumps(record['gate'])}`. Details and unfinished tasks: #4 / PR #14.\n" + END
            gh("issue", "edit", "1", body=replace_status(current, block))
        save_manifest(output / "workflow.json", record)
        message = f"**What:** {record['status']}. **Why:** follow the fixed reproduction gate. **Why not:** no tuning or extra issues. **How:** record `{output}/workflow.json`; commit `{commit}`. Next: {record['next_action']}\n\n" + result_table(record)
        publish(record, message)
        vault_log(record, f"Status {record['status']}. {record['next_action']}")
        print(message, flush=True)
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["next_action"] = f"Inspect {output}/workflow.json and {record['current_stage']}.log; report the missing dependency or failure. Do not silently retry, resume or change settings."
        save_manifest(output / "workflow.json", record)
        try:
            gh("issue", "edit", "4", "--add-label", "blocked")
            publish(record, f"**What:** hard stop: {record['error']}. **Why:** the approved protocol must not silently degrade. **Why not:** no fallback or automatic retry. **How:** {record['next_action']}")
            vault_log(record, record["error"])
        except Exception as tracking_error:
            print(f"Tracking also failed: {tracking_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
