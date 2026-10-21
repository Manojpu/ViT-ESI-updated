# Issue #4: baseline reproduction and handoff

Start with [issue #4](https://github.com/Project-Synapse-AI/ViT-ESI-updated/issues/4)
and [draft PR #14](https://github.com/Project-Synapse-AI/ViT-ESI-updated/pull/14).
Use `ablation/baselines`; keep all related fixes, runs and handoff tasks there.
Do not create small child issues or start an ablation before resolving this gate.

## User-approved matched DeepSIF comparison (2026-10-19)

The user explicitly authorized checking ViTESI versus DeepSIF while waiting for
BrainForge's settings. This overrides the no-further-training stop **only for
this matched DeepSIF comparison**. The thesis reproduction gate remains failed.
Do not launch the old full workflow to perform this comparison: it retrains ViTESI.

```powershell
.\venv\Scripts\python.exe model_training/ablation/matched_deepsif.py --stage smoke --run-name matched-deepsif-20261019
.\venv\Scripts\python.exe model_training/ablation/matched_deepsif.py --stage full --run-name matched-deepsif-20261019 --smoke-record model_training/results/ablation/matched_deepsif/matched-deepsif-20261019/smoke/workflow.json
```

Use a fresh output name; clean committed source and same-commit DeepSIF CUDA
smoke are required. Preflight checks the locked environment, anatomy and all
referenced dataset files, as well as unchanged scientific code and frozen
config/index hashes against the prior ViTESI run. Smoke trains DeepSIF for two
epochs on existing genuine mes_debug_python (75 train/25 held out), verifies
checkpoint exports/loss minima and loads the primary weights through the harness.

Full stage trains DeepSIF only, from scratch, for 100 epochs on the same
6000/2000 split, batch 8, Adam 0.001, cosine loss, linear scaling, white noise
5 dB, seed 0, 500 Hz / 1 second, CUDA, no early stopping. Both checkpoint
criteria are retained, but only validation-selected weights enter the comparison.
Reuse ViTESI validation-selected weights from clean experimental commit 84d435c,
SHA-256 `0385eca30a2a423639cea9ca5bf53993cf9996ffb3a76235eccfdcd1da186a69`.
Do not substitute the historical 7.04 mm train-selected model.

The full runner evaluates DeepSIF on held-out train8k_s0 and independent test2k_s1,
checks matching IDs/noise protocol, copies the existing ViTESI evaluation records,
and reports all five metrics plus paired DeepSIF-minus-ViTESI differences.
The primary endpoint is independent-test mean LE; checkpoints never depend on
test LE. This is one training seed and the inherited DeepSIF implementation
retrained on SEREEGA, not reproduction of its original NMM-trained paper.
Unverified thesis settings remain unverified, including varying versus fixed
SNR and thesis 500 ms/1 ms resolution versus frozen 1000 ms/500 Hz.

Records: `model_training/results/ablation/matched_deepsif/<run-name>/<stage>/workflow.json`.
Full completion updates #4, PR #14, roadmap #1 and the vault, and creates
`matched-deepsif-handoff.zip` with both models, evaluations, paired CSVs,
commands, provenance, README and checksums. Manual Drive upload remains pending.
No automatic retry/resume, new seeds, tuning, CNN/LSTM/ablation runs or automatic
merge/closure. The original failed-gate records and package stay preserved.

## Evidence and decision

Historical commit `f7c8f13` trained ViTESI for 100 epochs at batch 8 / SNR 5.
The training-loss-selected epoch 98 scored LE 7.0370807 mm on the held-out
2,000 of `train8k_s0` and 7.0563444 mm on `test2k_s1`.
Weights SHA-256: `9ced236cce62aa7d049cef52d8453da9e55ed72d9b1cbc57fa28e00adacc8e01`.
Original files remain in `model_training/results/baseline/` and
`model_training/results/ablation/baseline/`.

Thesis section 4.4 says to select by validation loss. The old training code
selected by training loss. TensorBoard preferred validation epoch 96, score
−0.1135961, but those weights are unavailable. The saved epoch 98 validation
loss was −0.1131989. This small loss gap does not establish the cause of the
roughly 2 mm localization gap. Save both selections from one new trajectory.

Batch 8 and SNR 5 are approved, explicitly unverified thesis defaults.
No other settings are to be changed in this experiment. The loader supports
random SNR but the CLI does not; random SNR is outside this experiment.

## Current outcome: blocked after the controlled run (2026-10-13)

The user-approved fresh restart `checkpoint-selection-20261013-restart`
completed all 100 epochs on clean CUDA commit `84d435c`. Frozen datasets,
batch 8, SNR 5, optimizer/loss/scaling and full-run runtime settings stayed fixed.
Training plus validation took 15,471.377 seconds (4.30 hours); concurrent
machine workloads were observed, so this is not isolated throughput.

| Selection / weights | Epoch index | train8k_s0 held-out 2,000 LE mm | test2k_s1 2,000 LE mm |
|---|---:|---:|---:|
| validation_loss / VIT_model.pt | 93 | 7.147851 | 7.100556 |
| train_loss / VIT_model_train_loss.pt | 99 | 7.189857 | 7.061963 |

Validation selection improves held-out LE by only 0.042005 mm and fails the
4.57–5.57 mm gate. **Do not start another training or ablation.** Obtain
BrainForge's exact training/evaluation SNR, batch size and original notebook/
configuration for Table 5.1; continue in existing #4 / draft PR #14.

Full provenance, all five metrics and exact checkpoint paths/hashes are in
`model_training/results/ablation/reproduction/checkpoint-selection-20261013-restart/full/workflow.json`
and PR #14. Both checkpoint/weight hashes, all four 2,000-row CSVs, split
exclusion, finite metrics/summary means and CUDA/commit provenance were
independently checked. The local `baseline-handoff.zip` at that same location
is 115,996,268 bytes; ZIP CRC, SHA-256 and 16 internal file hashes verified.
ZIP SHA-256: `b17d62593c78a8146b72e544f2ccac0b77296c4d3c4285ea9dc493fec260a5e6`.
Manual Drive upload is pending; upload alone does not close #4 while the gate
and the remaining CNN/LSTM/DeepSIF baselines are unresolved. Interrupted v3
and failed smoke attempts remain preserved and are not reported as baselines.

## Reproduction commands (PowerShell, repo root)

Use the existing locked `venv`; do not reinstall or change dependencies.
Source must be committed and clean before each workflow starts.

```powershell
.\venv\Scripts\python.exe model_training/ablation/baseline_workflow.py --stage smoke --run-name checkpoint-selection-20261012
.\venv\Scripts\python.exe model_training/ablation/baseline_workflow.py --stage full --run-name checkpoint-selection-20261012 --smoke-record model_training/results/ablation/reproduction/checkpoint-selection-20261012/smoke/workflow.json
```

The full workflow requires a successful smoke record from the **same commit**.
Use a new run name if another run directory already exists; first investigate
the prior record, classify it as failed/interrupted/smoke, and document the
reason for another run. Never automatically retry or resume a partial run.
Both commands may take time; logs are redirected to the run directory.

The smoke uses existing SEREEGA `mes_debug_python` (100 genuine samples),
75 train / 25 validation, two epochs, batch 8, CUDA. It compares legacy
training-loss selection with dual selection: exact loss trajectories, split
IDs, loss histories and training-selected weights must match exactly. Only the
short verification runs use `-deterministic` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
the full scientific runs retain their original nondeterministic runtime settings.
Initial smoke attempts demonstrated cross-process CUDA variation, including an
unchanged legacy repeat; those failed verifications remain recorded. Each
export must match its own saved checkpoint bit-for-bit. Each checkpoint must
match its logged minimum and load successfully through the evaluation harness.

Full runs use frozen `train8k_s0`: 8,000 loaded, 6,000 train / 2,000 held out,
split seed 0, 100 full epochs, batch 8, Adam lr 0.001, cosine loss, linear
scaling, white noise 5 dB, CUDA; existing architectures and package versions.
`-checkpoint_monitor validation_loss -save_comparison_checkpoint -require_cuda`
are used. **Omit `-no_early_stop`**: the inherited flag is inverted and would
enable early stopping. The default disables it, matching the historical run.

Both checkpoint selections are scored on held-out `train8k_s0` and all of
`test2k_s1` using the existing harness with seed 0. Test LE never selects weights.
Mean held-out LE from the validation-selected weights must be **4.57–5.57 mm**.
On failure, stop and request BrainForge's training/evaluation SNR, batch size
and original notebook/configuration; do not tune or start another model.
On success, train 1D-CNN, LSTM and DeepSIF sequentially with the same recipe,
retain both checkpoint criteria, and evaluate validation-selected weights on
both datasets. Finish all baseline work within #4.

## Records and completion

Outputs: `model_training/results/ablation/reproduction/<run-name>/<stage>/`.
`workflow.json` records the workflow/child PIDs, stages, exact argv and working
directories, failures, gate, paired metric differences and next action.
Each training writes `run_manifest.json` with arguments, commit, environment,
split indices and IDs, duration, parameter counts, selected epochs and hashes.
Primary weights keep `<MODEL>_model.pt`; comparison weights use
`<MODEL>_model_<monitor>.pt`. Checkpoint folders are separate by monitor.

The runner updates #4 / PR #14 at milestones, records final results in #1,
and updates the Obsidian progress/objectives/results/hub notes. If GitHub
tracking fails, it stops and leaves the local record; repair access and
document completed work before attempting anything else.

A completed full workflow prepares `baseline-handoff.zip` with checkpoints,
weights, manifests, evaluation CSVs, README and checksums. The user approved
local packaging followed by **manual** upload to
`SynapseAI-FYP/checkpoints/baseline/`; no dataset upload is needed.
Keep #4 and PR #14 open until the gate and all four baselines are complete,
the user has uploaded the package, and the Drive link is recorded. Then fill
the results/experiment sections, mark PR #14 ready, merge to `dev`, close #4,
and proceed to existing #5 (A3 peak position).
