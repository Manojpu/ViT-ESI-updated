# Issue #4: baseline reproduction and handoff

Start with [issue #4](https://github.com/Project-Synapse-AI/ViT-ESI-updated/issues/4)
and [draft PR #14](https://github.com/Project-Synapse-AI/ViT-ESI-updated/pull/14).
Use `ablation/baselines`; keep all related fixes, runs and handoff tasks there.
Do not create small child issues or start an ablation before resolving this gate.

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
IDs and training-selected weights must match. Each selected checkpoint must
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
