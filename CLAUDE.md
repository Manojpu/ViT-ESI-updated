# CLAUDE.md

Team SynapseAI's fork of BrainForge's ViTESI: EEG source imaging that maps 75 scalp electrodes to 994 cortical regions, 500 timepoints @ 500 Hz. `upstream` = Project-BrainForge (read-only). See `CONTRIBUTING.md` for branches and remotes.

## How we work (follow exactly)
1. **Pick an issue.** Current plan: milestone *Obj 2 · Ablation study*, roadmap in #1, work issues in the order its flow shows.
2. **Branch off `dev`** (`ablation/<test>`, `real-data/<dataset>`, `freq/<topic>`, `chore/<thing>`) and open a **draft PR right away**, titled `WIP: <issue title>`, body `Closes #N`, using the PR template.
3. **Post every decision as a PR comment**, in plain simple English (no jargon without a one-line explanation), with these four parts:
   - **What** we decided.
   - **Why** this option.
   - **Why not** the alternatives we considered.
   - **How** it's done (files, commands, key numbers).
4. Push small commits as you go. When done: fill in the PR template's results, update the results table in #1, mark the PR ready, merge into `dev`.
5. **Keep related work within its existing issue and PR.** Track small tasks with checklists; do not create child issues for each fix or run. The issue must give a new agent the current status, commands, evidence, blockers and exact next action.

## ⛔ Hard stop rule
If any dependency or piece of infra is missing or not visible, **stop immediately** and report it. That means a package, CUDA/GPU, dataset, checkpoint, anatomy file, thesis setting, credential or permission. **Don't hack around it**: workarounds silently degrade results.

The report should say:
- what is missing,
- where you looked,
- what it blocks,
- what the user needs to provide.

Add the `blocked` label to the issue.

Never, without asking first:
- fall back to CPU,
- shrink the dataset, epochs or batch "just to make it run",
- use mock or random data,
- swap package versions,
- edit `anatomy/`,
- skip a metric,
- guess a thesis hyperparameter.

## Setup & commands
Windows + PowerShell, RTX 3060 Laptop 6 GB. From the repo root:
1. `python -m venv venv; venv\Scripts\activate`
2. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
3. `pip install -r requirements.txt`
4. `$env:PROJECT_ROOT = (Get-Location).Path`

Exact versions are in `model_training/ablation/requirements-lock.txt`. Once per clone, run `python model_training/ablation/build_model_folder.py -root .`. It builds the head-model folder that the evaluation code needs.
- Generate data: `python data_generation/sereega/simu_extended_source.py -sin <name> -ne <N> -mk standard_1020 -ss fsav_994 -o constrained -sn fsaverage -rf "$PROJECT_ROOT" --leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" -fs 500 -d 1000 -af "$PROJECT_ROOT/anatomy"`
- Train (from `model_training/`): `python main_train.py <simu_name> -simu_type sereega -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained -model {VIT,1DCNN,LSTM,DEEPSIF} -loss cosine -scaler linear -eeg_snr 5 -n_times 500 -per_valid 0.25 -to_load 8000 -n_epochs 100 -leadfield_mat ... -simu_folder ... -results_path "$PROJECT_ROOT/model_training/results"`. ViT size flags: `-vit_depth -vit_heads -vit_embed_dim -vit_mlp_dim`.
- **Evaluate ablations with the harness**, not eval.py: `python model_training/ablation/evaluate.py <simu_name> -root_simu . -ckpt <.pt> -model VIT -to_load <N> -out model_training/results/ablation/<test>/<run>`.
  - It writes `per_sample.csv` (LE_mm, AUC, nMSE, PSNR, time_err_ms, peak_time_ms…) and `summary.json`.
  - Each sample is seeded with `seed + index`, so every run sees identical noise.
  - `-heldout -per_valid <p>` scores only the samples main_train held out for that `-to_load`/`-per_valid`.
  - Use `-device cuda` for every run you compare. CPU differs by float rounding.
  - In Python: `ablation.evaluate.load_context()` + `evaluate(model, ctx)`.
- eval.py (multi-method, thesis script): pass `-root_simu <repo>/simulation/fsaverage`, the subject folder, not the repo root.
- Params/FLOPs: `python count_flops.py --model VIT --input_shape 500 75`.
- No test suite. "Testing" means a smoke run with small `-ne`, `-to_load` and `-n_epochs`, and it must pass before any full run.
- Baseline reproduction #4: see `model_training/ablation/BASELINE_REPRODUCTION.md`. Use `-require_cuda -checkpoint_monitor validation_loss`; `-save_comparison_checkpoint` also saves the best training-loss checkpoint. The default remains `train_loss` for old commands. Runs write `run_manifest.json` with split indices and checkpoint provenance.
- **Legacy early-stop flag is inverted:** omitting `-no_early_stop` disables early stopping; passing it enables early stopping. Do not pass it in the planned full 100-epoch baseline runs.

## Code facts that matter
- `model_training/models/vit.py` `EEGViT`: each **timepoint is a token** (75 → 256 embed, depth 6, 8 heads), built on `nn.TransformerEncoderLayer`. In `eval()` it uses PyTorch's fused fast path, which skips Python hooks. Disable it (`torch.backends.mha.set_fastpath_enabled(False)`) before masking heads.
- SEREEGA data is EEG `(75, 500)` sensors-first. `Jact` holds only active regions, scattered back to 994 via `md/<id>_md_json_flie.json`.
  - **ERP peaks:** the default base centre is 125–375 ms, and each patch varies ±30% around it, so peaks fall at about 88–488 ms of the 1000 ms window.
  - **Peak metadata:** the md JSON stores `base_center` plus each patch's `erp` (`center`, `width`, `ampl`).
  - **Seed:** generator `--seed`, default 0. Seed 0 equals the old hard-coded seed, so any new set needs a new seed or it duplicates the training data.
  - **Patch counts:** patches per sample and patch order are actually 1–4, because `randint` excludes its upper bound.
- The metric math lives in `utils/utl_eval.py` (`sample_metrics`), moved verbatim from eval.py, quirks included. Don't change it: any edit breaks comparability with the thesis.
- The 5 metrics: LE (mm), AUC, nMSE, PSNR (dB), time error (ms). Thesis baselines (params M / LE mm): 1D-CNN 5.61/6.52 · LSTM 0.45/7.17 · DeepSIF 22.36/5.19 · **ViTESI 3.57/5.07**.

## Data & results
- Never commit `simulation/`, `model_training/results/`, `.pt`/`.ckpt` or `.mat` outputs.
- **Frozen datasets** (see `data_generation/FROZEN_DATASETS.md`):
  - `train8k_s0` (train), `test2k_s1` (clean test) and `widepeak2k_s2` (A3).
  - They are rebuilt locally from the commands in that file, not downloaded, and come out byte-identical.
  - Never regenerate them with other settings.
- **Thesis protocol:** always train with `-to_load 8000 -per_valid 0.25`, which gives 6,000 train / 2,000 held out. The split seed is the hard-coded `seed_everything(0)`. Report each model twice:
  - **held-out 2,000** (`evaluate.py train8k_s0 -to_load 8000 -heldout -per_valid 0.25`): comparable with the thesis's 5.07 mm;
  - **`test2k_s1`**: never seen, not even for early stopping.
- Checkpoints go to the team Drive (`SynapseAI-FYP/checkpoints/`) with a README giving the exact command, seed and commit hash.
- For #4 the user approved local checkpoint packaging followed by manual Drive upload. Training may proceed without Drive access; keep #4 open until its required runs and upload/link handoff are complete.
- Every reported number must say which dataset, checkpoint and commit it came from.
- Compare against the #4 baseline on the same frozen test set.
