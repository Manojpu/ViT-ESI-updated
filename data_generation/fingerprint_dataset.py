"""
Content fingerprint of a SEREEGA simulation folder, so a rebuild on another machine
(e.g. a Kaggle notebook) can be checked against our local frozen datasets.

Why not file hashes: scipy's savemat writes a creation timestamp into every .mat header,
and the EEG is a leadfield matrix product whose last float bits can differ between BLAS
builds. So each sample is checked on two levels:
  - structure (exact): sample order, seeds, patch orders, patch region ids;
  - numbers (tolerant): ERP parameters and summary statistics of EEG and Jact, rtol 1e-6.
An exact hash of the raw arrays is stored too, and reported, but a mismatch there alone
is not a failure.

  make : python data_generation/fingerprint_dataset.py make  <simu_folder> -out <ref.json.gz>
  check: python data_generation/fingerprint_dataset.py check <simu_folder> -ref <ref.json.gz> [-n N]

-n N checks only the first N samples, which works because seed-0 runs of any size share a
prefix (FROZEN_DATASETS.md), e.g. a smoke run's 100 samples against train8k_s0's reference.
"""

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np
from scipy.io import loadmat

RTOL = 1e-6
STAT_NAMES = ["sum", "abs_sum", "l2", "max", "min"]


def _sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    return [float(x.sum()), float(np.abs(x).sum()), float(np.sqrt((x**2).sum())), float(x.max()), float(x.min())]


def _load_struct_field(path, outer, inner):
    return np.asarray(loadmat(path)[outer][inner][0, 0], dtype=np.float64)


def sample_record(simu_folder, key, md_name):
    num = md_name.split("_")[0]
    with open(os.path.join(simu_folder, "md", md_name), "r") as f:
        md = json.load(f)
    eeg = _load_struct_field(os.path.join(simu_folder, "eeg", "infdb", f"{num}_eeg.mat"), "eeg_data", "EEG")
    jact = _load_struct_field(os.path.join(simu_folder, "sources", "Jact", f"{num}_src_act.mat"), "Jact", "Jact")

    structure = {k: md[k] for k in ("id", "seeds", "orders", "n_patch", "act_src")}
    erp = [float(e[k]) for e in md.get("erp", []) for k in ("ampl", "width", "center")]
    md_floats = [float(md[k]) for k in ("base_center", "base_width", "base_amplitude") if k in md] + erp
    return {
        "key": key,
        "structure": _sha(json.dumps(structure, sort_keys=True).encode()),
        "md_floats": md_floats,
        "eeg_shape": list(eeg.shape),
        "eeg_stats": _stats(eeg),
        "jact_shape": list(jact.shape),
        "jact_stats": _stats(jact),
        "exact": _sha(json.dumps(md, sort_keys=True).encode() + eeg.tobytes() + jact.tobytes()),
    }


def fingerprint(simu_folder, n=None):
    simu_name = os.path.basename(os.path.normpath(simu_folder))
    match_files = [f for f in os.listdir(simu_folder) if f.endswith("_match_json_file.json")]
    if len(match_files) != 1:
        raise SystemExit(f"expected one *_match_json_file.json in {simu_folder}, found {match_files}")
    with open(os.path.join(simu_folder, match_files[0]), "r") as f:
        match = json.load(f)
    config_files = [f for f in os.listdir(simu_folder) if f.endswith("_config.json")]
    with open(os.path.join(simu_folder, config_files[0]), "r") as f:
        config = json.load(f)

    keys = list(match.keys())  # loader order: the train/validation split indexes into this list
    if n is not None:
        keys = keys[:n]
    samples = []
    for i, key in enumerate(keys):
        md_name = os.path.basename(match[key]["md_json_file_name"].replace("\\", "/"))
        samples.append(sample_record(simu_folder, key, md_name))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} samples fingerprinted", flush=True)
    return {
        "simu_name": simu_name,
        "n_samples": len(match),
        "rec_info": config.get("rec_info"),
        "n_sources": config["source_space"].get("n_sources"),
        "n_electrodes": config["electrode_space"].get("n_electrodes"),
        "rtol": RTOL,
        "samples": samples,
    }


def compare(ref, new, n=None):
    problems = []
    for k in ("rec_info", "n_sources", "n_electrodes"):
        if ref[k] != new[k]:
            problems.append(f"config {k}: reference {ref[k]} vs rebuild {new[k]}")
    ref_s = ref["samples"][:n] if n is not None else ref["samples"]
    new_s = new["samples"]
    if n is None and ref["n_samples"] != new["n_samples"]:
        problems.append(f"sample count: reference {ref['n_samples']} vs rebuild {new['n_samples']}")
    if len(new_s) < len(ref_s):
        problems.append(f"rebuild has {len(new_s)} samples, need {len(ref_s)}")

    n_exact = 0
    for i, (a, b) in enumerate(zip(ref_s, new_s)):
        bad = []
        for field in ("key", "structure", "eeg_shape", "jact_shape"):
            if a[field] != b[field]:
                bad.append(field)
        for field in ("md_floats", "eeg_stats", "jact_stats"):
            if len(a[field]) != len(b[field]) or not np.allclose(a[field], b[field], rtol=RTOL, atol=0):
                bad.append(field)
        if bad:
            problems.append(f"sample {i} ({a['key']}): {', '.join(bad)} differ")
        n_exact += a["exact"] == b["exact"]
    return problems, n_exact, len(ref_s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["make", "check"])
    p.add_argument("simu_folder", help=".../simu/<simu_name>")
    p.add_argument("-out", help="make: reference file to write (.json.gz)")
    p.add_argument("-ref", help="check: reference file to compare against")
    p.add_argument("-n", type=int, default=None, help="check: only the first N samples (seed-0 prefix)")
    args = p.parse_args()

    if args.mode == "make":
        fp = fingerprint(args.simu_folder)
        with gzip.open(args.out, "wt", encoding="utf-8") as f:
            json.dump(fp, f, separators=(",", ":"))
        print(f"wrote {args.out}: {fp['simu_name']}, {fp['n_samples']} samples")
        return

    with gzip.open(args.ref, "rt", encoding="utf-8") as f:
        ref = json.load(f)
    new = fingerprint(args.simu_folder, n=args.n)
    problems, n_exact, n_checked = compare(ref, new, n=args.n)
    result = {
        "reference": ref["simu_name"],
        "rebuild": new["simu_name"],
        "checked": n_checked,
        "exact_array_matches": n_exact,
        "passed": not problems,
        "problems": problems[:20],
        "n_problems": len(problems),
    }
    print(json.dumps(result, indent=2))
    if problems:
        sys.exit(f"FINGERPRINT MISMATCH: {len(problems)} problem(s); do not train on this rebuild")


if __name__ == "__main__":
    main()
