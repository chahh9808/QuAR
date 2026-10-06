#!/usr/bin/env python3
"""Re-run ONLY the Ours cells of the OOD experiments at the DEPLOYED scope (block8_9_10_11_all,
blocks 8-11 + fc2, 16 modules), lambda=quar_local_lambda=0.5, EMA=quar_local_stream_alpha=0.05.
Baselines are scope-independent and are left untouched.

For each existing w{bw}_OURS.log we parse its Namespace to recover the EXACT eval config (data,
corruption set, level, 5k indices, quant checkpoint, seed, output/tag) and re-issue main.py with
ONLY the local branch/CLS flags swapped to the canonical B3 config. The old w{bw}_OURS subdir is backed up
first, then overwritten in place so the existing report generators pick up the corrected numbers.

Usage:
  python rerun_ood_ours_b3.py --dry-run           # print reconstructed commands, run nothing
  python rerun_ood_ours_b3.py --only 3dcc_..._w6a6 # run a single matching cell (smoke test)
  python rerun_ood_ours_b3.py                      # run all, 8-way concurrent
"""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import argparse, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

PY = "python3"
BACKUP = os.environ.get("TMPDIR", "/tmp") + "/ours_old_backup_b3"
NGPU = 8
# Per-dataset paths for the two sets not covered by the Namespace defaults
V2_PATH = os.environ.get("DATA_V2", "")
A_PATH = os.environ.get("DATA_A", "")

# Target run dirs (from runs/last_*_dir.txt), per dataset+seed. imagenet_a is seed0-only.
TARGET_DIRS = [
    f"{_REPO_ROOT}/runs/heldout_c_20260612_005130",
    f"{_REPO_ROOT}/runs/heldout_c_seed1_20260613_144731",
    f"{_REPO_ROOT}/runs/heldout_c_seed2_20260613_150748",
    f"{_REPO_ROOT}/runs/cbar_20260612_023016",
    f"{_REPO_ROOT}/runs/cbar_seed1_20260613_103809",
    f"{_REPO_ROOT}/runs/cbar_seed2_20260613_123730",
    f"{_REPO_ROOT}/runs/3dcc_20260612_224633",
    f"{_REPO_ROOT}/runs/3dcc_seed1_20260613_112731",
    f"{_REPO_ROOT}/runs/3dcc_seed2_20260613_132645",
    f"{_REPO_ROOT}/runs/v2_20260613_005138",
    f"{_REPO_ROOT}/runs/v2_seed1_20260613_102729",
    f"{_REPO_ROOT}/runs/v2_seed2_20260613_122650",
    f"{_REPO_ROOT}/runs/imagenet_a_20260612_001447",
]


def parse_ns(logpath):
    """Pull the relevant fields out of the Namespace(...) line of an Ours log."""
    ns = None
    with open(logpath, errors="ignore") as f:
        for line in f:
            if "Namespace(" in line:
                ns = line[line.index("Namespace("):]
                break
    if ns is None:
        return None

    def g(key, default=None):
        m = re.search(rf"\b{key}=('([^']*)'|None|True|False|-?\d+\.?\d*(?:e-?\d+)?)", ns)
        if not m:
            return default
        raw = m.group(1)
        if raw == "None":
            return None
        if raw == "True":
            return True
        if raw == "False":
            return False
        if raw.startswith("'"):
            return m.group(2)
        return raw  # numeric as string

    return {
        "data": g("data"),
        "data_corruption": g("data_corruption"),
        "corruption": g("corruption"),
        "level": g("level"),
        "seed": g("seed"),
        "quant_w_bit": g("quant_w_bit"),
        "quant_a_bit": g("quant_a_bit"),
        "adalog_checkpoint": g("adalog_checkpoint"),
        "ptq_checkpoint": g("ptq_checkpoint"),
        "imagenet_5k_eval": g("imagenet_5k_eval"),
        "ids": g("imagenet_5k_indices_file"),
        "cmap": g("imagenet_class_map_file"),
        "output": g("output"),
    }


def build_cmd(ns, gpu):
    w, a = ns["quant_w_bit"], ns["quant_a_bit"]
    bw = f"w{w}a{a}"
    out_parent = os.path.dirname(ns["output"].rstrip("/"))   # .../w6a6_OURS/no_adapt_.. -> .../w6a6_OURS
    tag = "_" + os.path.basename(out_parent)                 # _w6a6_OURS
    cmd = [PY, "-u", f"{_REPO_ROOT}/main.py", "--data", ns["data"],
           "--batch_size", "64", "--workers", "8", "--seed", str(ns["seed"]),
           "--quant", "--quant_w_bit", str(w), "--quant_a_bit", str(a)]
    if ns["adalog_checkpoint"]:
        cmd += ["--adalog_checkpoint", ns["adalog_checkpoint"]]
    elif ns["ptq_checkpoint"]:
        cmd += ["--ptq_checkpoint", ns["ptq_checkpoint"]]
    corr = ns["corruption"]
    if corr == "v2":
        cmd += ["--data_v2", V2_PATH, "--corruption", "v2"]
    elif corr == "imagenet_a":
        cmd += ["--data_a", A_PATH, "--corruption", "imagenet_a"]
    else:
        cmd += ["--data_corruption", ns["data_corruption"], "--level", str(ns["level"]),
                "--corruption", corr]
        if ns["imagenet_5k_eval"]:
            cmd += ["--imagenet_5k_eval", "--imagenet_5k_indices_file", ns["ids"],
                    "--imagenet_class_map_file", ns["cmap"]]
    # canonical B3 Ours = no_adapt + the local branch(block8_9_10_11_all, lambda 0.5, EMA 0.05, all-layers) + CLS(1.0)
    cmd += ["--algorithm", "no_adapt",
            "--quar_local_enable", "--quar_local_mode", "explicit", "--quar_local_scope", "block8_9_10_11_all",
            "--quar_local_stats", "mean_std", "--quar_local_eps", "1e-6", "--quar_local_clip_ratio", "3.0",
            "--quar_local_lambda", "0.5",
            "--quar_local_stats_file", f"{_REPO_ROOT}/qac_stats_{bw}_all_layers_seed0.pt",
            "--quar_local_stream_ema", "--quar_local_stream_alpha", "0.05", "--quar_local_stream_nwarm", "0", "--quar_local_stream_lag",
            "--quar_global_enable", "--quar_global_position", "post_ln", "--quar_global_mode", "mean_std",
            "--quar_global_alpha", "1.0", "--quar_global_eps", "1e-6", "--quar_global_clip_ratio", "3.0",
            "--quar_global_stats_file", f"{_REPO_ROOT}/cls_stats_{bw}_seed0.pt",
            "--quar_global_target_ema_alpha", "0.05",
            "--output", out_parent, "--tag", tag]
    return cmd, out_parent, tag, bw


def cells():
    out = []
    for d in TARGET_DIRS:
        if not os.path.isdir(d):
            print(f"WARN missing dir {d}", file=sys.stderr)
            continue
        for fn in sorted(os.listdir(d)):
            if re.fullmatch(r"w\d+a\d+_OURS\.log", fn):
                logp = os.path.join(d, fn)
                ns = parse_ns(logp)
                if ns is None:
                    print(f"WARN no Namespace in {logp}", file=sys.stderr)
                    continue
                name = f"{os.path.basename(d)}::{fn[:-4]}"
                out.append((name, d, logp, ns))
    return out


def run_one(idx, name, d, ns, dry):
    gpu = idx % NGPU
    cmd, out_parent, tag, bw = build_cmd(ns, gpu)
    if dry:
        print(f"[{name}] GPU{gpu} bw={bw}\n  {' '.join(cmd)}\n")
        return (name, "dry", 0)
    # back up old Ours subdir + log
    rel = os.path.relpath(out_parent, _REPO_ROOT).replace("/", "__")
    bdir = os.path.join(BACKUP, rel)
    os.makedirs(BACKUP, exist_ok=True)
    if os.path.isdir(out_parent) and not os.path.exists(bdir):
        shutil.move(out_parent, bdir)
    logp = os.path.join(d, os.path.basename(out_parent) + ".log")
    if os.path.exists(logp):
        shutil.copy2(logp, os.path.join(BACKUP, rel + ".oldlog"))
    os.makedirs(out_parent, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.time()
    with open(logp, "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    return (name, "ok" if rc == 0 else f"FAIL rc={rc}", time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on cell name")
    args = ap.parse_args()
    cs = cells()
    if args.only:
        cs = [c for c in cs if args.only in c[0]]
    print(f"{len(cs)} Ours cells to re-run (dry={args.dry_run}). Backup -> {BACKUP}")
    if args.dry_run:
        for i, (name, d, logp, ns) in enumerate(cs):
            run_one(i, name, d, ns, True)
        return
    results = []
    with ThreadPoolExecutor(max_workers=NGPU) as ex:
        futs = [ex.submit(run_one, i, name, d, ns, False) for i, (name, d, logp, ns) in enumerate(cs)]
        for f in futs:
            name, status, dt = f.result()
            print(f"[{status}] {name} ({dt:.0f}s)", flush=True)
            results.append((name, status))
    nfail = sum(1 for _, s in results if s != "ok")
    print(f"\nDONE: {len(results)-nfail}/{len(results)} ok, {nfail} failed.")


if __name__ == "__main__":
    main()
