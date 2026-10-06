#!/usr/bin/env python3
"""3-seed aggregation for the OOD appendix datasets (ImageNet-C-bar, ImageNet-3DCC, ImageNet-V2).

For each dataset, reads the seed-0/1/2 run dirs (runs/last_<ds>_seed{0,1,2}_dir.txt), computes the
per-seed scalar per (method, bitwidth) -- mean-over-corruptions for cbar/3dcc, the single v2 top-1
for v2 -- then aggregates across seeds as mean +/- sample std (ddof=1). Reuses the validated
parse()/meanN()/CORRS from generate_{cbar,3dcc,v2}.py. Emits outputs/<ds>_3seed_accuracy.{tex,md,json}.

Source calibration stats are fixed at _seed0 for every seed (project convention); only --seed
(hence the if_shuffle batch order) varies, so No Adapt shows ~0 variance and streaming/stochastic
methods show real variance.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import importlib.util, json, os, statistics

OUT = f"{_REPO_ROOT}/outputs"
SEEDS = [0, 1, 2]
BWS = ["w3a3", "w4a4", "w6a6", "w8a8"]
BW_LABEL = {"w3a3": "W3A3", "w4a4": "W4A4", "w6a6": "W6A6", "w8a8": "W8A8"}
DATASETS = [
    ("heldout_c", "generate_heldout_c.py", "Held-out ImageNet-C (4 corruptions, sev5; mean over 4)"),
    ("cbar", "generate_cbar.py", "ImageNet-C-bar (10 algorithmic corruptions, sev5; mean over 10)"),
    ("3dcc", "generate_3dcc.py", "ImageNet-3DCC (all 12 corruptions, sev5; mean over 12)"),
    ("v2",   "generate_v2.py",   "ImageNet-V2 matched-frequency (10k, full eval; single top-1)"),
]


def load(modfile):
    # A missing per-dataset generator skips that dataset rather than failing the others.
    if not os.path.exists(f"{_REPO_ROOT}/tools/{modfile}"):
        print(f"[skip] {modfile} is not in tools/")
        return None
    spec = importlib.util.spec_from_file_location(modfile[:-3], f"{_REPO_ROOT}/tools/{modfile}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def seed_dir(ds, seed):
    p = f"{_REPO_ROOT}/runs/last_{ds}_seed{seed}_dir.txt"
    if os.path.exists(p):
        d = open(p).read().strip()
        if d and os.path.isdir(d):
            return d
    return None


def per_seed_value(ds, gen, rd, bw, tag):
    """Scalar for one (seed, method, bw): mean-over-corruptions (cbar/3dcc) or v2 top-1."""
    if rd is None:
        return None
    if ds == "v2":
        v, _ = gen.parse(rd, bw, tag)
        return v
    per, _ = gen.parse(rd, bw, tag)
    meanfn = getattr(gen, "meanN", None) or getattr(gen, "mean4", None)  # cbar/3dcc use meanN; heldout uses mean4
    return meanfn(per)


def agg(vals):
    """vals: list of per-seed scalars (None dropped). Returns (mean, std, n, kept)."""
    kept = [v for v in vals if v is not None]
    if not kept:
        return None, None, 0, []
    mean = sum(kept) / len(kept)
    std = statistics.stdev(kept) if len(kept) >= 2 else 0.0
    return round(mean, 4), round(std, 4), len(kept), kept


def collect(ds, gen):
    dirs = {s: seed_dir(ds, s) for s in SEEDS}
    TAGS = gen.TAGS
    out = {bw: {} for bw in BWS}      # out[bw][tag] = (mean,std,n)
    per = {bw: {} for bw in BWS}      # per[bw][tag] = {seed: val}
    for bw in BWS:
        for tag in TAGS:
            vals = {s: per_seed_value(ds, gen, dirs[s], bw, tag) for s in SEEDS}
            m, sd, n, _ = agg([vals[s] for s in SEEDS])
            out[bw][tag] = (m, sd, n)
            per[bw][tag] = vals
    return dirs, out, per, TAGS, gen


def best_per_bw(out, TAGS):
    best = {}
    for bw in BWS:
        ms = [out[bw][t][0] for t in TAGS if out[bw][t][0] is not None]
        best[bw] = max(ms) if ms else None
    return best


def cell(out, bw, tag, best, latex=False):
    m, sd, n = out[bw][tag]
    if m is None:
        return "--"
    bold = best[bw] is not None and abs(m - best[bw]) < 1e-6
    if latex:
        s = f"{m:.2f}$\\pm${sd:.2f}"
        return f"\\textbf{{{s}}}" if bold else s
    s = f"{m:.2f}±{sd:.2f}"
    return f"**{s}**" if bold else s


def plain(gen, tag):
    return gen.LABEL[tag].replace("\\textbf{", "").replace("}", "")


def emit_md(ds, about, dirs, out, per, TAGS, gen):
    best = best_per_bw(out, TAGS)
    bestm = {bw: next((plain(gen, t) for t in TAGS
                       if out[bw][t][0] is not None and abs(out[bw][t][0] - best[bw]) < 1e-6), None)
             for bw in BWS}
    L = [f"# {about} — 3-seed (mean ± std over seeds 0,1,2)\n"]
    L.append("**Setting:** quantized ViT-B | 7 methods × 4 bit-widths | bs=64 | episodic | **no ECE** | "
             "seeds {0,1,2}. Source calibration stats fixed at seed0 for all seeds (project convention); "
             "only `--seed` varies (if_shuffle=True ⇒ different batch order). Ours quar_local_lambda=0.5, EMA 0.05.\n")
    L.append("## Mean top-1 accuracy (%) ± std over 3 seeds\n")
    L.append("| Method | W3A3 | W4A4 | W6A6 | W8A8 |")
    L.append("|--------|-----:|-----:|-----:|-----:|")
    for t in TAGS:
        lbl = "**Ours**" if t == "OURS" else gen.LABEL[t]
        L.append(f"| {lbl} | " + " | ".join(cell(out, bw, t, best) for bw in BWS) + " |")
    L.append("")
    na = {bw: out[bw]["NOADAPT"][0] for bw in BWS}
    ou = {bw: out[bw]["OURS"][0] for bw in BWS}
    L.append("## Analysis\n")
    L.append("**Best method per bit-width (by 3-seed mean):** " +
             ", ".join(f"{BW_LABEL[bw]}: {bestm[bw]} ({best[bw]:.2f})" for bw in BWS if best[bw] is not None) + ".\n")
    L.append("**Ours vs No Adapt (Δ mean pp):** " +
             ", ".join(f"{BW_LABEL[bw]}: {ou[bw]-na[bw]:+.2f}" for bw in BWS
                       if ou[bw] is not None and na[bw] is not None) + ".\n")
    ranks = []
    for bw in BWS:
        present = [(t, out[bw][t][0]) for t in TAGS if out[bw][t][0] is not None]
        ranked = sorted(present, key=lambda x: -x[1])
        r = next((i + 1 for i, (t, _) in enumerate(ranked) if t == "OURS"), None)
        if r is None:
            ranks.append(f"{BW_LABEL[bw]}: n/a")
        elif r == 1:
            ranks.append(f"{BW_LABEL[bw]}: #1")
        else:
            ranks.append(f"{BW_LABEL[bw]}: #{r} ({out[bw]['OURS'][0]-ranked[0][1]:+.2f} vs best)")
    L.append("**Ours rank (of 7, by mean) per bit-width:** " + ", ".join(ranks) + ".\n")
    # seed-count sanity
    nmap = {(bw, t): out[bw][t][2] for bw in BWS for t in TAGS}
    full = all(n == 3 for n in nmap.values())
    L.append(f"**Seed coverage:** {'all cells have 3 seeds' if full else 'WARNING: some cells <3 seeds'} "
             f"(min={min(nmap.values())}, max={max(nmap.values())}).\n")
    L.append("## Per-seed values (top-1 %)\n")
    for bw in BWS:
        L.append(f"### {BW_LABEL[bw]}\n")
        L.append("| Method | seed0 | seed1 | seed2 | mean±std |")
        L.append("|--------|------:|------:|------:|------:|")
        for t in TAGS:
            lbl = "**Ours**" if t == "OURS" else gen.LABEL[t]
            vs = per[bw][t]
            cells = [("--" if vs[s] is None else f"{vs[s]:.2f}") for s in SEEDS]
            m, sd, n = out[bw][t]
            ms = "--" if m is None else f"{m:.2f}±{sd:.2f}"
            L.append(f"| {lbl} | " + " | ".join(cells) + f" | {ms} |")
        L.append("")
    L.append("Run dirs: " + ", ".join(f"seed{s}=`{dirs[s]}`" for s in SEEDS) + ".\n")
    return "\n".join(L)


def emit_tex(ds, about, out, TAGS, gen):
    best = best_per_bw(out, TAGS)
    rows = [f"  {gen.LABEL[t]} & " + " & ".join(cell(out, bw, t, best, latex=True) for bw in BWS) + " \\\\" for t in TAGS]
    return ("""\\begin{table}[t]
\\centering
\\caption{Top-1 accuracy (\\%, mean$\\pm$std over 3 seeds) on """ + about.replace("&", "\\&") + """ with quantized ViT-B across bit-widths, batch size 64, episodic reset. Source calibration statistics fixed across seeds; only the random seed (batch order) varies. Ours is Local-Global Activation Recalibration (local branch\\_alpha$=$0.5, EMA $\\alpha{=}0.05$).}
\\label{tab:""" + ds + """_3seed}
\\begin{tabular}{l|rrrr}
\\toprule
Method & W3A3 & W4A4 & W6A6 & W8A8 \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")


def emit_json(ds, about, dirs, out, per, TAGS, gen):
    best = best_per_bw(out, TAGS)
    return {
        "_about": f"3-seed (seeds 0,1,2) aggregation for {about}. mean+/-std (sample, ddof=1) of the "
                  "per-seed scalar. Source stats fixed at seed0; only --seed varies.",
        "seeds": SEEDS,
        "run_dirs": {f"seed{s}": dirs[s] for s in SEEDS},
        "best_per_bitwidth_mean": {BW_LABEL[bw]: best[bw] for bw in BWS},
        "mean_std": {plain(gen, t): {BW_LABEL[bw]: {"mean": out[bw][t][0], "std": out[bw][t][1], "n_seeds": out[bw][t][2]}
                                     for bw in BWS} for t in TAGS},
        "per_seed": {plain(gen, t): {BW_LABEL[bw]: {f"seed{s}": per[bw][t][s] for s in SEEDS}
                                     for bw in BWS} for t in TAGS},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for ds, modfile, about in DATASETS:
        gen = load(modfile)
        if gen is None:
            continue
        dirs, out, per, TAGS, gen = collect(ds, gen)
        miss = [(bw, t) for bw in BWS for t in TAGS if out[bw][t][2] < 3]
        print(f"[{ds}] seed dirs: " + ", ".join(f"s{s}={'ok' if dirs[s] else 'MISSING'}" for s in SEEDS) +
              f" | cells with <3 seeds: {len(miss)}")
        open(f"{OUT}/{ds}_3seed_accuracy.tex", "w").write(emit_tex(ds, about, out, TAGS, gen))
        open(f"{OUT}/{ds}_3seed_accuracy.md", "w").write(emit_md(ds, about, dirs, out, per, TAGS, gen))
        json.dump(emit_json(ds, about, dirs, out, per, TAGS, gen),
                  open(f"{OUT}/{ds}_3seed_accuracy.json", "w"), indent=2)
        summary.append((ds, len(miss)))
    print("Wrote outputs/{cbar,3dcc,v2}_3seed_accuracy.{tex,md,json}")
    for ds, miss in summary:
        print(f"  {ds}: {'COMPLETE (3 seeds all cells)' if miss == 0 else f'{miss} cells <3 seeds'}")


if __name__ == "__main__":
    main()
