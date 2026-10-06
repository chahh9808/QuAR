#!/usr/bin/env python3
"""Aggregate the ImageNet-C-bar sweep (10 algorithmic corruptions x 7 methods x 4 bitwidths).

ImageNet-C-bar (Mintun et al., NeurIPS 2021) = 10 algorithmically-generated corruptions chosen
to be perceptually dissimilar from the standard ImageNet-C set, hence a stronger test of
generalization to a DIFFERENT corruption family. Evaluated at severity level 5 on the
RobustBench 5k subset (1000-class, no mask).

Reads runs/last_cbar_dir.txt (fallback: newest runs/cbar_2*) -> per-cell corruption_metrics.csv
(10 rows). Emits outputs/cbar_accuracy.{tex,json,md}. Main number per (bitwidth, method) =
mean top-1 over the 10 corruptions. No ECE. Ours uses quar_local_lambda=0.5 (deployed default), EMA=0.05.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, glob, json, os

OUT = f"{_REPO_ROOT}/outputs"
BWS = ["w3a3", "w4a4", "w6a6", "w8a8"]
BW_LABEL = {"w3a3": "W3A3", "w4a4": "W4A4", "w6a6": "W6A6", "w8a8": "W8A8"}
CORRS = ["caustic_refraction", "inverse_sparkles", "sparkles", "perlin_noise",
         "blue_noise_sample", "plasma_noise", "checkerboard_cutout",
         "cocentric_sine_waves", "single_frequency_greyscale", "brownish_noise"]
CORR_LABEL = {"caustic_refraction": "Caustic", "inverse_sparkles": "Inv.spark",
              "sparkles": "Sparkles", "perlin_noise": "Perlin", "blue_noise_sample": "Blue noise",
              "plasma_noise": "Plasma", "checkerboard_cutout": "Checkerbd",
              "cocentric_sine_waves": "Sine wave", "single_frequency_greyscale": "Single freq",
              "brownish_noise": "Brown noise"}
METHODS = [("NOADAPT", "no_adapt", "No Adapt"), ("T3A", "t3a", "T3A"),
           ("FOA", "foa", "FOA"), ("FOZO", "fozo", "FOZO"), ("ZOA", "zoa", "ZOA"),
           ("NEO", "no_adapt", "NEO"), ("OURS", "no_adapt", "\\textbf{Ours}")]
TAGS = [m[0] for m in METHODS]
ALGO = {m[0]: m[1] for m in METHODS}
LABEL = {m[0]: m[2] for m in METHODS}
PLAIN = {t: LABEL[t].replace("\\textbf{", "").replace("}", "") for t in TAGS}


def run_dir():
    p = f"{_REPO_ROOT}/runs/last_cbar_dir.txt"
    if os.path.exists(p):
        b = open(p).read().strip()
        if b and os.path.isdir(b):
            return b
    c = sorted(glob.glob(f"{_REPO_ROOT}/runs/cbar_2*"))
    return c[-1] if c else None


def parse(rd, bw, tag):
    csvp = os.path.join(rd, f"{bw}_{tag}", f"{ALGO[tag]}_{bw}_{tag}", "corruption_metrics.csv")
    per = {}
    if not os.path.exists(csvp):
        return per, csvp
    for r in csv.DictReader(open(csvp)):
        if r["corruption"] in CORRS:
            per[r["corruption"]] = round(float(r["top1"]), 4)
    return per, csvp


def meanN(per):
    if all(c in per for c in CORRS):
        return round(sum(per[c] for c in CORRS) / len(CORRS), 4)
    return None


def collect():
    rd = run_dir()
    per = {bw: {} for bw in BWS}
    mean = {bw: {} for bw in BWS}
    prov = {bw: {} for bw in BWS}
    for bw in BWS:
        for tag in TAGS:
            p, csvp = parse(rd, bw, tag)
            per[bw][tag] = p
            mean[bw][tag] = meanN(p)
            prov[bw][tag] = csvp
            print(f"  {bw}/{tag}: mean10={mean[bw][tag]} ({len(p)}/10 corruptions)")
    return rd, per, mean, prov


def best_per_bw(mean):
    return {bw: max([v for v in mean[bw].values() if v is not None], default=None) for bw in BWS}


def fmt(v, best=None):
    if v is None:
        return "--"
    s = f"{v:.2f}"
    return f"\\textbf{{{s}}}" if (best is not None and abs(v - best) < 1e-6) else s


def emit_tex(mean):
    best = best_per_bw(mean)
    rows = []
    for tag in TAGS:
        cells = [fmt(mean[bw][tag], best[bw]) for bw in BWS]
        rows.append(f"  {LABEL[tag]} & " + " & ".join(cells) + " \\\\")
    return """\\begin{table}[t]
\\centering
\\caption{Top-1 accuracy (\\%) on ImageNet-C-bar (mean over 10 algorithmic corruptions of Mintun et al. 2021, perceptually dissimilar from the standard ImageNet-C set; severity level 5) with quantized ViT-B, RobustBench 5k subset (1000-class, no mask), batch size 64, seed 0, episodic reset. Ours is Local-Global Activation Recalibration with the deployed default local branch\\_alpha$=$0.5, EMA $\\alpha{=}0.05$.}
\\label{tab:cbar}
\\begin{tabular}{l|rrrr}
\\toprule
Method & W3A3 & W4A4 & W6A6 & W8A8 \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def emit_json(rd, per, mean, prov):
    best = best_per_bw(mean)
    return {
        "_about": "ImageNet-C-bar (10 algorithmic corruptions) top-1 accuracy, 7 methods x 4 bitwidths. "
                  "Main number = mean top-1 over the 10 corruptions at severity level 5. "
                  "Ours quar_local_lambda=0.5 (deployed default), EMA alpha=0.05.",
        "setting": {"dataset": "ImageNet-C-bar (10 corruptions), severity level 5, RobustBench 5k subset, 1000-class (no mask)",
                    "corruptions": CORRS, "severity_level": 5, "batch_size": 64, "seed": 0,
                    "episodic": True, "ece_reported": False,
                    "ours_config": "quar_local_lambda=0.5, quar_local_stream_alpha=0.05, quar_global_alpha=1.0, cls EMA=0.05, explicit/qba_valid/lagged/no-warmup"},
        "run_dir": rd,
        "best_per_bitwidth_mean10": {BW_LABEL[bw]: best[bw] for bw in BWS},
        "mean_over_10": {PLAIN[t]: {BW_LABEL[bw]: mean[bw][t] for bw in BWS} for t in TAGS},
        "per_corruption": {PLAIN[t]: {BW_LABEL[bw]: per[bw][t] for bw in BWS} for t in TAGS},
        "provenance": {BW_LABEL[bw]: {t: prov[bw][t] for t in TAGS} for bw in BWS},
    }


def emit_md(rd, per, mean):
    best = best_per_bw(mean)
    bestm = {bw: next((PLAIN[t] for t in TAGS
                       if mean[bw][t] is not None and abs(mean[bw][t] - best[bw]) < 1e-6), None)
             for bw in BWS}

    def mark(v, b):
        if v is None: return "--"
        return f"**{v:.2f}**" if (b is not None and abs(v - b) < 1e-6) else f"{v:.2f}"

    L = []
    L.append("# ImageNet-C-bar — baselines vs Ours (all bit-widths)\n")
    L.append("**Setting:** quantized ViT-B | ImageNet-C-bar = 10 algorithmic corruptions "
             "(Mintun et al., NeurIPS 2021), chosen to be perceptually dissimilar from the "
             "standard ImageNet-C set | severity level 5 | RobustBench 5k subset | "
             "**1000-class (no mask)** | bs=64 | seed 0 | episodic | **no ECE**. C-bar images "
             "generated on the 5k subset (Resize256+CenterCrop224 then corruption), float "
             "severities remapped to integer levels 1-5; level 5 = highest severity.\n")
    L.append("**Ours config:** Local-Global Activation Recalibration, deployed default **quar_local_lambda=0.5**, "
             "EMA alpha=0.05 (qac_stream + cls), quar_global_alpha=1.0, explicit / qba_valid / lagged / "
             "no warm-up.\n")
    L.append("## Mean top-1 accuracy (%) over the 10 C-bar corruptions\n")
    L.append("| Method | W3A3 | W4A4 | W6A6 | W8A8 |")
    L.append("|--------|-----:|-----:|-----:|-----:|")
    for tag in TAGS:
        lbl = "**Ours**" if tag == "OURS" else LABEL[tag]
        L.append(f"| {lbl} | " + " | ".join(mark(mean[bw][tag], best[bw]) for bw in BWS) + " |")
    L.append("")

    na = {bw: mean[bw]["NOADAPT"] for bw in BWS}
    ou = {bw: mean[bw]["OURS"] for bw in BWS}
    L.append("## Analysis\n")
    L.append("**Best method per bit-width (mean over 10):** " +
             ", ".join(f"{BW_LABEL[bw]}: {bestm[bw]} ({best[bw]:.2f})" for bw in BWS if best[bw] is not None) + ".\n")
    L.append("**Ours vs No Adapt (Δ pp, mean over 10):** " +
             ", ".join(f"{BW_LABEL[bw]}: {ou[bw]-na[bw]:+.2f}" for bw in BWS
                       if ou[bw] is not None and na[bw] is not None) + ".\n")
    L.append("**Ours rank (of 7) per bit-width:** ")
    ranks = []
    for bw in BWS:
        present = [(t, mean[bw][t]) for t in TAGS if mean[bw][t] is not None]
        ranked = sorted(present, key=lambda x: -x[1])
        r = next((i + 1 for i, (t, _) in enumerate(ranked) if t == "OURS"), None)
        if r is None:
            ranks.append(f"{BW_LABEL[bw]}: n/a")
        elif r == 1:
            ranks.append(f"{BW_LABEL[bw]}: #1 (best)")
        else:
            gap = ranked[0][1] - mean[bw]["OURS"]
            ranks.append(f"{BW_LABEL[bw]}: #{r} ({-gap:+.2f} vs best)")
    L.append(", ".join(ranks) + ".\n")

    L.append("## Appendix: per-corruption top-1 accuracy (%)\n")
    for bw in BWS:
        L.append(f"### {BW_LABEL[bw]}\n")
        L.append("| Method | " + " | ".join(CORR_LABEL[c] for c in CORRS) + " | Mean |")
        L.append("|--------|" + "------:|" * (len(CORRS) + 1))
        for tag in TAGS:
            lbl = "**Ours**" if tag == "OURS" else LABEL[tag]
            cells = []
            for c in CORRS:
                v = per[bw][tag].get(c)
                cells.append("--" if v is None else f"{v:.2f}")
            m = mean[bw][tag]
            cells.append("--" if m is None else f"{m:.2f}")
            L.append(f"| {lbl} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("> Report the standing honestly: state where Ours helps and where it does not, "
             "and do not spin a negative result as a win. quar_local_lambda=0.5 per the current tuned "
             "default (consistent with the updated paper tables).\n")
    L.append(f"Run dir: `{rd}`. Per-cell CSVs in the provenance section of the JSON.\n")
    return "\n".join(L)


def main():
    rd, per, mean, prov = collect()
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/cbar_accuracy.tex", "w").write(emit_tex(mean))
    json.dump(emit_json(rd, per, mean, prov), open(f"{OUT}/cbar_accuracy.json", "w"), indent=2)
    open(f"{OUT}/cbar_accuracy.md", "w").write(emit_md(rd, per, mean))
    print("Wrote outputs/cbar_accuracy.{tex,json,md}")


if __name__ == "__main__":
    main()
