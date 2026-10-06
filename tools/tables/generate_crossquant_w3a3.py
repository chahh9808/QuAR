#!/usr/bin/env python3
"""Matched-bit-width cross-quantizer comparison at W3A3: AdaLog vs PTQ4ViT.

Reads runs/last_crossquant_w3a3_dir.txt -> per-cell <quant>_<method>/<algo>_<quant>_<method>/
corruption_metrics.csv (15 ImageNet-C corruptions). Main number per (quantizer, method) = mean
top-1 over the 15 corruptions. Shows Ours' gain over No-Adapt is consistent across the underlying
PTQ method (the quantizer-generalization claim), at a FIXED bit-width. Emits
outputs/crossquant_w3a3_accuracy.{tex,md,json}. No ECE. Ours quar_local_lambda=0.5, EMA=0.05.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, glob, json, os

OUT = f"{_REPO_ROOT}/outputs"
QUANTS = [("adalog", "AdaLog"), ("ptq4vit", "PTQ4ViT")]
# 15 standard ImageNet-C corruptions
CORRS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
         'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
         'elastic_transform', 'pixelate', 'jpeg_compression']
METHODS = [("NOADAPT", "no_adapt", "No Adapt"), ("T3A", "t3a", "T3A"),
           ("FOA", "foa", "FOA"), ("FOZO", "fozo", "FOZO"), ("ZOA", "zoa", "ZOA"),
           ("NEO", "no_adapt", "NEO"), ("OURS", "no_adapt", "\\textbf{Ours}")]
TAGS = [m[0] for m in METHODS]; ALGO = {m[0]: m[1] for m in METHODS}; LABEL = {m[0]: m[2] for m in METHODS}
PLAIN = {t: LABEL[t].replace("\\textbf{", "").replace("}", "") for t in TAGS}


def run_dir():
    p = f"{_REPO_ROOT}/runs/last_crossquant_w3a3_dir.txt"
    if os.path.exists(p):
        b = open(p).read().strip()
        if b and os.path.isdir(b):
            return b
    c = sorted(glob.glob(f"{_REPO_ROOT}/runs/crossquant_w3a3_2*"))
    return c[-1] if c else None


def parse(rd, q, tag):
    csvp = os.path.join(rd, f"{q}_{tag}", f"{ALGO[tag]}_{q}_{tag}", "corruption_metrics.csv")
    per = {}
    if not os.path.exists(csvp):
        return per, csvp
    for r in csv.DictReader(open(csvp)):
        if r["corruption"] in CORRS:
            per[r["corruption"]] = round(float(r["top1"]), 4)
    return per, csvp


def mean15(per):
    return round(sum(per[c] for c in CORRS) / len(CORRS), 4) if all(c in per for c in CORRS) else None


def collect():
    rd = run_dir()
    data = {q: {} for q, _ in QUANTS}; prov = {q: {} for q, _ in QUANTS}
    for q, _ in QUANTS:
        for tag in TAGS:
            per, csvp = parse(rd, q, tag)
            data[q][tag] = mean15(per); prov[q][tag] = csvp
            print(f"  {q}/{tag}: mean15={data[q][tag]} ({len(per)}/15)")
    return rd, data, prov


def emit_md(rd, data):
    L = ["# Cross-quantizer comparison @ W3A3 (matched bit-width) — AdaLog vs PTQ4ViT\n"]
    L.append("**Setting:** quantized ViT-B | **fixed W3A3** | ImageNet-C 15 corruptions | 5k subset | "
             "bs=64 | seed 0 | level 5 | episodic | **no ECE**. Same protocol for both quantizers; only "
             "the PTQ method differs. Ours uses per-quantizer source stats (quar_local_lambda=0.5, EMA 0.05). "
             "Tests whether Ours' gain is consistent regardless of the underlying quantizer.\n")
    L.append("## Mean top-1 accuracy (%) over 15 corruptions\n")
    L.append("| Method | AdaLog@W3A3 | PTQ4ViT@W3A3 |")
    L.append("|--------|-----:|-----:|")
    best = {q: max([data[q][t] for t in TAGS if data[q][t] is not None], default=None) for q, _ in QUANTS}
    for tag in TAGS:
        lbl = "**Ours**" if tag == "OURS" else LABEL[tag]
        cells = []
        for q, _ in QUANTS:
            v = data[q][tag]
            if v is None: cells.append("--")
            else: cells.append(f"**{v:.2f}**" if (best[q] is not None and abs(v - best[q]) < 1e-6) else f"{v:.2f}")
        L.append(f"| {lbl} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Analysis\n")
    for q, ql in QUANTS:
        na, ou = data[q]["NOADAPT"], data[q]["OURS"]
        if na is not None and ou is not None:
            present = [(t, data[q][t]) for t in TAGS if data[q][t] is not None]
            ranked = sorted(present, key=lambda x: -x[1])
            r = next((i + 1 for i, (t, _) in enumerate(ranked) if t == "OURS"), None)
            L.append(f"- **{ql}@W3A3:** No Adapt {na:.2f} -> Ours {ou:.2f} (**{ou-na:+.2f}** pp), Ours rank #{r}/7.")
    L.append("")
    L.append("> Quantizer-generalization read: compare Ours' Δ-over-No-Adapt across the two quantizers. "
             "Consistent positive Δ on both = the recalibration helps independent of the PTQ method. "
             "Report honestly (note PTQ4ViT is not designed for 4-bit, so its absolute accuracy is lower).\n")
    L.append(f"Run dir: `{rd}`.\n")
    return "\n".join(L)


def emit_tex(data):
    best = {q: max([data[q][t] for t in TAGS if data[q][t] is not None], default=None) for q, _ in QUANTS}
    rows = []
    for tag in TAGS:
        cells = []
        for q, _ in QUANTS:
            v = data[q][tag]
            if v is None: cells.append("--")
            else: cells.append(f"\\textbf{{{v:.2f}}}" if (best[q] is not None and abs(v - best[q]) < 1e-6) else f"{v:.2f}")
        rows.append(f"  {LABEL[tag]} & " + " & ".join(cells) + " \\\\")
    return """\\begin{table}[t]
\\centering
\\caption{Cross-quantizer comparison at a fixed bit-width (W3A3): AdaLog vs PTQ4ViT, mean top-1 (\\%) over 15 ImageNet-C corruptions, quantized ViT-B, 5k subset, seed 0, episodic. Ours (Local-Global Activation Recalibration, local branch\\_alpha$=$0.5) uses per-quantizer source statistics.}
\\label{tab:crossquant_w3a3}
\\begin{tabular}{l|rr}
\\toprule
Method & AdaLog@W3A3 & PTQ4ViT@W3A3 \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def emit_json(rd, data, prov):
    return {"_about": "Cross-quantizer @ W3A3 (AdaLog vs PTQ4ViT), mean top-1 over 15 ImageNet-C corruptions. "
                      "Quantizer-generalization at fixed bit-width. Ours quar_local_lambda=0.5, EMA 0.05.",
            "run_dir": rd,
            "mean_over_15": {PLAIN[t]: {ql: data[q][t] for q, ql in QUANTS} for t in TAGS},
            "ours_vs_noadapt_delta": {ql: (data[q]["OURS"] - data[q]["NOADAPT"])
                                      if (data[q]["OURS"] is not None and data[q]["NOADAPT"] is not None) else None
                                      for q, ql in QUANTS},
            "provenance": {ql: {t: prov[q][t] for t in TAGS} for q, ql in QUANTS}}


def main():
    rd, data, prov = collect()
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/crossquant_w3a3_accuracy.tex", "w").write(emit_tex(data))
    open(f"{OUT}/crossquant_w3a3_accuracy.md", "w").write(emit_md(rd, data))
    json.dump(emit_json(rd, data, prov), open(f"{OUT}/crossquant_w3a3_accuracy.json", "w"), indent=2)
    print("Wrote outputs/crossquant_w3a3_accuracy.{tex,md,json}")


if __name__ == "__main__":
    main()
