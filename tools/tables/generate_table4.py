#!/usr/bin/env python3
"""Parse Table 4 (ImageNet-R / ImageNet-Sketch) results and emit LaTeX/JSON/MD.

Usage:
  python generate_table4.py [w6a6|w4a4|w8a8]
"""
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, json, os, sys

BW = sys.argv[1] if len(sys.argv) > 1 else 'w6a6'

OUTDIR = _os.path.join(_REPO_ROOT, "outputs")
LAST = f'{_REPO_ROOT}/runs/last_table4_{BW}_dir.txt'
LAST_R_RERUN = f'{_REPO_ROOT}/runs/last_table4_{BW}_R_rerun_dir.txt'
if not os.path.exists(LAST):
    print(f'MISSING: {LAST}'); sys.exit(1)
OUTBASE = open(LAST).read().strip()
OUTBASE_R_RERUN = open(LAST_R_RERUN).read().strip() if os.path.exists(LAST_R_RERUN) else None

# (tag, algorithm, display_label)
METHODS = [
    ('NOADAPT', 'no_adapt', 'No Adapt'),
    ('T3A',     't3a',      'T3A'),
    ('FOA',     'foa',      'FOA'),
    ('FOZO',    'fozo',     'FOZO'),
    ('ZOA',     'zoa',      'ZOA'),
    ('NEO',     'no_adapt', 'NEO'),
    ('OURS',    'no_adapt', '\\textbf{Ours}'),
]


def parse_csv(path):
    """Returns {corruption: top1}."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row['corruption']] = float(row['top1'])
    return out


def collect():
    """Sketch from OUTBASE; R from OUTBASE_R_RERUN (mask-fix patch) if it exists,
    else from OUTBASE.

    T3A exception: T3A's argmax-encoded supports yield 1000-dim output that
    aligns with ImageFolder labels [0..199] by construction; applying the
    imagenet_r_mask in validate_adapt breaks this alignment. For W6A6 the
    R-rerun explicitly skips T3A's masking via main.py's t3a-exception; for
    runs created BEFORE that exception (like the W6A6 initial run), T3A's R
    value is unaffected. We always prefer the T3A R value from the initial
    OUTBASE (no mask applied by validate_adapt for T3A in either case).
    """
    results = {}
    for tag, algo, label in METHODS:
        sketch_dir = os.path.join(OUTBASE, tag, f'{algo}_{tag}_{BW}')
        sketch_csv = os.path.join(sketch_dir, 'corruption_metrics.csv')
        s_data = parse_csv(sketch_csv)
        sketch_val = s_data.get('sketch') if s_data else None

        if tag == 'T3A':
            # T3A uses initial-run R value (validate_adapt's t3a-exception in
            # main.py ensures the mask is not applied; both initial and rerun
            # produce the same T3A R value when the patch was already in place).
            rendition_dir = sketch_dir
            rendition_csv = sketch_csv
            rendition_val = s_data.get('rendition') if s_data else None
        elif OUTBASE_R_RERUN:
            rendition_dir = os.path.join(OUTBASE_R_RERUN, tag, f'{algo}_{tag}_{BW}_R')
            rendition_csv = os.path.join(rendition_dir, 'corruption_metrics.csv')
            r_data = parse_csv(rendition_csv)
            rendition_val = r_data.get('rendition') if r_data else None
            if rendition_val is None:
                rendition_dir = sketch_dir; rendition_csv = sketch_csv
                rendition_val = s_data.get('rendition') if s_data else None
        else:
            rendition_dir = sketch_dir; rendition_csv = sketch_csv
            rendition_val = s_data.get('rendition') if s_data else None

        results[tag] = {
            'label': label,
            'rendition': rendition_val,
            'sketch': sketch_val,
            'run_dir_sketch': sketch_dir,
            'csv_sketch': sketch_csv,
            'run_dir_rendition': rendition_dir,
            'csv_rendition': rendition_csv,
        }
        print(f'  OK: {tag} R={rendition_val} Sketch={sketch_val}')
    return results


def emit_latex(results):
    bw_label = {'w3a3': 'W3A3', 'w4a4': 'W4A4', 'w6a6': 'W6A6', 'w8a8': 'W8A8'}[BW]
    ckpt_kind = 'PTQ4ViT' if BW in ('w6a6', 'w8a8') else 'AdaLog'
    # find best per column
    r_vals = [(t, d['rendition']) for t, d in results.items() if d['rendition'] is not None]
    s_vals = [(t, d['sketch'])    for t, d in results.items() if d['sketch']    is not None]
    a_vals = []
    for t, d in results.items():
        if d['rendition'] is not None and d['sketch'] is not None:
            a_vals.append((t, (d['rendition'] + d['sketch']) / 2))
    r_best = max((v for _, v in r_vals), default=None)
    s_best = max((v for _, v in s_vals), default=None)
    a_best = max((v for _, v in a_vals), default=None)

    def fmt(v, best):
        if v is None: return '--'
        s = f'{v:.2f}'
        return f'\\textbf{{{s}}}' if (best is not None and abs(v-best) < 1e-6) else s

    rows = []
    for tag, _, _ in METHODS:
        d = results[tag]
        r = d['rendition']; s = d['sketch']
        avg = (r + s) / 2 if (r is not None and s is not None) else None
        rows.append('  ' + d['label'] + ' & ' +
                    fmt(r, r_best) + ' & ' + fmt(s, s_best) + ' & ' + fmt(avg, a_best) + ' \\\\')

    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Top-1 accuracy on ImageNet-R and ImageNet-Sketch with {bw_label} ({ckpt_kind}) quantized ViT-B at batch size 64, seed 0, episodic reset. ImageNet-R is evaluated over its 200-class subset (with the standard \\texttt{{imagenet\\_r\\_mask}}); ImageNet-Sketch uses the full 1000-class set.; cost columns are in Table 1.}}
\\label{{tab:table4_{BW}_r_sketch}}
\\begin{{tabular}}{{l|rrr}}
\\toprule
Method & ImageNet-R & ImageNet-Sketch & Avg. \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    return tex


def emit_md(results):
    bw_label = {'w3a3': 'W3A3', 'w4a4': 'W4A4', 'w6a6': 'W6A6', 'w8a8': 'W8A8'}[BW]
    ckpt_kind = 'PTQ4ViT' if BW in ('w6a6', 'w8a8') else 'AdaLog'
    md = f"""# Table 4 ({bw_label}) — Robustness beyond ImageNet-C (ImageNet-R + ImageNet-Sketch)

**Setting:** {bw_label} {ckpt_kind} quantized ViT-B | batch size 64 | seed 0 | episodic reset
| ImageNet-R (rendition split, 200 classes with `imagenet_r_mask`) +
ImageNet-Sketch (1000 classes, full set).

**Run dir:** `{OUTBASE}`

## Top-1 accuracy (%)

| Method | ImageNet-R | ImageNet-Sketch | Avg. |
|--------|-----------:|----------------:|----:|
"""
    for tag, _, _ in METHODS:
        d = results[tag]
        r = d['rendition']; s = d['sketch']
        avg = (r + s) / 2 if (r is not None and s is not None) else None
        r_s = f'{r:.2f}' if r is not None else '--'
        s_s = f'{s:.2f}' if s is not None else '--'
        a_s = f'{avg:.2f}' if avg is not None else '--'
        label_md = '**Ours**' if tag == 'OURS' else d['label'].replace('\\textbf{', '').replace('}', '')
        md += f'| {label_md} | {r_s} | {s_s} | {a_s} |\n'

    md += """
## Provenance per number

ImageNet-R numbers (excl. T3A) come from a re-run with the
`validate_adapt`+`imagenet_r_mask` patch in main.py, so 1000-dim outputs
from raw-logit methods are correctly masked down to the 200-class label
space. ImageNet-Sketch (1000-class) does not need the mask.

T3A's R value is taken from the initial run because T3A's argmax-encoded
supports yield 1000-dim output that aligns with ImageFolder labels
[0..199] by construction; applying the mask breaks T3A.

| Method | Rendition csv | Sketch csv |
|--------|---------------|------------|
"""
    for tag, _, _ in METHODS:
        d = results[tag]
        md += f'| {tag} | `{d["csv_rendition"]}` | `{d["csv_sketch"]}` |\n'

    if BW == 'w6a6':
        ckpt = 'checkpoints/ptq4vit_w6a6_vitb.pt'
    elif BW == 'w4a4':
        ckpt = 'w4a4_checkpoint_opt.pt'
    elif BW == 'w3a3':
        ckpt = 'w3a3_checkpoint_opt.pt'
    elif BW == 'w8a8':
        ckpt = 'checkpoints/ptq4vit_w8a8_vitb.pt'
    else:
        ckpt = '<unknown>'

    md += f"""
## Configuration

- **Bitwidth:** {bw_label} ({ckpt_kind})
- **Checkpoint:** `{ckpt}`
- **Seed:** 0
- **Batch size:** 64
- **Workers:** 8
- **Datasets:**
  - ImageNet-R: `$DATA_R`
  - ImageNet-Sketch: `$DATA_SKETCH`
- **Episodic reset:** yes (no `--continual`)
- **No ECE column.**
"""

    md += """
## Notes

- No cost columns (latency / memory / updated params) — those are reported in Table 1.
- If a method's row shows `--`, the run failed; check `runs/.../<METHOD>.log`.
"""
    return md


def main():
    print(f'BW: {BW}  Outbase: {OUTBASE}')
    results = collect()

    tex = emit_latex(results)
    p = os.path.join(OUTDIR, f'table4_{BW}_r_sketch_accuracy.tex')
    open(p, 'w').write(tex); print(f'Wrote {p}')

    md = emit_md(results)
    p = os.path.join(OUTDIR, f'table4_{BW}_r_sketch_accuracy_report.md')
    open(p, 'w').write(md); print(f'Wrote {p}')

    json_obj = {
        'bitwidth': BW,
        'outbase': OUTBASE,
        'outbase_r_rerun': OUTBASE_R_RERUN,
        'setting': {
            'batch_size': 64, 'seed': 0, 'episodic_reset': True,
            'datasets': {
                'rendition': os.environ.get("DATA_R", ""),
                'sketch':    os.environ.get("DATA_SKETCH", ""),
            },
        },
        'methods': {tag: {
            'label': results[tag]['label'].replace('\\textbf{', '').replace('}', ''),
            'rendition_top1': results[tag]['rendition'],
            'sketch_top1':    results[tag]['sketch'],
            'avg_top1': ((results[tag]['rendition'] + results[tag]['sketch']) / 2
                         if results[tag]['rendition'] is not None and results[tag]['sketch'] is not None
                         else None),
            'csv_rendition': results[tag]['csv_rendition'],
            'csv_sketch':    results[tag]['csv_sketch'],
        } for tag, _, _ in METHODS},
    }
    p = os.path.join(OUTDIR, f'table4_{BW}_r_sketch_accuracy.json')
    json.dump(json_obj, open(p, 'w'), indent=2); print(f'Wrote {p}')


if __name__ == '__main__':
    main()
