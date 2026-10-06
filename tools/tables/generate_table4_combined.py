#!/usr/bin/env python3
"""Combined Table 4 (ImageNet-R + ImageNet-Sketch) across all bitwidths.

W6A6 is the representative MAIN-table setting; W3A3/W4A4/W6A6/W8A8 go to the
appendix. Reuses the validated W3A3 and W6A6 per-bitwidth JSONs; parses the
fresh W4A4 and W8A8 single-pass run dirs (rendition + sketch in one CSV).

Emits:
  outputs/table4_r_sketch_w6a6_main.tex
  outputs/table4_r_sketch_all_bitwidths_appendix.tex
  outputs/table4_r_sketch_all_bitwidths.json
  outputs/table4_r_sketch_report.md

Never fabricates: a missing number is reported as null / '--'.
"""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, glob, json, os

OUT = f'{_REPO_ROOT}/outputs'

BITWIDTHS = ['w3a3', 'w4a4', 'w6a6', 'w8a8']
BW_LABEL = {'w3a3': 'W3A3', 'w4a4': 'W4A4', 'w6a6': 'W6A6', 'w8a8': 'W8A8'}
CKPT_KIND = {'w3a3': 'AdaLog', 'w4a4': 'AdaLog', 'w6a6': 'PTQ4ViT', 'w8a8': 'PTQ4ViT'}
CKPT_PATH = {
    'w3a3': 'w3a3_checkpoint_opt.pt',
    'w4a4': 'w4a4_checkpoint_opt.pt',
    'w6a6': 'checkpoints/ptq4vit_w6a6_vitb.pt',
    'w8a8': 'checkpoints/ptq4vit_w8a8_vitb.pt',
}

# tag -> (algo subdir prefix, display label)
METHODS = [
    ('NOADAPT', 'no_adapt', 'No Adapt'),
    ('T3A',     't3a',      'T3A'),
    ('FOA',     'foa',      'FOA'),
    ('FOZO',    'fozo',     'FOZO'),
    ('ZOA',     'zoa',      'ZOA'),
    ('NEO',     'no_adapt', 'NEO'),
    ('OURS',    'no_adapt', 'Ours'),
]
TAGS = [m[0] for m in METHODS]
LABEL = {m[0]: m[2] for m in METHODS}
ALGO = {m[0]: m[1] for m in METHODS}

DATA_R = os.environ.get("DATA_R", "")
DATA_SKETCH = os.environ.get("DATA_SKETCH", "")


def parse_csv_rows(path):
    """Return {corruption: top1} or None if file missing."""
    if not path or not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[row['corruption']] = float(row['top1'])
            except (KeyError, ValueError):
                pass
    return out or None


def from_existing_json(bw):
    """Load the per-bit-width W3A3/W6A6 result jsons (current or archived location)."""
    p = f'{OUT}/table4_{bw}_r_sketch_accuracy.json'
    if not os.path.exists(p):
        p = f'{OUT}/deprecated/table4_{bw}_r_sketch_accuracy.json'
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    res = {}
    for tag in TAGS:
        m = d['methods'].get(tag, {})
        res[tag] = {
            'rendition': m.get('rendition_top1'),
            'sketch': m.get('sketch_top1'),
            'csv_rendition': m.get('csv_rendition'),
            'csv_sketch': m.get('csv_sketch'),
        }
    return res


def find_outbase(bw):
    """Newest fresh single-pass table4_<bw>_full_* dir first, else pointer file."""
    cands = sorted(glob.glob(f'{_REPO_ROOT}/runs/table4_{bw}_full_*'))
    if cands:
        return cands[-1]
    ptr = f'{_REPO_ROOT}/runs/last_table4_{bw}_dir.txt'
    if os.path.exists(ptr):
        base = open(ptr).read().strip()
        if base and os.path.isdir(base):
            return base
    return None


def from_run_dir(bw):
    """Parse fresh single-pass W4A4/W8A8 run dir (both rows in one CSV)."""
    base = find_outbase(bw)
    res = {}
    for tag in TAGS:
        csv_path = None
        if base:
            csv_path = os.path.join(base, tag, f'{ALGO[tag]}_{tag}_{bw}',
                                    'corruption_metrics.csv')
        rows = parse_csv_rows(csv_path)
        res[tag] = {
            'rendition': rows.get('rendition') if rows else None,
            'sketch': rows.get('sketch') if rows else None,
            'csv_rendition': csv_path,
            'csv_sketch': csv_path,
        }
    return res


def collect():
    data = {}
    for bw in BITWIDTHS:
        if bw in ('w3a3', 'w6a6'):
            res = from_existing_json(bw)
            src = 'reused validated JSON'
        else:
            res = from_run_dir(bw)
            src = find_outbase(bw)
        if res is None:
            print(f'  {bw}: MISSING ({src})')
            data[bw] = None
            continue
        for tag in TAGS:
            r = res[tag]['rendition']; s = res[tag]['sketch']
            res[tag]['avg'] = (r + s) / 2 if (r is not None and s is not None) else None
        data[bw] = res
        ours = res['OURS']['avg']
        print(f'  {bw}: OK (src={src}) Ours avg={ours}')
    return data


def fmt(v, best=None):
    if v is None:
        return '--'
    s = f'{v:.2f}'
    return f'\\textbf{{{s}}}' if (best is not None and abs(v - best) < 1e-6) else s


def best_in(vals):
    vv = [v for v in vals if v is not None]
    return max(vv) if vv else None


def emit_main_tex(data):
    """W6A6-only main table."""
    bw = 'w6a6'
    d = data[bw]
    r_best = best_in([d[t]['rendition'] for t in TAGS])
    s_best = best_in([d[t]['sketch'] for t in TAGS])
    a_best = best_in([d[t]['avg'] for t in TAGS])
    rows = []
    for tag in TAGS:
        lbl = '\\textbf{Ours}' if tag == 'OURS' else LABEL[tag]
        rows.append(f'  {lbl} & {fmt(d[tag]["rendition"], r_best)} & '
                    f'{fmt(d[tag]["sketch"], s_best)} & {fmt(d[tag]["avg"], a_best)} \\\\')
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Top-1 accuracy (\\%) on ImageNet-R and ImageNet-Sketch with W6A6 (PTQ4ViT) quantized ViT-B at batch size 64, seed 0, episodic reset. W6A6 is the representative setting; the other bitwidths are in the appendix (Table~\\ref{{tab:table4_appendix}}). ImageNet-R is evaluated over its 200-class subset via the standard \\texttt{{imagenet\\_r\\_mask}}; ImageNet-Sketch uses all 1000 classes.}}
\\label{{tab:table4_main_w6a6}}
\\begin{{tabular}}{{l|rrr}}
\\toprule
Method & ImageNet-R & ImageNet-Sketch & Avg. \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def emit_appendix_tex(data):
    """All four bitwidths, wide: per bw (R, Sketch, Avg)."""
    # best per (bw, metric) column for bolding
    best = {}
    for bw in BITWIDTHS:
        d = data.get(bw)
        if d is None:
            best[bw] = {'r': None, 's': None, 'a': None}
            continue
        best[bw] = {
            'r': best_in([d[t]['rendition'] for t in TAGS]),
            's': best_in([d[t]['sketch'] for t in TAGS]),
            'a': best_in([d[t]['avg'] for t in TAGS]),
        }
    rows = []
    for tag in TAGS:
        lbl = '\\textbf{Ours}' if tag == 'OURS' else LABEL[tag]
        cells = []
        for bw in BITWIDTHS:
            d = data.get(bw)
            if d is None:
                cells += ['--', '--', '--']
            else:
                cells += [fmt(d[tag]['rendition'], best[bw]['r']),
                          fmt(d[tag]['sketch'], best[bw]['s']),
                          fmt(d[tag]['avg'], best[bw]['a'])]
        rows.append(f'  {lbl} & ' + ' & '.join(cells) + ' \\\\')

    top = ' & '.join(f'\\multicolumn{{3}}{{c|}}{{{BW_LABEL[bw]} ({CKPT_KIND[bw]})}}'
                     for bw in BITWIDTHS)
    sub = ' & '.join('R & Sk. & Avg.' for _ in BITWIDTHS)
    colspec = 'l|' + '|'.join('rrr' for _ in BITWIDTHS)
    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{Top-1 accuracy (\\%) on ImageNet-R (R, 200-class subset via \\texttt{{imagenet\\_r\\_mask}}) and ImageNet-Sketch (Sk., 1000 classes) across all bitwidths. Quantized ViT-B, batch size 64, seed 0, episodic reset. Avg.\\ is the mean of the two datasets. W6A6 (also in the main paper, Table~\\ref{{tab:table4_main_w6a6}}) is the representative setting.}}
\\label{{tab:table4_appendix}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{colspec}}}
\\toprule
Method & {top} \\\\
 & {sub} \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular}%
}
\\end{table*}
"""


def emit_json(data):
    obj = {
        'setting': {
            'batch_size': 64, 'seed': 0, 'episodic_reset': True,
            'datasets': {'rendition': DATA_R, 'sketch': DATA_SKETCH},
            'main_table_bitwidth': 'w6a6',
            'main_table_rationale': ('W6A6 fixed a priori as the representative '
                'setting (mid-range, PTQ4ViT, consistent with Tables 1-3); NOT '
                'chosen post-hoc for best Ours performance.'),
            'ece_reported': False,
        },
        'checkpoints': {bw: CKPT_PATH[bw] for bw in BITWIDTHS},
        'bitwidths': {},
    }
    for bw in BITWIDTHS:
        d = data.get(bw)
        if d is None:
            obj['bitwidths'][bw] = None
            continue
        obj['bitwidths'][bw] = {
            'label': BW_LABEL[bw], 'ckpt_kind': CKPT_KIND[bw], 'checkpoint': CKPT_PATH[bw],
            'methods': {tag: {
                'label': LABEL[tag],
                'rendition_top1': d[tag]['rendition'],
                'sketch_top1': d[tag]['sketch'],
                'avg_top1': d[tag]['avg'],
                'csv_rendition': d[tag]['csv_rendition'],
                'csv_sketch': d[tag]['csv_sketch'],
            } for tag in TAGS},
        }
    return obj


def emit_report(data, obj):
    def g(bw, tag, k):
        d = data.get(bw)
        return d[tag][k] if d else None

    # best (bw, method) by avg
    best_avg = None; best_where = None
    for bw in BITWIDTHS:
        if data.get(bw) is None:
            continue
        for tag in TAGS:
            a = data[bw][tag]['avg']
            if a is not None and (best_avg is None or a > best_avg):
                best_avg = a; best_where = (bw, tag)

    md = ["# Table 4 — Robustness beyond ImageNet-C (ImageNet-R + ImageNet-Sketch)\n",
          "**Setting:** quantized ViT-B | batch size 64 | seed 0 | episodic reset "
          "(no `--continual`). ImageNet-R uses its 200-class subset via the standard "
          "`imagenet_r_mask`; ImageNet-Sketch uses all 1000 classes.\n",
          "**Main vs appendix:** W6A6 is the representative MAIN-table setting; "
          "W3A3/W4A4/W6A6/W8A8 are in the appendix. W6A6 was fixed a priori "
          "(mid-range bitwidth, PTQ4ViT, consistent with Tables 1–3) and **not** "
          "selected post-hoc to maximize Ours.\n",
          "**Ours = Local-Global Activation Recalibration:** a **local pre-quant / "
          "input-side activation recalibration** (`quar_local_mode=explicit`, at selected "
          "intermediate activation interfaces, matched to source calibration statistics) "
          "plus a **global final-CLS recalibration**, both with source-initialized running "
          "target moments (α=0.05), no parameter updates, no past-sample storage.\n",
          "## Accuracy by bitwidth (top-1 %)\n"]

    for bw in BITWIDTHS:
        d = data.get(bw)
        md.append(f"### {BW_LABEL[bw]} ({CKPT_KIND[bw]})\n")
        if d is None:
            md.append("_MISSING — run did not produce parseable results._\n")
            continue
        md.append("| Method | ImageNet-R | ImageNet-Sketch | Avg. |")
        md.append("|--------|-----------:|----------------:|-----:|")
        r_best = best_in([d[t]['rendition'] for t in TAGS])
        s_best = best_in([d[t]['sketch'] for t in TAGS])
        a_best = best_in([d[t]['avg'] for t in TAGS])
        def mark(v, b):
            if v is None: return '--'
            return f'**{v:.2f}**' if (b is not None and abs(v-b) < 1e-6) else f'{v:.2f}'
        for tag in TAGS:
            lbl = '**Ours**' if tag == 'OURS' else LABEL[tag]
            md.append(f"| {lbl} | {mark(d[tag]['rendition'], r_best)} | "
                      f"{mark(d[tag]['sketch'], s_best)} | {mark(d[tag]['avg'], a_best)} |")
        md.append("")

    # ---- analysis ----
    md.append("## Analysis\n")
    if best_where:
        bw, tag = best_where
        md.append(f"- **Best average overall:** {LABEL[tag]} at {BW_LABEL[bw]} "
                  f"= {best_avg:.2f}%.\n")

    md.append("- **Does Ours improve over No Adapt?** (per bitwidth, Δ pp)\n")
    md.append("| Bitwidth | ΔR (Ours−NoAdapt) | ΔSketch | ΔAvg |")
    md.append("|----------|------------------:|--------:|-----:|")
    for bw in BITWIDTHS:
        d = data.get(bw)
        if d is None:
            md.append(f"| {BW_LABEL[bw]} | -- | -- | -- |"); continue
        def delta(k):
            o = d['OURS'][k]; n = d['NOADAPT'][k]
            return (o - n) if (o is not None and n is not None) else None
        dr, ds, da = delta('rendition'), delta('sketch'), delta('avg')
        f = lambda x: f'{x:+.2f}' if x is not None else '--'
        md.append(f"| {BW_LABEL[bw]} | {f(dr)} | {f(ds)} | {f(da)} |")
    md.append("")

    md.append("- **Ours vs strong baselines (NEO / ZOA / FOZO), by Avg:**\n")
    md.append("| Bitwidth | Ours | NEO | ZOA | FOZO | Ours rank (of 7) | verdict |")
    md.append("|----------|-----:|----:|----:|-----:|-----------------:|---------|")
    for bw in BITWIDTHS:
        d = data.get(bw)
        if d is None:
            md.append(f"| {BW_LABEL[bw]} | -- | -- | -- | -- | -- | -- |"); continue
        avgs = {t: d[t]['avg'] for t in TAGS}
        ours = avgs['OURS']
        present = [(t, a) for t, a in avgs.items() if a is not None]
        ranked = sorted(present, key=lambda x: -x[1])
        rank = next((i + 1 for i, (t, _) in enumerate(ranked) if t == 'OURS'), None)
        top = ranked[0][1] if ranked else None
        others = [a for t, a in present if t != 'OURS']
        if ours is None:
            verdict = 'missing'
        elif rank == 1:
            # strict best, unless another method is within 0.10 pp -> tied-best
            verdict = 'tied-best' if any(abs(ours - a) < 0.10 for a in others) else 'best'
        else:
            gap = ours - top  # negative: how far below the best
            verdict = f'competitive (#{rank}, {gap:+.2f} vs best)'
        f = lambda x: f'{x:.2f}' if x is not None else '--'
        md.append(f"| {BW_LABEL[bw]} | {f(ours)} | {f(avgs['NEO'])} | {f(avgs['ZOA'])} | "
                  f"{f(avgs['FOZO'])} | {rank if rank else '--'} | {verdict} |")
    md.append("")

    # ---- provenance ----
    md.append("## Provenance per number\n")
    md.append("Every entry below is parsed from `corruption_metrics.csv` (`top1` column, "
              "`corruption` row = `rendition` or `sketch`). All runs: seed 0, batch size 64, "
              "episodic reset. W3A3/W6A6 reuse prior validated runs; W4A4/W8A8 are fresh "
              "single-pass runs.\n")
    md.append("| Bitwidth | Checkpoint | Run dir (R / Sketch CSV) |")
    md.append("|----------|------------|--------------------------|")
    for bw in BITWIDTHS:
        d = data.get(bw)
        cr = d['OURS']['csv_rendition'] if d else None
        cs = d['OURS']['csv_sketch'] if d else None
        same = (cr == cs)
        loc = f"`{cr}`" if same else f"R: `{cr}`<br>Sk: `{cs}`"
        md.append(f"| {BW_LABEL[bw]} | `{CKPT_PATH[bw]}` | {loc} |")
    md.append("")
    md.append("Full per-method CSV paths and numbers are in "
              "`outputs/table4_r_sketch_all_bitwidths.json`.\n")

    md.append("## ImageNet-R label-space / mask handling\n")
    md.append("- Targets are ImageFolder local indices `[0,199]` (alphabetical R synsets).\n"
              "- `validate_adapt` (main.py:108–126) applies `imagenet_r_mask` (1000→200) "
              "to any method returning 1000-dim logits, logging a `[R-audit]` line on the "
              "first batch. Verified per run:\n"
              "  - No Adapt / NEO / Ours / FOZO / ZOA: `raw_out_dim=1000 masked=True "
              "final_out_dim=200`.\n"
              "  - FOA: `raw_out_dim=200 masked=False` (FOA masks internally; no double-mask).\n"
              "  - T3A: `raw_out_dim=1000 masked=True` — the rewritten T3A (tta_library/t3a.py) "
              "maps its masked-argmax back to true ImageNet-1k IDs, so external masking is "
              "correct.\n"
              "- ImageNet-Sketch: full 1000-class space, no mask.\n")
    return '\n'.join(md)


def main():
    print('Collecting...')
    data = collect()
    os.makedirs(OUT, exist_ok=True)

    p = f'{OUT}/table4_r_sketch_w6a6_main.tex'
    open(p, 'w').write(emit_main_tex(data)); print(f'Wrote {p}')

    p = f'{OUT}/table4_r_sketch_all_bitwidths_appendix.tex'
    open(p, 'w').write(emit_appendix_tex(data)); print(f'Wrote {p}')

    obj = emit_json(data)
    p = f'{OUT}/table4_r_sketch_all_bitwidths.json'
    json.dump(obj, open(p, 'w'), indent=2); print(f'Wrote {p}')

    p = f'{OUT}/table4_r_sketch_report.md'
    open(p, 'w').write(emit_report(data, obj)); print(f'Wrote {p}')


if __name__ == '__main__':
    main()
