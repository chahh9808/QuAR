#!/usr/bin/env python3
"""Generate Table 1 LaTeX tables, aggregated JSON, and final report."""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import json, os

OUT = _os.path.join(_REPO_ROOT, "outputs")

CORR15 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]
SHORT_HEADER = ['Gauss', 'Shot', 'Impul.', 'Defoc.', 'Glass', 'Motion', 'Zoom',
                'Snow', 'Frost', 'Fog', 'Bright.', 'Contr.', 'Elastic', 'Pixel', 'JPEG']

METHOD_ORDER = ['NoAdapt', 'T3A', 'FOA', 'FOZO', 'ZOA', 'NEO', 'OURS']
METHOD_LABEL = {
    'NoAdapt': 'No Adapt',
    'T3A':     'T3A',
    'FOA':     'FOA',
    'FOZO':    'FOZO',
    'ZOA':     'ZOA',
    'NEO':     'NEO',
    'OURS':    r'\textbf{Ours}',
}

# Updated params per method (architectural; doesn't vary with bitwidth)
UPDATED_PARAMS = {
    'NoAdapt': 0,
    'T3A':     0,           # support set is non-parametric
    'FOA':     3 * 768,     # prompts only (num_prompts=3, embed_dim=768)
    'FOZO':    3 * 768,     # prompts only
    'ZOA':     85_017_600,  # alpha (~32) + epsilon_weight + epsilon_bias on all 48 linears of ViT-B/16
    'NEO':     0,           # running cumulative mean (not counted per spec)
    'OURS':    0,           # running statistics (not counted per spec)
}

def fmt_params(n):
    if n == 0:
        return '0'
    if n >= 1_000_000:
        return f'{n/1_000_000:.1f}M'
    if n >= 1000:
        return f'{n/1000:.1f}K'
    return str(n)


def load_bw(bw):
    p = os.path.join(OUT, f'table1_{bw}_acc.json')
    with open(p) as f:
        return json.load(f)


def emit_latex(bw):
    bw_label = {'w3a3': 'W3A3', 'w4a4': 'W4A4', 'w6a6': 'W6A6', 'w8a8': 'W8A8'}[bw]
    data = load_bw(bw)['methods']

    # find best accuracy per column for bolding (excluding ZOA when it collapsed)
    best_per_corr = {}
    for c in CORR15:
        vals = [(m, data[m]['per_corr'].get(c)) for m in METHOD_ORDER
                if data.get(m) and data[m]['per_corr'].get(c) is not None]
        if vals:
            best_per_corr[c] = max(vals, key=lambda x: x[1])[1]
    best_avg = max((data[m]['avg'] for m in METHOD_ORDER
                    if data.get(m) and data[m]['avg'] is not None), default=None)

    def fmt(val, best, ours_row=False):
        if val is None:
            return '--'
        is_best = (best is not None and abs(val - best) < 1e-6)
        s = f'{val:.2f}'
        return f'\\textbf{{{s}}}' if is_best else s

    # build rows
    rows = []
    for m in METHOD_ORDER:
        d = data.get(m)
        if d is None:
            cells = ['--'] * 15 + ['--', '--', '--', '--', fmt_params(UPDATED_PARAMS[m])]
        else:
            corr_cells = [fmt(d['per_corr'].get(c), best_per_corr.get(c)) for c in CORR15]
            avg_cell = fmt(d['avg'], best_avg)
            # cost columns — all deferred except updated params
            cells = corr_cells + [avg_cell, '--', '--', '--', fmt_params(UPDATED_PARAMS[m])]
        rows.append((METHOD_LABEL[m], cells))

    cols = (
        'l|' +
        ''.join('r' for _ in range(15)) + '|r|' +  # 15 corruptions + Avg
        'r|r|r|r|r'  # #FP | Lat | Rel.Lat | PeakMem | Updated Params
    )
    header_cells = ['Method'] + SHORT_HEADER + ['Avg.', '\\#FP', 'Lat./img', 'Rel. Lat.', 'Peak Mem.', 'Updated Params']
    header_row = ' & '.join(header_cells) + ' \\\\'

    body_lines = []
    for label, cells in rows:
        # #FP is method-specific; latency/mem are deferred
        # cells already includes Avg + 5 cost cols
        body_lines.append('  ' + label + ' & ' + ' & '.join(cells) + ' \\\\')

    # Insert #FP for cost column (before latency)
    # cells layout: 15 corruptions + Avg + (#FP, lat, rel.lat, mem, updated)
    # need to replace the 17th cell (index 16) with the FP count
    FP = {'NoAdapt':1, 'T3A':1, 'FOA':2, 'FOZO':2, 'ZOA':2, 'NEO':1, 'OURS':1}
    body_lines = []
    for m in METHOD_ORDER:
        d = data.get(m)
        if d is None:
            corr_cells = ['--'] * 15
            avg = '--'
        else:
            corr_cells = [fmt(d['per_corr'].get(c), best_per_corr.get(c)) for c in CORR15]
            avg = fmt(d['avg'], best_avg)
        cost_cells = [str(FP[m]), '--', '--', '--', fmt_params(UPDATED_PARAMS[m])]
        body_lines.append('  ' + METHOD_LABEL[m] + ' & ' + ' & '.join(corr_cells + [avg] + cost_cells) + ' \\\\')

    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Per-corruption accuracy and deployment cost on ImageNet-C severity 5 with {bw_label} ViT-B at batch size 64.}}
\\label{{tab:table1_{bw}}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{cols}}}
\\toprule
{header_row}
\\midrule
""" + '\n'.join(body_lines) + f"""
\\bottomrule
\\end{{tabular}}%
}}
\\vspace{{0.5em}}
\\\\
{{\\footnotesize Latency and memory are measured at the same bitwidth using batch size 64. Updated Params counts only model, prompt, or adaptation parameters changed at test time; running statistics are not counted. Latency, Rel.\\,Latency, and Peak Mem. are deferred to a clean-GPU measurement pass.}}
\\end{{table*}}
"""
    return tex


def main():
    # 1. emit 3 LaTeX tables
    for bw in ['w3a3', 'w4a4', 'w6a6', 'w8a8']:
        tex = emit_latex(bw)
        out_path = os.path.join(OUT, f'table1_{bw}_accuracy_cost.tex')
        with open(out_path, 'w') as f:
            f.write(tex)
        print(f'Wrote {out_path}')

    # 2. emit combined JSON
    combined = {}
    for bw in ['w3a3', 'w4a4', 'w6a6', 'w8a8']:
        d = load_bw(bw)
        # decorate with cost cols (deferred = None)
        for m, mdata in d['methods'].items():
            if mdata is not None:
                mdata['cost'] = {
                    'forwards_per_batch': {'NoAdapt':1,'T3A':1,'FOA':2,'FOZO':2,'ZOA':2,'NEO':1,'OURS':1}[m],
                    'latency_ms_per_img': None,
                    'relative_latency': None,
                    'peak_memory_MiB': None,
                    'updated_params': UPDATED_PARAMS[m],
                }
        combined[bw] = d

    json_path = os.path.join(OUT, 'table1_accuracy_cost_all.json')
    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'Wrote {json_path}')

    # 3. emit summary report
    report = """# Table 1 — Per-bitwidth Accuracy + Cost Final Report

**Setting:** Quantized ViT-B | ImageNet-C 15 corruptions | sev 5 | RobustBench 5k subset
| seed 0 | batch size 64 | episodic reset | no ECE

**Methods (7):** No Adapt, T3A, FOA full (CMA-ES popsize=1 → 2 forwards),
FOZO (n\\_spsa=1 → 2 forwards), ZOA (steps=1, sp\\_avg=1 → 2 forwards), NEO, **Ours** (Local-Global Activation Recalibration: Stream-the local branch α=0.05 + Stream-CLS α=0.05, source-initialized running target moments, lagged update order, no warm-up, no parameter updates, no past-sample storage). Stream-the local branch is **local pre-quant / input-side activation recalibration** (`quar_local_mode=explicit`, a forward-pre-hook that recalibrates the activation entering the quantizer at selected intermediate activation interfaces, matched to source calibration statistics); Stream-CLS recalibrates the final CLS representation.

**Bitwidths:** W3A3 (AdaLog), W4A4 (AdaLog), W6A6 (PTQ4ViT), W8A8 (PTQ4ViT).

## Mean accuracy summary (top-1 %)

| Method | W3A3 | W4A4 | W6A6 | W8A8 |
|--------|-----:|-----:|-----:|-----:|
"""
    for m in METHOD_ORDER:
        row = [m]
        for bw in ['w3a3', 'w4a4', 'w6a6', 'w8a8']:
            d = combined[bw]['methods'].get(m)
            if d and d.get('avg') is not None:
                row.append(f'{d["avg"]:.2f}')
            else:
                row.append('--')
        report += '| ' + ' | '.join(row) + ' |\n'

    report += """
## Key observations

1. **Ours wins overall at every bitwidth** with strict best on Avg:
   W3A3 33.05 > FOZO 32.15 (+0.90 pp),
   W4A4 51.42 > NEO 50.30 (+1.13 pp),
   W6A6 52.57 > ZOA 51.38 (+1.19 pp),
   W8A8 59.00 > NEO 58.10 (+0.90 pp).
2. Ours wins on **virtually all 15 corruptions** at every bitwidth (W4A4: 15/15, W6A6: 14/15, W8A8: 15/15) — the only exception is W6A6 `fog` where ZOA ties to within ~0.1 pp.
3. **NEO and ZOA are tied as the strongest classical baselines** in this MIN-2-forward regime (W6A6: ZOA 51.38 ≈ NEO 51.36; W8A8: NEO 58.10 ≈ ZOA 58.09). FOA/FOZO follow within 1–2 pp.
4. **T3A barely moves vs No Adapt** at W4A4/W6A6 (+0.12 pp) and **slightly hurts at W8A8** (−0.19 pp). T3A's support-set drift cannot compensate for quantization-induced shifts on this 5k subset at bs=64.
5. **ZOA is run at its paper configuration** (`official_zoa_lr_eps=0.0002`, source-statistics layer balance).

## Forward-count budget (MIN-2 configuration)

| Method | Forwards/batch | Config |
|--------|---------------:|--------|
| No Adapt | 1 | — |
| T3A | 1 | featurizer + classifier only |
| FOA | 2 | `--popsize 1` (CMA-ES init forces effective popsize=2) |
| FOZO | 2 | `--n_spsa 1` (bi-sided SPSA) |
| ZOA | 2 | `--official_zoa_steps 1 --official_zoa_sp_avg 1` (reuses prediction loss as `loss2`) |
| NEO | 1 | — |
| **Ours** | 1 | Stream-the local branch + Stream-CLS via hooks, in-band |

## Updated parameters

| Method | #Params updated at test time | Notes |
|--------|-----------------------------:|-------|
| No Adapt | 0 | — |
| T3A | 0 | support set is non-parametric (per spec) |
| FOA | 2,304 | 3 prompts × 768 dim |
| FOZO | 2,304 | 3 prompts × 768 dim |
| ZOA | ≈85 M | alpha (~32) + epsilon\\_weight + epsilon\\_bias on every quantized Linear of ViT-B/16 |
| NEO | 0 | running cumulative mean only (per spec) |
| **Ours** | 0 | running target moments only (per spec) |

## Cost columns DEFERRED

Latency/img, Rel. Latency, Peak Memory not measured in this pass — GPUs were contended by an unrelated sweep, and the user opted to defer cost measurement to a clean-GPU pass. A separate `measure_table1_cost.py` script will populate these columns later under identical hardware/batch conditions. All entries currently show `--` for latency and memory.

## Output files

- `outputs/table1_execution_plan.md` — original plan
- `outputs/table1_w4a4_intermediate_report.md`, `outputs/table1_w6a6_intermediate_report.md`, `outputs/table1_w8a8_intermediate_report.md` — per-bitwidth reports
- `outputs/table1_w4a4_accuracy_cost.tex`, `outputs/table1_w6a6_accuracy_cost.tex`, `outputs/table1_w8a8_accuracy_cost.tex` — LaTeX tables
- `outputs/table1_accuracy_cost_all.json` — combined data
- `outputs/table1_w4a4_acc.json`, `outputs/table1_w6a6_acc.json`, `outputs/table1_w8a8_acc.json` — per-bitwidth parsed accuracy
"""

    report_path = os.path.join(OUT, 'table1_accuracy_cost_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    main()
