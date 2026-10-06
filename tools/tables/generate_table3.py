#!/usr/bin/env python3
"""Parse Table 3 rounds experiment and emit LaTeX / JSON / markdown report."""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, json, os, sys

OUTDIR = _os.path.join(_REPO_ROOT, "outputs")
LAST = _os.path.join(_REPO_ROOT, "runs/last_table3_dir.txt")
if not os.path.exists(LAST):
    print(f'MISSING: {LAST}'); sys.exit(1)
OUTBASE = open(LAST).read().strip()
print(f'Table 3 OUTBASE: {OUTBASE}')

CORR15 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

# (tag, algo, label, fp_per_batch, has_rounds)
# Long-term continual 150-segment stream. ZOA's continual carry behavior is
# officially supported and well-defined.
METHODS = [
    ('OURS',      'no_adapt', 'Ours',        1, True),
    ('ZOA',       'zoa',      'ZOA',         2, True),
    ('ZOA_OURS',  'zoa',      'ZOA + Ours',  2, True),
]


def parse_continual_csv(path):
    """Long-term continual corruption_metrics.csv (--corruption_repeat 10).
    The 'round' column = stream_round = the r-th pass through 15 corruptions.
    Returns dict {round_idx: {corruption: top1}}.
    """
    out = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            r = int(row['round'])
            out.setdefault(r, {})[row['corruption']] = float(row['top1'])
    return out


def parse_corr_csv(path):
    """Single-pass corruption_metrics.csv → {corruption: top1}."""
    out = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            # In single-pass mode, ignore round column (all rows have round=1).
            out[row['corruption']] = float(row['top1'])
    return out


def mean_acc(per_corr):
    """Mean over 15 corruptions; require all present."""
    vals = [per_corr.get(c) for c in CORR15]
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def collect():
    results = {}
    for tag, algo, label, fp, has_rounds in METHODS:
        run_dir = os.path.join(OUTBASE, tag, f'{algo}_{tag}_w6a6')
        if has_rounds:
            rounds = parse_continual_csv(os.path.join(run_dir, 'corruption_metrics.csv'))
            if rounds is None:
                print(f'  MISSING (rounds): {tag} @ {run_dir}')
                results[tag] = None
                continue
            per_round = {r: mean_acc(rounds[r]) for r in sorted(rounds)}
            results[tag] = {
                'label': label, 'fp_per_batch': fp, 'has_rounds': True,
                'per_round': per_round,
                'rounds_detail': {r: rounds[r] for r in sorted(rounds)},
            }
            print(f'  OK: {tag} rounds={sorted(rounds)} mean={[f"{per_round[r]:.2f}" if per_round[r] else "--" for r in sorted(rounds)]}')
        else:
            d = parse_corr_csv(os.path.join(run_dir, 'corruption_metrics.csv'))
            if d is None:
                print(f'  MISSING: {tag} @ {run_dir}')
                results[tag] = None; continue
            avg = mean_acc(d)
            results[tag] = {
                'label': label, 'fp_per_batch': fp, 'has_rounds': False,
                'per_round': {1: avg},
                'rounds_detail': {1: d},
            }
            print(f'  OK: {tag} (single-pass) mean={avg:.3f}')
    return results


def emit_latex(results):
    """Method | R1..R10 | Avg LaTeX table."""
    rounds = list(range(1, 11))

    # Per-method R1..R10 values + average
    def values_of(tag):
        d = results.get(tag)
        if d is None: return [None] * 10, None
        vs = []
        for r in rounds:
            vs.append(d['per_round'].get(r) if d['has_rounds'] else d['per_round'].get(1))
        valid = [v for v in vs if v is not None]
        avg = sum(valid) / len(valid) if valid else None
        return vs, avg

    method_data = {tag: values_of(tag) for tag, _, _, _, _ in METHODS}

    # Best per column (R1..R10 + Avg) across methods
    best = {}
    for i, r in enumerate(rounds):
        col_vals = [(tag, method_data[tag][0][i]) for tag, _, _, _, _ in METHODS
                    if method_data[tag][0][i] is not None]
        if col_vals:
            best[r] = max(col_vals, key=lambda x: x[1])[1]
    avg_vals = [(tag, method_data[tag][1]) for tag, _, _, _, _ in METHODS
                if method_data[tag][1] is not None]
    best_avg = max((v for _, v in avg_vals), default=None)

    def fmt(v, b):
        if v is None: return '--'
        s = f'{v:.2f}'
        return f'\\textbf{{{s}}}' if (b is not None and abs(v-b) < 1e-6) else s

    rows = []
    for tag, algo, label, fp, has_rounds in METHODS:
        vs, avg = method_data[tag]
        cells = [fmt(v, best.get(r)) for v, r in zip(vs, rounds)]
        cells.append(fmt(avg, best_avg))
        disp = '\\textbf{Ours}' if tag == 'OURS' else label
        rows.append('  ' + disp + ' & ' + ' & '.join(cells) + ' \\\\')

    tex = """\\begin{table*}[t]
\\centering
\\caption{Mean accuracy on ImageNet-C severity 5 (W6A6 ViT-B, 15 corruptions, 5k subset, bs=64, seed 0) under long-term continual adaptation. The full stream is 10 rounds $\\times$ 15 corruptions = 150 corruption passes processed sequentially with state carried through the entire stream (\\texttt{--continual --corruption\\_repeat 10}); reset only once at stream start. Round $r$ accuracy is the mean top-1 over the 15 corruptions during the $r$-th pass. Avg.\\ is the mean over R1--R10.}
\\label{tab:table3_longterm_continual_zoa}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{l|""" + 'r' * 10 + """|r}
\\toprule
Method & R1 & R2 & R3 & R4 & R5 & R6 & R7 & R8 & R9 & R10 & Avg. \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular}%
}
\\end{table*}
"""
    return tex


def emit_report(results):
    rounds = list(range(1, 11))

    def get_val(tag, r):
        d = results.get(tag)
        if d is None: return None
        if d['has_rounds']:
            return d['per_round'].get(r)
        return d['per_round'].get(1)

    md = """# Table 3 — Long-term continual adaptation across rounds

## 1. Exact stream order

The full evaluation stream is **150 corruption segments** processed
sequentially: 10 rounds × 15 corruptions per round. Within each round, the
15 corruptions are processed in this fixed order:

```
gaussian_noise → shot_noise → impulse_noise →
defocus_blur → glass_blur → motion_blur → zoom_blur →
snow → frost → fog → brightness →
contrast → elastic_transform → pixelate → jpeg_compression
```

Round 2 repeats the same 15-corruption sequence (gaussian_noise → ... →
jpeg_compression), and so on through Round 10. Each segment uses the
RobustBench 5k subset at severity 5 with bs=64 and seed 0.

Round `r` accuracy = mean top-1 over the 15 corruption segments during the
r-th pass (entries with `stream_round == r` in `corruption_metrics.csv`).

## 2. Reset policy

- Adaptation state is reset **only once** at the very beginning of the
  150-segment stream (model load + source-stat initialisation).
- **No reset** at corruption boundaries.
- **No reset** between rounds.
- For Ours, the Stream-the local branch EMA (`_quar_stream_m/q/n`) and Stream-CLS target
  EMA (`_mu_t_ema/_sig_t_ema`) are source-initialised once and carried
  through the entire 150-segment stream.
- For ZOA, the official implementation includes a built-in safety reset on
  detected domain change (`if domain_change: self.hist_stat = None`); this
  is the only intra-stream reset and is intrinsic to the method, not added
  by us.
- For FOZO, no intrinsic safety reset exists; state accumulates through the
  entire stream.

## 3. State carry summary per method

| Method | ZOA prompts/eps | local-branch stream EMA | CLS target EMA |
|--------|------------------|----------------|----------------|
| Ours        | --              | source-init → carries all 150 | source-init → carries all 150 |
| ZOA         | source-init → carries all 150 | -- | -- |
| ZOA + Ours  | source-init → carries all 150 | source-init → carries all 150 | source-init → carries all 150 |

ZOA's official implementation has a built-in safety reset of `hist_stat` on
detected domain change; this is intrinsic to ZOA and is the only intra-stream
state perturbation we permit.

## 4. Commands / configs

W6A6 PTQ4ViT ViT-B, ImageNet-C sev 5, 5k subset, bs=64, seed 0.
All methods use `--corruption_repeat 10 --continual` to realise the
150-segment continual stream.

```bash
# Shared
BASE="--data ... --data_corruption ... --level 5 --imagenet_5k_eval --batch_size 64 \\
      --corruption all --seed 0 --imagenet_5k_indices_file ... --imagenet_class_map_file ..."
W6A6="--quant --quant_w_bit 6 --quant_a_bit 6 --ptq_checkpoint .../ptq4vit_w6a6_calibrated.pt"
CONTINUAL="--corruption_repeat 10 --continual"

OURS_FLAGS="--quar_local_enable --quar_local_mode explicit --quar_local_scope qba_valid \\
  --quar_local_stats mean_std --quar_local_eps 1e-6 --quar_local_clip_ratio 3.0 --quar_local_lambda 1.0 \\
  --quar_local_stats_file qac_stats_w6a6_seed0.pt --quar_local_exclude_layers mlp.fc2 \\
  --quar_local_stream_ema --quar_local_stream_alpha 0.05 --quar_local_stream_nwarm 0 --quar_local_stream_lag \\
  --quar_global_enable --quar_global_position post_ln --quar_global_mode mean_std \\
  --quar_global_alpha 1.0 --quar_global_eps 1e-6 --quar_global_clip_ratio 3.0 \\
  --quar_global_stats_file cls_stats_w6a6_seed0.pt --quar_global_target_ema_alpha 0.05"

ZOA_FLAGS="--algorithm zoa --official_zoa_steps 1 --official_zoa_sp_avg 1 \\
  --official_zoa_lr_eps 0.0002 --zoa_layer_balance none \\
  --official_zoa_stats_file zoa_stats/w6a6.pt"

# Method dispatches
$BASE $W6A6 $CONTINUAL --algorithm no_adapt $OURS_FLAGS   # Ours
$BASE $W6A6 $CONTINUAL $ZOA_FLAGS                          # ZOA
$BASE $W6A6 $CONTINUAL $ZOA_FLAGS $OURS_FLAGS              # ZOA + Ours
```

`ZOA + Ours` differs from base ZOA only by adding the `OURS_FLAGS` hook
bundle (Stream-the local branch + Stream-CLS). ZOA hyperparameters are identical
between base and +Ours.

## 5. Per-round mean accuracy (top-1 %, mean over 15 corruptions)

| Method | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | Avg. |
|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|----:|----:|
"""
    for tag, algo, label, fp, has_rounds in METHODS:
        row = ['**Ours**' if tag == 'OURS' else label]
        vals = []
        for r in rounds:
            v = get_val(tag, r)
            vals.append(v)
            row.append(f'{v:.2f}' if v is not None else '--')
        valid = [v for v in vals if v is not None]
        avg = sum(valid) / len(valid) if valid else None
        row.append(f'{avg:.2f}' if avg is not None else '--')
        md += '| ' + ' | '.join(row) + ' |\n'

    md += """
## 6. Gain of ZOA + Ours over ZOA (pp)

| Round | ZOA | ZOA + Ours | Δ |
|------:|----:|-----------:|--:|
"""
    for r in rounds:
        z = get_val('ZOA', r); zo = get_val('ZOA_OURS', r)
        if z is None or zo is None:
            md += f'| R{r} | -- | -- | -- |\n'
        else:
            md += f'| R{r} | {z:.2f} | {zo:.2f} | {zo-z:+.2f} |\n'

    md += """
## 7. Cumulative #FP per round (forwards per sample, all 15 corruptions in one round)

Each corruption segment processes 5000 / 64 ≈ 79 batches. Per-round FP cost is
`(forwards/batch) × 79 × 15`. Cumulative through round R is `R × (one-round cost)`.

| Method | Forwards/batch | One-round FP | Cumul. FP at R10 |
|--------|---------------:|-------------:|-----------------:|
"""
    for tag, _, label, fp, _ in METHODS:
        per_round = fp * 79 * 15
        cumul = per_round * 10
        disp = '**Ours**' if tag == 'OURS' else label
        md += f'| {disp} | {fp} | {per_round:,} | {cumul:,} |\n'

    md += """
## 8. Does Ours shift the ZOA continual adaptation curve upward?

"""
    plus_tag = 'ZOA_OURS'
    base_tag, base_label = 'ZOA', 'ZOA'
    deltas = []
    for r in rounds:
        v_b = get_val(base_tag, r); v_p = get_val(plus_tag, r)
        if v_b is None or v_p is None: continue
        deltas.append((r, v_p - v_b))
    if not deltas:
        md += f'- **{base_label}**: insufficient data.\n'
    else:
        positive_rounds = sum(1 for _, d in deltas if d > 0)
        mean_delta = sum(d for _, d in deltas) / len(deltas)
        max_delta = max(deltas, key=lambda x: x[1])
        min_delta = min(deltas, key=lambda x: x[1])
        verdict = 'YES' if positive_rounds == len(deltas) else (
                  'partial (positive in some rounds)' if positive_rounds > len(deltas) / 2 else 'NO')
        md += (f'- **{base_label}**: + Ours shifts curve upward in '
               f'{positive_rounds}/{len(deltas)} measured rounds. '
               f'Mean Δ = {mean_delta:+.2f} pp, max Δ = {max_delta[1]:+.2f} pp at R{max_delta[0]}, '
               f'min Δ = {min_delta[1]:+.2f} pp at R{min_delta[0]}. **Verdict: {verdict}**.\n')

    md += """
## 9. Does ZOA eventually surpass Ours after enough rounds?

"""
    base_tag, base_label = 'ZOA', 'ZOA'
    crossover = None
    for r in rounds:
        v_b = get_val(base_tag, r); v_o = get_val('OURS', r)
        if v_b is None or v_o is None: continue
        if v_b > v_o:
            crossover = (r, v_b, v_o); break
    if crossover is None:
        v_b_last = get_val(base_tag, 10); v_o_last = get_val('OURS', 10)
        if v_b_last is not None and v_o_last is not None:
            md += (f'- **{base_label}**: never surpasses Ours within 10 rounds. '
                   f'At R10, {base_label} = {v_b_last:.2f}, Ours = {v_o_last:.2f}, '
                   f'gap = {(v_o_last - v_b_last):+.2f} pp.\n')
        else:
            md += f'- **{base_label}**: insufficient data to evaluate crossover.\n'
    else:
        r, vb, vo = crossover
        md += (f'- **{base_label}**: first surpasses Ours at R{r} '
               f'({base_label}={vb:.2f} > Ours={vo:.2f}, +{vb-vo:.2f} pp).\n')

    md += """
## 10. Caveat

This experiment evaluates a **long-term continual multi-pass setting**:
the full evaluation stream is 150 corruption passes (10 × 15) with state
carried through the entire stream — no episodic reset. At R10 the
cumulative compute is ~10× the default single-pass deployment cost, and the
adaptation state has been updated by ~150 × (number of batches per
corruption) test-time iterations. This is **not** the recommended production
setting (Table 1's single-pass episodic regime is) and is included here
only to show **composability**: whether Ours' source-aligning hooks remain
beneficial when stacked with prompt-based optimization that has effectively
unlimited test-time budget on a domain stream.

For the default single-pass deployment cost, see Table 1.
"""
    return md


def main():
    results = collect()

    # LaTeX
    tex = emit_latex(results)
    tex_path = os.path.join(OUTDIR, 'table3_longterm_continual_zoa.tex')
    with open(tex_path, 'w') as f: f.write(tex)
    print(f'\nWrote {tex_path}')

    # JSON
    json_path = os.path.join(OUTDIR, 'table3_longterm_continual_zoa.json')
    json_obj = {
        'outbase': OUTBASE,
        'setting': {
            'bitwidth': 'W6A6', 'corruptions': CORR15, 'severity': 5,
            'subset': 'robustbench 5k', 'batch_size': 64, 'seed': 0,
            'episodic_reset_per_corruption': True, 'rounds_per_corruption': 10,
            'note_ours': 'Ours is single-pass; per_round[1] is the only measured value, R2-R10 are repeated as constant reference in the LaTeX table.',
        },
        'methods': {tag: (d if d is None else {
            'label': d['label'], 'fp_per_batch': d['fp_per_batch'],
            'has_rounds': d['has_rounds'], 'per_round_mean': d['per_round'],
            'avg_over_rounds': (sum(v for v in d['per_round'].values() if v is not None) /
                                max(1, len([v for v in d['per_round'].values() if v is not None]))),
            'per_round_per_corruption': d['rounds_detail'],
        }) for tag, d in zip([m[0] for m in METHODS], [results.get(m[0]) for m in METHODS])},
    }
    with open(json_path, 'w') as f: json.dump(json_obj, f, indent=2)
    print(f'Wrote {json_path}')

    # Markdown report
    md = emit_report(results)
    md_path = os.path.join(OUTDIR, 'table3_longterm_continual_zoa_report.md')
    with open(md_path, 'w') as f: f.write(md)
    print(f'Wrote {md_path}')


if __name__ == '__main__':
    main()
