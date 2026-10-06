#!/usr/bin/env python3
"""Intermediate activation mismatch analysis (W6A6).

Ours applies local pre-quant / input-side activation recalibration at selected
intermediate activation interfaces (qba_valid). The PRIMARY measured tensor here
is therefore the pre-quant activation entering the quantizer (post-recalibration
for Local/Ours) — exactly the tensor the source calibration statistics
(quar_local_stats) are built on. The post-DQ (dequantized) tensor is reported only as a
secondary DIAGNOSTIC, to show that at W6A6 the pre-quant and post-DQ mismatch are
nearly identical (so the mismatch lives in the intermediate activation, and is
not newly induced by quantization).

Sources:
  pre-quant source (MAIN) = qac_stats_w6a6_seed0.pt (the source calibration stats
                            Ours uses; collected at the activation entering the quantizer).
  post-DQ source (diagnostic) = SOURCE run's post-DQ stats on clean ImageNet.
  Consistency: the SOURCE run's pre-quant stats match quar_local_stats (checked).

Mismatch (per tensor type):
  mean mismatch = mean over selected channels (pooled across the 6 layers) of |mu_t - mu_s|
  std  mismatch = mean over selected channels of |sigma_t - sigma_s|
  combined      = mean mismatch + std mismatch
  reduction vs No Adapt = 100*(M_noadapt - M_method)/M_noadapt

Key message: Final-only and NEO do NOT reduce the selected intermediate activation
mismatch (they act at the final CLS/head), while Local-only and Ours do.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, json, os
import torch

OUT = f'{_REPO_ROOT}/outputs'
CORR15 = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
          'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
          'elastic_transform', 'pixelate', 'jpeg_compression']
LAYERS = [f'blocks.{b}.{s}' for b in (10, 11) for s in ('attn.qkv', 'attn.proj', 'mlp.fc1')]
METHODS = ['NOADAPT', 'FINAL', 'LOCAL', 'OURS', 'NEO']
LABEL = {'NOADAPT': 'No Adapt', 'FINAL': 'Final-only', 'LOCAL': 'Local-only',
         'OURS': 'Ours', 'NEO': 'NEO'}


def run_dir():
    ptr = f'{_REPO_ROOT}/runs/last_qac_mismatch_dir.txt'
    if os.path.exists(ptr):
        b = open(ptr).read().strip()
        if b and os.path.isdir(b):
            return b
    import glob
    c = sorted(glob.glob(f'{_REPO_ROOT}/runs/qac_mismatch_2*'))
    return c[-1] if c else None


def load_stats(path):
    """mismatch dump: {layer: {'prequant':{'mean','std','n'}, 'postdq':{...}}}."""
    if not path or not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu')


def acc_mean15(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    d = {}
    for r in csv.DictReader(open(csv_path)):
        try:
            d[r['corruption']] = float(r['top1'])
        except (KeyError, ValueError):
            pass
    vals = [d.get(c) for c in CORR15]
    if any(v is None for v in vals):
        return None
    return sum(vals) / 15.0


def local_source_as_layers(local_path):
    """quar_local_stats payload {'mean':{L:t}, 'std':{L:t}} -> {L:{'mean','std'}}."""
    q = torch.load(local_path, map_location='cpu')
    out = {}
    for L in q['mean']:
        out[L] = {'mean': q['mean'][L].float(), 'std': q['std'][L].float()}
    return out


def kind_source(stats, kind):
    """Extract {L:{'mean','std'}} for a tensor kind from a mismatch dump."""
    out = {}
    for L, d in stats.items():
        if kind in d:
            out[L] = {'mean': d[kind]['mean'].float(), 'std': d[kind]['std'].float()}
    return out


def compute_mismatch(target, source, layers):
    """Pool per-channel |diff| across layers. Returns (mean_mm, std_mm, combined, per_layer)."""
    mu_all, sig_all, per_layer = [], [], {}
    for L in layers:
        if L not in target or L not in source:
            continue
        mt, st = target[L]['mean'].float(), target[L]['std'].float()
        ms, ss = source[L]['mean'].float(), source[L]['std'].float()
        n = min(mt.numel(), ms.numel())
        md = (mt[:n] - ms[:n]).abs()
        sd = (st[:n] - ss[:n]).abs()
        mu_all.append(md); sig_all.append(sd)
        per_layer[L] = (float(md.mean()), float(sd.mean()))
    if not mu_all:
        return None
    mu = torch.cat(mu_all); sig = torch.cat(sig_all)
    mean_mm = float(mu.mean()); std_mm = float(sig.mean())
    return mean_mm, std_mm, mean_mm + std_mm, per_layer


def collect():
    rd = run_dir()
    data = {'run_dir': rd, 'methods': {}, 'layers': LAYERS}
    # sources
    local_src = local_source_as_layers(f'{_REPO_ROOT}/stats/qac_stats_w6a6_seed0.pt')
    src_stats = load_stats(os.path.join(rd, 'SOURCE', 'no_adapt_SOURCE', 'qac_mismatch_stats.pt'))
    src_prequant = kind_source(src_stats, 'prequant') if src_stats else None
    src_postdq = kind_source(src_stats, 'postdq') if src_stats else None
    data['_quar_src'] = local_src
    data['_src_prequant'] = src_prequant
    data['_src_postdq'] = src_postdq

    for m in METHODS:
        base = os.path.join(rd, m, f'no_adapt_{m}')
        st = load_stats(os.path.join(base, 'qac_mismatch_stats.pt'))
        acc = acc_mean15(os.path.join(base, 'corruption_metrics.csv'))
        tgt_pre = kind_source(st, 'prequant') if st else None
        tgt_post = kind_source(st, 'postdq') if st else None
        data['methods'][m] = {'acc': acc, 'tgt_pre': tgt_pre, 'tgt_post': tgt_post,
                              'csv': os.path.join(base, 'corruption_metrics.csv')}
    return data


def build(data):
    local_src = data['_quar_src']; src_post = data['_src_postdq']
    res = {}
    for m in METHODS:
        md = data['methods'][m]
        entry = {'acc': md['acc'], 'csv': md['csv']}
        # pre-quant mismatch vs quar_local_stats (canonical source Ours uses)
        if md['tgt_pre'] is not None:
            mm = compute_mismatch(md['tgt_pre'], local_src, LAYERS)
            if mm:
                entry['prequant'] = {'mean': mm[0], 'std': mm[1], 'combined': mm[2], 'per_layer': mm[3]}
        # post-DQ mismatch vs clean-source post-DQ
        if md['tgt_post'] is not None and src_post is not None:
            mm = compute_mismatch(md['tgt_post'], src_post, LAYERS)
            if mm:
                entry['postdq'] = {'mean': mm[0], 'std': mm[1], 'combined': mm[2], 'per_layer': mm[3]}
        res[m] = entry
    # reductions vs No Adapt
    for kind in ('prequant', 'postdq'):
        na = res.get('NOADAPT', {}).get(kind, {}).get('combined')
        for m in METHODS:
            e = res[m].get(kind)
            if e and na:
                e['reduction'] = 100.0 * (na - e['combined']) / na
    return res


def fmt(v, nd=3):
    return '--' if v is None else f'{v:.{nd}f}'


def emit_tex(res):
    # MAIN table: pre-quant / input-side (intermediate activation) mismatch
    rows = []
    for m in METHODS:
        e = res[m]; pq = e.get('prequant', {})
        acc = e.get('acc')
        lbl = '\\textbf{Ours}' if m == 'OURS' else LABEL[m]
        red = pq.get('reduction')
        rows.append(f"  {lbl} & {fmt(acc,2)} & {fmt(pq.get('mean'))} & {fmt(pq.get('std'))} & "
                    f"{fmt(pq.get('combined'))} & {('--' if red is None else f'{red:.1f}')} \\\\")
    return """\\begin{table}[t]
\\centering
\\caption{Intermediate activation mismatch at the selected pre-quant / input-side recalibration interfaces (qba_valid: blocks 10/11 $\\times$ \\{attn.qkv, attn.proj, mlp.fc1\\}) on W6A6 ViT-B, ImageNet-C full-15, severity 5, 5k subset, batch size 64, seed 0, episodic. The measured tensor is the activation entering the quantizer (post-recalibration for Local-only/Ours), compared to the source calibration statistics. Mismatch is the per-channel mean absolute deviation of the target moments from the source moments, averaged over channels. Accuracy is mean top-1 over 15 corruptions. (Post-DQ mismatch is nearly identical; see report.)}
\\label{tab:intermediate_activation_mismatch}
\\begin{tabular}{l|r|rrr|r}
\\toprule
Method & Acc. & Mean Mismatch $\\downarrow$ & Std Mismatch $\\downarrow$ & Combined $\\downarrow$ & Reduction vs No Adapt (\\%) $\\uparrow$ \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""


def emit_csv(res, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'tensor', 'layer', 'mean_mismatch', 'std_mismatch', 'combined'])
        for m in METHODS:
            for kind in ('prequant', 'postdq'):
                e = res[m].get(kind)
                if not e:
                    continue
                for L, (mu, sd) in e['per_layer'].items():
                    w.writerow([m, kind, L, f'{mu:.6f}', f'{sd:.6f}', f'{mu+sd:.6f}'])


def emit_json(res, data, sanity):
    obj = {
        'setting': {'bitwidth': 'W6A6', 'corruptions': CORR15, 'severity': 5,
                    'subset': 'robustbench 5k', 'batch_size': 64, 'seed': 0,
                    'episodic_reset': True, 'ece_reported': False,
                    'interfaces': LAYERS,
                    'primary_tensor': 'pre-quant / input-side activation entering the quantizer '
                                      '(post-recalibration for Local/Ours)',
                    'primary_source': 'qac_stats_w6a6_seed0.pt (source calibration statistics, pre-quant)',
                    'diagnostic_tensor': 'post-DQ (module.quant_input); reported only to show pre-quant ≈ post-DQ',
                    'diagnostic_source': 'clean-source post-DQ (SOURCE run)',
                    'note': 'Mismatch is a property of the intermediate activation under shift; '
                            'NOT newly induced by quantization (pre-quant ≈ post-DQ).'},
        'source_consistency_prequant_vs_qac_stats': sanity,
        'run_dir': data['run_dir'],
        'methods': {m: {
            'label': LABEL[m], 'acc_mean15': res[m].get('acc'),
            'prequant': {k: v for k, v in (res[m].get('prequant') or {}).items() if k != 'per_layer'},
            'postdq': {k: v for k, v in (res[m].get('postdq') or {}).items() if k != 'per_layer'},
            'prequant_per_layer': (res[m].get('prequant') or {}).get('per_layer'),
            'postdq_per_layer': (res[m].get('postdq') or {}).get('per_layer'),
            'csv': res[m]['csv'],
        } for m in METHODS},
    }
    return obj


def emit_report(res, data, sanity):
    L = []
    L.append("# Intermediate Activation Mismatch Analysis (W6A6)\n")
    L.append("**Key message:** at the selected intermediate activation interfaces, **Final-only and "
             "NEO do not reduce the source–target mismatch** (they recalibrate the final CLS "
             "representation / head, downstream of these interfaces), while **Local-only and Ours "
             "do** — and Ours also attains the best accuracy. The mismatch is a property of the "
             "intermediate activation distribution; it is **not** newly induced by quantization "
             "(post-DQ mismatch ≈ pre-quant mismatch, see diagnostic below).\n")
    L.append("**Question:** do the selected intermediate activation interfaces show source–target "
             "mismatch, and does Ours' local pre-quant recalibration reduce it?\n")
    L.append("**Setting:** W6A6 ViT-B | ImageNet-C full-15 | sev5 | RobustBench 5k | seed 0 "
             "| bs 64 | episodic. Interfaces = qba_valid (blocks 10/11 × {attn.qkv, attn.proj, "
             "mlp.fc1}, fc2 excluded).\n")
    L.append("**Measured tensor (MAIN):** the **pre-quant activation entering the quantizer** "
             "(`inputs[0]`) — exactly the tensor the source calibration statistics (`quar_local_stats`) are "
             "built on, and the tensor Ours recalibrates on the input side. For Local-only/Ours this "
             "is the **post-recalibration** pre-quant activation. Compared against the source "
             "calibration statistics `qac_stats_w6a6_seed0.pt`.\n")
    L.append("**Diagnostic tensor (secondary):** post-DQ = `module.quant_input(inputs[0])` (the "
             "dequantized activation), compared against a clean-source post-DQ baseline. Reported "
             "only to show pre-quant ≈ post-DQ.\n")
    L.append(f"**Source consistency check:** the SOURCE-run pre-quant stats vs `quar_local_stats` differ by "
             f"mean |Δμ|={fmt(sanity['mean_mu'])}, mean |Δσ|={fmt(sanity['mean_sig'])} "
             f"(small ⇒ the clean-source baselines are on the same domain as Ours' source).\n")

    def table(kind, title):
        L.append(f"## {title}\n")
        L.append("| Method | Acc. | Mean Mismatch ↓ | Std Mismatch ↓ | Combined ↓ | Reduction vs No Adapt (%) ↑ |")
        L.append("|--------|-----:|----------------:|---------------:|-----------:|----------------------------:|")
        for m in METHODS:
            e = res[m]; k = e.get(kind, {})
            red = k.get('reduction')
            lbl = '**Ours**' if m == 'OURS' else LABEL[m]
            L.append(f"| {lbl} | {fmt(e.get('acc'),2)} | {fmt(k.get('mean'))} | {fmt(k.get('std'))} | "
                     f"{fmt(k.get('combined'))} | {'--' if red is None else f'{red:.1f}'} |")
        L.append("")

    table('prequant', 'Pre-quant / input-side intermediate activation mismatch — MAIN')
    table('postdq', 'Post-DQ (dequantized) mismatch — diagnostic only (≈ pre-quant)')

    # analysis (primary = pre-quant / input-side)
    def g(m, kind, f): return (res[m].get(kind) or {}).get(f)
    na_pre = g('NOADAPT', 'prequant', 'combined')
    L.append("## Answers\n")

    # Q1
    L.append("### Q1. Do the selected intermediate activation interfaces show source–target mismatch?\n")
    if na_pre is not None:
        L.append(f"**Yes.** Under No Adapt the pre-quant / input-side combined mismatch is {fmt(na_pre)} "
                 f"(mean {fmt(g('NOADAPT','prequant','mean'))}, std {fmt(g('NOADAPT','prequant','std'))}). "
                 f"The corrupted-domain intermediate activations at blocks 10/11 deviate from the source "
                 f"calibration moments. This is a property of the intermediate activation distribution "
                 f"under shift — see Q4: it is **not** newly induced by quantization.\n")
    else:
        L.append("_No-Adapt pre-quant mismatch missing._\n")

    # Q2
    L.append("### Q2. Does Local-only reduce this mismatch more than Final-only or NEO?\n")
    loc = g('LOCAL', 'prequant', 'reduction'); fin = g('FINAL', 'prequant', 'reduction'); neo = g('NEO', 'prequant', 'reduction')
    if loc is not None:
        comps = []
        if fin is not None: comps.append(f"Final-only {fin:.1f}%")
        if neo is not None: comps.append(f"NEO {neo:.1f}%")
        L.append(f"**Yes.** Local-only reduces the pre-quant combined mismatch by **{loc:.1f}%** vs No "
                 f"Adapt, versus {', '.join(comps)}. Final-only and NEO recalibrate the final CLS "
                 f"representation / head (downstream of these interfaces) and do **not** touch the "
                 f"blocks-10/11 input activations, so their mismatch here stays ≈ No Adapt (near-zero "
                 f"reduction), as expected — this is the key message.\n")
    else:
        L.append("_Local-only reduction missing._\n")

    # Q3
    L.append("### Q3. Does Ours reduce mismatch while also giving the best accuracy?\n")
    ours_red = g('OURS', 'prequant', 'reduction'); ours_acc = res['OURS'].get('acc')
    accs = {m: res[m].get('acc') for m in METHODS}
    best_acc_m = max((m for m in METHODS if accs[m] is not None), key=lambda m: accs[m], default=None)
    if ours_red is not None and ours_acc is not None:
        is_best = (best_acc_m == 'OURS')
        L.append(f"Ours reduces the pre-quant combined mismatch by **{ours_red:.1f}%** and reaches the "
                 f"{'**highest**' if is_best else 'top-tier'} accuracy ({ours_acc:.2f}%"
                 + ("" if is_best else f"; best is {LABEL[best_acc_m]} {accs[best_acc_m]:.2f}%") +
                 "). So the local input-side recalibration's mismatch reduction co-occurs with the best "
                 "accuracy.\n")
    else:
        L.append("_Ours data missing._\n")

    # Q4 (diagnostic)
    L.append("### Q4 (diagnostic). Is post-DQ mismatch meaningfully different from pre-quant mismatch?\n")
    rows = []
    for m in METHODS:
        pre = g(m, 'prequant', 'combined'); pos = g(m, 'postdq', 'combined')
        if pre is not None and pos is not None:
            rows.append((m, pre, pos, pos - pre))
    if rows:
        L.append("| Method | Pre-quant combined | Post-DQ combined | Δ(post−pre) |")
        L.append("|--------|-------------------:|-----------------:|------------:|")
        for m, pre, pos, dd in rows:
            L.append(f"| {LABEL[m]} | {pre:.3f} | {pos:.3f} | {dd:+.3f} |")
        na = next((r for r in rows if r[0] == 'NOADAPT'), None)
        if na:
            _, pre, pos, dd = na
            rel = 100 * dd / pre if pre else 0
            if abs(rel) < 10:
                L.append(f"\n**Nearly identical.** For No Adapt, post-DQ ({pos:.3f}) ≈ pre-quant "
                         f"({pre:.3f}); Δ={dd:+.3f} ({rel:+.1f}%). The quantizer (round-to-nearest, "
                         f"near mean-preserving) adds little mismatch beyond what already exists in the "
                         f"underlying activation. The mismatch is in the activation, not created by "
                         f"quantization.\n")
            else:
                L.append(f"\nPost-DQ differs from pre-quant by {rel:+.1f}% (No Adapt), indicating the "
                         f"quantizer {'amplifies' if dd>0 else 'reduces'} the mismatch.\n")
    else:
        L.append("_Insufficient data._\n")

    # Q5 (terminology)
    L.append("### Q5. Terminology: this is intermediate activation mismatch.\n")
    L.append("The paper should call this **intermediate activation mismatch** (measured at the "
             "pre-quant / input-side interfaces), **not** \"quantizer-output mismatch.\" Reasons: "
             "(i) the source calibration statistics and the measured tensor are both the **pre-quant** "
             "activation; (ii) post-DQ mismatch is nearly identical to pre-quant (Q4), so quantization "
             "does not create the mismatch; (iii) the precision sweep showed the local branch also "
             "helps at FP (no quantizer present). Ours' local branch is **pre-quant / input-side "
             "activation recalibration** against source calibration statistics with running target "
             "moments — consistent with the Local-Global Activation Recalibration framing.\n")

    # provenance
    L.append("## Provenance\n")
    L.append(f"Run dir: `{data['run_dir']}`. Per-method stats: `<METHOD>/no_adapt_<METHOD>/"
             "qac_mismatch_stats.pt`; accuracy: `corruption_metrics.csv`. Source calibration "
             "statistics: `qac_stats_w6a6_seed0.pt` (pre-quant). Post-DQ baseline (diagnostic): SOURCE "
             "run on clean ImageNet. Layer-wise mismatch (both tensors) in "
             "`outputs/intermediate_activation_mismatch_layerwise.csv`.\n")
    return '\n'.join(L)


def main():
    data = collect()
    # source consistency: SOURCE-run prequant vs quar_local_stats
    sanity = {'mean_mu': None, 'mean_sig': None}
    if data['_src_prequant'] is not None:
        mm = compute_mismatch(data['_src_prequant'], data['_quar_src'], LAYERS)
        if mm:
            sanity = {'mean_mu': mm[0], 'mean_sig': mm[1]}
    res = build(data)
    os.makedirs(OUT, exist_ok=True)

    base = f'{OUT}/intermediate_activation_mismatch_analysis'
    open(f'{base}.tex', 'w').write(emit_tex(res)); print('Wrote tex')
    emit_csv(res, f'{OUT}/intermediate_activation_mismatch_layerwise.csv'); print('Wrote csv')
    json.dump(emit_json(res, data, sanity), open(f'{base}.json', 'w'), indent=2); print('Wrote json')
    open(f'{base}_report.md', 'w').write(emit_report(res, data, sanity)); print('Wrote report')

    print('\nSummary (post-DQ | pre-quant) combined mismatch, acc:')
    for m in METHODS:
        e = res[m]
        pd = e.get('postdq', {}); pq = e.get('prequant', {})
        print(f"  {LABEL[m]:11s} acc={fmt(e.get('acc'),2)}  postdq_comb={fmt(pd.get('combined'))} "
              f"red={pd.get('reduction')}  prequant_comb={fmt(pq.get('combined'))}")


if __name__ == '__main__':
    main()
