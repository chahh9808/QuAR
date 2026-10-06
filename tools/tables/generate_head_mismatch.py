#!/usr/bin/env python3
"""Head-quantizer activation mismatch (W6A6), companion to
generate_intermediate_activation_mismatch.py.

Same metric, different interface. The intermediate table reads the mismatch at
blocks 8-11 x {qkv,proj,fc1,fc2}, where Final-only and NEO necessarily score 0%
because neither acts there. This reads it at the interface they do act at, the
classifier head's activation quantizer input.

  mean mismatch = mean over channels of |mu_t - mu_s|
  std  mismatch = mean over channels of |sigma_t - sigma_s|
  combined      = mean mismatch + std mismatch
  reduction     = 100 * (M_noadapt - M_method) / M_noadapt

Source stats = cls_stats_w6a6_seed0.pt['post_ln'], the statistics the global CLS
branch matches to (collected at the same net.head pre-hook position).

The PRIMARY tensor is the pre-quant activation entering the head quantizer, the
tensor the source stats were built on. post-DQ is reported as a secondary check.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import os
import torch

CLS_STATS = f'{_REPO_ROOT}/stats/cls_stats_w6a6_seed0.pt'
METHODS = ['NOADAPT', 'NEO', 'FINAL', 'LOCAL', 'OURS']
LABEL = {'NOADAPT': 'No Adapt', 'NEO': 'NEO', 'FINAL': 'Final-only',
         'LOCAL': 'Local-only', 'OURS': 'Ours'}


def run_dir():
    ptr = f'{_REPO_ROOT}/runs/last_head_mismatch_dir.txt'
    if os.path.exists(ptr):
        b = open(ptr).read().strip()
        if b and os.path.isdir(b):
            return b
    import glob
    c = sorted(glob.glob(f'{_REPO_ROOT}/runs/head_mismatch_2*'))
    return c[-1] if c else None


def load_head(tag, base):
    """-> {'prequant': {'mean','std'}, 'postdq': {...}} for the head module."""
    import glob
    hits = glob.glob(os.path.join(base, tag, '**', 'qac_mismatch_stats.pt'),
                     recursive=True)
    if not hits:
        return None
    d = torch.load(sorted(hits)[0], map_location='cpu')
    key = next((k for k in d if k.split('.')[-1] == 'head'), None)
    return d[key] if key else None


def mismatch(t, s):
    """t, s: {'mean','std'} -> (mean_mm, std_mm, combined)."""
    mm = (t['mean'].float() - s['mean'].float()).abs().mean().item()
    sm = (t['std'].float() - s['std'].float()).abs().mean().item()
    return mm, sm, mm + sm


def main():
    base = run_dir()
    if not base:
        raise SystemExit('no head_mismatch run dir found')
    print(f'run dir: {base}\n')

    cls = torch.load(CLS_STATS, map_location='cpu')['post_ln']
    src = {'mean': cls['mean'].float(), 'std': cls['std'].float()}
    print(f'source (cls_stats post_ln): C={src["mean"].numel()} '
          f'|mu_s|mean={src["mean"].abs().mean():.4f} sigma_s mean={src["std"].mean():.4f}')

    # consistency: the SOURCE run's own pre-quant head stats vs cls_stats
    s_run = load_head('SOURCE', base)
    if s_run and 'prequant' in s_run:
        mm, sm, _ = mismatch(s_run['prequant'], src)
        print(f'consistency check, SOURCE run pre-quant vs cls_stats: '
              f'|dmu|={mm:.4f} |dsigma|={sm:.4f}  (should be ~0)')
    else:
        print('consistency check: SOURCE run stats unavailable')

    for kind in ('prequant', 'postdq'):
        rows, base_c = [], None
        for m in METHODS:
            h = load_head(m, base)
            if h is None or kind not in h:
                rows.append((LABEL[m], None, None, None, None))
                continue
            mm, sm, c = mismatch(h[kind], src)
            if m == 'NOADAPT':
                base_c = c
            red = None if base_c in (None, 0) else 100.0 * (base_c - c) / base_c
            rows.append((LABEL[m], mm, sm, c, red))

        tag = 'PRE-QUANT (primary)' if kind == 'prequant' else 'POST-DQ (secondary)'
        print(f'\n===== head quantizer input, {tag}, W6A6 =====')
        print(f'{"Method":<12}{"|dmu|":>9}{"|dsigma|":>10}{"combined":>10}{"reduction":>11}')
        for label, mm, sm, c, red in rows:
            if c is None:
                print(f'{label:<12}{"n/a":>9}{"n/a":>10}{"n/a":>10}{"n/a":>11}')
            else:
                r = '--' if red is None else f'{red:+.1f}%'
                print(f'{label:<12}{mm:>9.4f}{sm:>10.4f}{c:>10.4f}{r:>11}')


if __name__ == '__main__':
    main()
