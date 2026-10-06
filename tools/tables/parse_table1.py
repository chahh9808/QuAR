#!/usr/bin/env python3
"""Parse Table 1 per-bitwidth accuracy.

Usage:
  python parse_table1.py <bitwidth>   e.g. w6a6 / w4a4 / w8a8
"""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, os, sys, json

BW = sys.argv[1] if len(sys.argv) > 1 else 'w6a6'

# 15 corruptions in fixed paper order
CORR15 = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

# Method definitions: name → (run_dir_path, algorithm_subdir_name)
# For new Table 1 runs: $OUTBASE/<TAG>/<algorithm>_<TAG>_<bw>/
# For reused legacy runs: explicit path
def find_dir(outbase, tag, algorithm, bw):
    return os.path.join(outbase, tag, f'{algorithm}_{tag}_{bw}')


def zoa_row_dir(outbase, bw):
    """Locate the ZOA row.

    tab:main publishes the ZOA run under ZOA's paper LayerNorm set: 31.45 / 50.87 / 51.37 / 58.16.
    Earlier campaign directories hold ZOA runs under other settings and are not used.
    """
    ptr = f'{_REPO_ROOT}/runs/last_zoa_table1_refix_dir.txt'
    if os.path.exists(ptr):
        base = open(ptr).read().strip()
        tag = bw.upper()
        cand = os.path.join(base, f'{tag}_s0', f'zoa_{tag}_s0')
        if os.path.isdir(cand):
            return cand
    legacy = f'{_REPO_ROOT}/runs/table1_zoa_rerun_20260528_104640/{bw}/zoa_ZOA_{bw}'
    return legacy if os.path.isdir(legacy) else find_dir(outbase, 'ZOA', 'zoa', bw)


def ours_dir(outbase, bw):
    """Locate the QuAR row.

    The published row comes from run_cumulative_maintable.sh (deployed configuration); prefer
    that campaign and fall back to the local directory only when it is absent.
    """
    ptr = f'{_REPO_ROOT}/runs/last_cum_maintable_dir.txt'
    if os.path.exists(ptr):
        base = open(ptr).read().strip()
        cand = os.path.join(base, f'{bw}_cum_s0', f'no_adapt_{bw}_cum_s0')
        if os.path.isdir(cand):
            return cand
    return find_dir(outbase, 'OURS', 'no_adapt', bw)


def parse_corruption_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row['corruption']] = {
                'top1': float(row['top1']),
                'top5': float(row['top5']),
                'steady_forward_ms': float(row.get('steady_forward_ms', float('nan'))),
            }
    return out


def main():
    last_table1_file = f'{_REPO_ROOT}/runs/last_table1_{BW}_dir.txt'
    if not os.path.exists(last_table1_file):
        print(f'MISSING: {last_table1_file}')
        sys.exit(1)
    outbase = open(last_table1_file).read().strip()
    print(f'Outbase: {outbase}')

    # New Table 1 methods (run by run_table1_<bw>.sh).
    # For w6a6/w4a4/w8a8 the ZOA run in `runs/table1_zoa_rerun_20260528_104640/` is the
    # published one; for w3a3 the local ZOA dir is authoritative.
    zoa_dir = zoa_row_dir(outbase, BW)
    new_methods = {
        'T3A':  ('t3a',       find_dir(outbase, 'T3A',  't3a',      BW)),
        'FOA':  ('foa',       find_dir(outbase, 'FOA',  'foa',      BW)),
        'FOZO': ('fozo',      find_dir(outbase, 'FOZO', 'fozo',     BW)),
        'ZOA':  ('zoa',       zoa_dir),
        'OURS': ('no_adapt',  ours_dir(outbase, BW)),
    }

    # Legacy methods (NoAdapt, NEO) from prior runs
    if BW == 'w6a6':
        legacy = {
            'NoAdapt': _os.path.join(_REPO_ROOT, "runs/full15_20260527_170503/bs64/A/no_adapt_A_bs64"),
            'NEO':     _os.path.join(_REPO_ROOT, "runs/full15_20260527_170503/bs64/C/no_adapt_C_bs64"),
        }
    elif BW == 'w4a4':
        legacy = {
            'NoAdapt': _os.path.join(_REPO_ROOT, "runs/crossbw_w4a4_rerun_20260527_194629/w4a4/bs64/A/no_adapt_A_w4a4_bs64"),
            'NEO':     _os.path.join(_REPO_ROOT, "runs/crossbw_w4a4_rerun_20260527_194629/w4a4/bs64/C/no_adapt_C_w4a4_bs64"),
        }
    elif BW == 'w8a8':
        legacy = {
            'NoAdapt': _os.path.join(_REPO_ROOT, "runs/crossbw_20260527_180901/w8a8/bs64/A/no_adapt_A_w8a8_bs64"),
            'NEO':     _os.path.join(_REPO_ROOT, "runs/crossbw_20260527_180901/w8a8/bs64/C/no_adapt_C_w8a8_bs64"),
        }
    elif BW == 'w3a3':
        # W3A3 NoAdapt reused from prior validation; NEO ran fresh as part of W3A3 launcher.
        legacy = {
            'NoAdapt': _os.path.join(_REPO_ROOT, "runs/w3a3_validation_20260526_092645/w3a3_A_no_adapt/no_adapt_w3a3_A_no_adapt"),
        }
        # NEO at W3A3 is in the new methods (not legacy)
        new_methods['NEO'] = ('no_adapt', find_dir(outbase, 'NEO', 'no_adapt', BW))
    else:
        legacy = {}

    results = {}
    for method, (algo, run_dir) in new_methods.items():
        csv_path = os.path.join(run_dir, 'corruption_metrics.csv')
        d = parse_corruption_csv(csv_path)
        if d is None:
            print(f'  MISSING: {method} @ {csv_path}')
            results[method] = None
        else:
            results[method] = d
            print(f'  OK: {method} @ {csv_path} ({len(d)} corruptions)')

    for method, run_dir in legacy.items():
        csv_path = os.path.join(run_dir, 'corruption_metrics.csv')
        d = parse_corruption_csv(csv_path)
        if d is None:
            print(f'  MISSING (legacy): {method} @ {csv_path}')
            results[method] = None
        else:
            results[method] = d
            print(f'  OK (legacy): {method} @ {csv_path} ({len(d)} corruptions)')

    # Per-corruption table
    print(f'\n=== {BW.upper()} per-corruption top1 ===')
    order = ['NoAdapt', 'T3A', 'FOA', 'FOZO', 'ZOA', 'NEO', 'OURS']
    header = f'{"Method":<10}' + ''.join(f'{c[:10]:>12}' for c in CORR15) + f'{"Avg":>10}'
    print(header)
    print('-' * len(header))
    summary = {}
    for m in order:
        d = results.get(m)
        if d is None:
            print(f'{m:<10}' + ''.join(f'{"--":>12}' for _ in CORR15) + f'{"--":>10}')
            summary[m] = {'per_corr': {c: None for c in CORR15}, 'avg': None}
            continue
        per_corr = {c: d.get(c, {'top1': None})['top1'] for c in CORR15}
        valid_vals = [v for v in per_corr.values() if v is not None]
        avg = sum(valid_vals)/len(valid_vals) if valid_vals else None
        line = f'{m:<10}' + ''.join(f'{(per_corr[c] if per_corr[c] is not None else "--"):>12.3f}' if per_corr[c] is not None else f'{"--":>12}' for c in CORR15)
        line += f'{(avg if avg is not None else "--"):>10.3f}' if avg is not None else f'{"--":>10}'
        print(line)
        summary[m] = {'per_corr': per_corr, 'avg': avg}

    out_json = f'{_REPO_ROOT}/outputs/table1_{BW}_acc.json'
    with open(out_json, 'w') as f:
        json.dump({'bitwidth': BW, 'outbase': outbase, 'methods': summary}, f, indent=2)
    print(f'\nSaved: {out_json}')


if __name__ == '__main__':
    main()
