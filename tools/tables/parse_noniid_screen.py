#!/usr/bin/env python3
"""Parse non-IID class-order screen results."""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, os, sys

OUTBASE = sys.argv[1] if len(sys.argv) > 1 else open(
    _os.path.join(_REPO_ROOT, "runs/last_noniid_screen_dir.txt")
).read().strip()

CORR = ['gaussian_noise', 'motion_blur', 'elastic_transform', 'pixelate']
METHODS = ['A', 'B', 'C', 'D']
LABELS = {
    'A': 'no_adapt',
    'B': 'NEO',
    'C': 'Ours-local',
    'D': 'Ours-full',
}
ORDERS = ['iid', 'class_sorted']


def load_metrics(csv_path):
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row['corruption']] = float(row['top1'])
    return out


results = {}  # (method, order) -> {corruption: top1}
for m in METHODS:
    for o in ORDERS:
        # main.py output: $subdir/$tag/<algorithm><args.tag>/corruption_metrics.csv
        # launcher passes --tag "_${tag}_${order}" with order='iid'|'class_sorted'
        run_dir = os.path.join(OUTBASE, o, m, f'no_adapt_{m}_{o}')
        csv_path = os.path.join(run_dir, 'corruption_metrics.csv')
        if not os.path.exists(csv_path):
            print(f'MISSING: {csv_path}')
            continue
        results[(m, o)] = load_metrics(csv_path)

# print per-corruption table
print('\n=== Per-corruption accuracy (top1, %) ===')
header = f'{"Method":<14}{"Order":<14}' + ''.join(f'{c[:12]:>14}' for c in CORR) + f'{"mean":>10}'
print(header)
print('-' * len(header))
for m in METHODS:
    for o in ORDERS:
        if (m, o) not in results:
            continue
        d = results[(m, o)]
        accs = [d.get(c, float('nan')) for c in CORR]
        mean = sum(accs) / len(accs)
        row = f'{m}-{LABELS[m]:<11}{o:<14}' + ''.join(f'{a:>14.3f}' for a in accs) + f'{mean:>10.3f}'
        print(row)

# IID vs non-IID drop
print('\n=== IID vs non-IID drop (mean over 4 corruptions, pp) ===')
print(f'{"Method":<20}{"IID mean":>12}{"non-IID":>12}{"drop":>10}')
print('-' * 54)
drops = {}
for m in METHODS:
    if (m, 'iid') not in results or (m, 'class_sorted') not in results:
        continue
    iid_mean = sum(results[(m, 'iid')].values()) / len(CORR)
    nii_mean = sum(results[(m, 'class_sorted')].values()) / len(CORR)
    drop = iid_mean - nii_mean
    drops[m] = drop
    print(f'{m}-{LABELS[m]:<17}{iid_mean:>12.3f}{nii_mean:>12.3f}{drop:>+10.3f}')

# Verdict
print('\n=== Verdict ===')
if 'B' in drops and 'C' in drops:
    print(f'NEO drop:        {drops["B"]:+.3f} pp')
    print(f'Ours-local drop: {drops["C"]:+.3f} pp')
    if 'D' in drops:
        print(f'Ours-full drop:  {drops["D"]:+.3f} pp')
    print()
    if drops['B'] > drops.get('C', drops['B']) and drops['B'] > drops.get('D', drops['B']):
        print('-> NEO is more sensitive to class-non-IID ordering than Ours variants.')
    elif drops['B'] < min(drops.get('C', drops['B']), drops.get('D', drops['B'])):
        print('-> NEO is LESS sensitive than Ours variants (unexpected; reconsider).')
    else:
        print('-> Mixed result; differences small.')
