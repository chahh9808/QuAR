#!/usr/bin/env python
"""Aggregate the CIFAR-10-C / CIFAR-100-C extension campaign (scripts/run_cifar_queue.sh).

Reads every <run_dir>/<ds>_w<b>a<b>_<method>_s<seed>.log, takes the per-corruption
'Under shift type <c> After <alg> Top-1 Accuracy: <x>' lines (15 standard corruptions, severity 5,
10k images each), averages them per run, then reports mean +- std over seeds.
Outputs outputs/cifar_accuracy.{md,json,tex}.
"""
import argparse
import glob
import json
import os
import re
import statistics

CORR = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
METHODS = [('noadapt', 'No Adapt'), ('t3a', 'T3A'), ('foa', 'FOA'), ('zoa', 'ZOA'), ('neo', 'NEO'), ('fozo', 'FOZO'), ('quar', 'QuAR')]  # paper order; rows appear only when logs exist
DN_DOMAINS = ['dn_clipart', 'dn_painting', 'dn_sketch']
CORR_OF = {'cifar10': CORR, 'cifar100': CORR, 'domainnet126': DN_DOMAINS, 'imagenet_a': ['imagenet_a'], 'imagenet_v2': ['v2']}
LINE = re.compile(r"Under shift type (\S+) After (\S+) Top-1 Accuracy: ([0-9.]+)")
NAME = re.compile(r"^(cifar10|cifar100|domainnet126|imagenet_a|imagenet_v2)_w(\d)a(\d)_([a-z0-9]+)_s(\d+)\.log$")  # [a-z0-9]: "t3a" has a digit


def parse(path, corr):
    acc = {}
    for ln in open(path, errors='ignore'):
        m = LINE.search(ln)
        if m and m.group(1) in corr:
            acc[m.group(1)] = float(m.group(3))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default=None, help='one or more run dirs, comma-separated (default: runs/last_cifar_dir.txt)')
    ap.add_argument('--out_prefix', default='outputs/cifar_accuracy')
    args = ap.parse_args()
    run_dir = args.run_dir or open('runs/last_cifar_dir.txt').read().strip()
    run_dirs = [d for d in run_dir.split(',') if d]
    cells = {}   # (ds, bw, method) -> {seed: (mean, per_corr)}
    incomplete = []
    for p in sorted(q for d in run_dirs for q in glob.glob(os.path.join(d, '*.log'))):
        m = NAME.match(os.path.basename(p))
        if not m:
            continue
        ds, wb, ab, meth, seed = m.groups()
        corr = CORR_OF[ds]
        acc = parse(p, corr)
        if len(acc) < len(corr):
            incomplete.append((os.path.basename(p), len(acc)))
            continue
        cells.setdefault((ds, f'W{wb}A{ab}', meth), {})[int(seed)] = (sum(acc[c] for c in corr) / len(corr), acc)

    table = {}
    for (ds, bw, meth), seeds in cells.items():
        means = [v[0] for _, v in sorted(seeds.items())]
        table[(ds, bw, meth)] = (statistics.mean(means), statistics.pstdev(means) if len(means) > 1 else 0.0, len(means), seeds)

    dss = sorted({k[0] for k in table}, key=lambda d: {'cifar10': 0, 'cifar100': 1, 'domainnet126': 2, 'imagenet_a': 3, 'imagenet_v2': 4}.get(d, 9))
    bws = sorted({k[1] for k in table}, key=lambda b: -int(b[1]))  # W8A8, W6A6, W4A4, W3A3
    lines = [f'# CIFAR-C extension: quantized ViT-B/16 fine-tuned on CIFAR, severity 5, 15 corruptions x 10k images, bs 64, episodic',
             f'Run dir: `{run_dir}`. Cells: mean +- std over seeds (n seeds in parentheses).', '']
    hdr = '| Method | ' + ' | '.join(f'{d} {b}' for d in dss for b in bws) + ' |'
    lines += [hdr, '|' + '---|' * (1 + len(dss) * len(bws))]
    js = {}
    for key, name in METHODS:
        if not any(k[2] == key for k in table):
            continue
        row = [name]
        for d in dss:
            for b in bws:
                t = table.get((d, b, key))
                if t is None:
                    row.append('-'); continue
                mu, sd, n, seeds = t
                row.append(f'{mu:.2f} +- {sd:.2f} ({n})')
                js.setdefault(d, {}).setdefault(b, {})[key] = {
                    'mean': mu, 'std': sd, 'n_seeds': n,
                    'per_seed_mean': {s: v[0] for s, v in seeds.items()},
                    'per_corruption_seed0': seeds.get(0, (None, {}))[1]}
        lines.append('| ' + ' | '.join(row) + ' |')
    # per-corruption (seed 0) block
    for d in dss:
        for b in bws:
            corr = CORR_OF[d]
            lines += ['', f'## {d} {b}: per-corruption top-1, seed 0', '| Method | ' + ' | '.join(c[:6] for c in corr) + ' | Mean |',
                      '|' + '---|' * (len(corr) + 2)]
            for key, name in METHODS:
                t = table.get((d, b, key))
                if t is None or 0 not in t[3]:
                    continue
                pc = t[3][0][1]
                lines.append('| ' + name + ' | ' + ' | '.join(f'{pc[c]:.1f}' for c in corr) + f' | {t[3][0][0]:.2f} |')
    if incomplete:
        lines += ['', '## Incomplete logs (skipped)'] + [f'- {n}: {k}/15 corruptions' for n, k in incomplete]
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    open(args.out_prefix + '.md', 'w').write('\n'.join(lines) + '\n')
    json.dump({'run_dir': run_dir, 'table': js, 'incomplete': incomplete}, open(args.out_prefix + '.json', 'w'), indent=1)
    # LaTeX (means only; std in json)
    tex = ['\\begin{tabular}{l|' + 'c' * (len(dss) * len(bws)) + '}', '\\toprule',
           'Method & ' + ' & '.join(f'{d.replace("cifar", "CIFAR-").replace("domainnet126", "DomainNet-126").replace("imagenet_a", "ImageNet-A").replace("imagenet_v2", "ImageNet-V2")} {b}' for d in dss for b in bws) + ' \\\\', '\\midrule']
    for key, name in METHODS:
        if not any(k[2] == key for k in table):
            continue
        vals = []
        for d in dss:
            for b in bws:
                t = table.get((d, b, key)); vals.append(f'{t[0]:.2f}' if t else '--')
        tex.append(f'{name} & ' + ' & '.join(vals) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    open(args.out_prefix + '.tex', 'w').write('\n'.join(tex) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
