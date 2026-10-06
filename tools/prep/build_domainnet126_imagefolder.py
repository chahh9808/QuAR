#!/usr/bin/env python
"""Lay DomainNet-126 out as ImageFolder trees of symlinks, in the ImageNet-C-like layout the TTA loader expects.

Inputs: the extracted DomainNet "cleaned" domain folders (<img_root>/<domain>/<class>/*.jpg) and the
AdaContrast 126-class lists (<lists>/<domain>_list.txt, lines "<domain>/<class>/<file> <label>").

Output (class dirs are zero-padded label indices, so ImageFolder's class order equals the label order):
  <out>/real_src/train/<lll>/*      source split used for fine-tuning, PTQ calibration and source statistics
  <out>/real_src/val/<lll>/*        held-out source images (seeded 10% of the real list) for clean accuracy
  <out>/targets/dn_<domain>/5/<lll>/*   full target-domain lists (clipart, painting, sketch), plus dn_real = real val
"""
import argparse
import os
import random


def read_list(p):
    items = []
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        rel, lab = ln.rsplit(' ', 1)
        items.append((rel, int(lab)))
    return items


def link_items(items, img_root, out_dir, num_classes):
    for c in range(num_classes):
        os.makedirs(os.path.join(out_dir, f'{c:03d}'), exist_ok=True)
    n = 0; missing = 0
    for rel, lab in items:
        src = os.path.join(img_root, rel)
        if not os.path.exists(src):
            missing += 1; continue
        dst = os.path.join(out_dir, f'{lab:03d}', os.path.basename(rel))
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        n += 1
    print(f'[link] {out_dir}: {n} images ({missing} missing)', flush=True)
    return n, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_root', required=True, help='directory containing real/ clipart/ painting/ sketch/')
    ap.add_argument('--lists', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source', default='real')
    ap.add_argument('--targets', default='clipart,painting,sketch')
    ap.add_argument('--val_frac', type=float, default=0.1, help='seeded split of <source>_list.txt when no explicit split files are given')
    ap.add_argument('--source_train_list', default=None, help='explicit train list (e.g. AdaContrast real_train.txt); overrides the seeded split')
    ap.add_argument('--source_val_list', default=None, help='explicit held-out list (e.g. AdaContrast real_test.txt)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_classes', type=int, default=126)
    args = ap.parse_args()

    if args.source_train_list and args.source_val_list:
        train_items = read_list(args.source_train_list); val_items = read_list(args.source_val_list)
        split_desc = f'explicit lists {os.path.basename(args.source_train_list)} / {os.path.basename(args.source_val_list)}'
    else:
        src_items = read_list(os.path.join(args.lists, f'{args.source}_list.txt'))
        rng = random.Random(args.seed); idx = list(range(len(src_items))); rng.shuffle(idx)
        n_val = int(round(args.val_frac * len(src_items)))
        val_items = [src_items[i] for i in sorted(idx[:n_val])]
        train_items = [src_items[i] for i in sorted(idx[n_val:])]
        split_desc = f'seeded split val_frac={args.val_frac} seed={args.seed}'
    src_root = os.path.join(args.out, f'{args.source}_src')
    link_items(train_items, args.img_root, os.path.join(src_root, 'train'), args.num_classes)
    link_items(val_items, args.img_root, os.path.join(src_root, 'val'), args.num_classes)
    link_items(val_items, args.img_root, os.path.join(args.out, 'targets', f'dn_{args.source}', '5'), args.num_classes)
    for dom in args.targets.split(','):
        items = read_list(os.path.join(args.lists, f'{dom}_list.txt'))
        link_items(items, args.img_root, os.path.join(args.out, 'targets', f'dn_{dom}', '5'), args.num_classes)
    with open(os.path.join(args.out, 'SPLIT.txt'), 'w') as f:
        f.write(f'source={args.source} split={split_desc} img_root={args.img_root} train={len(train_items)} val={len(val_items)}\n')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
