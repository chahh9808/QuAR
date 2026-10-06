#!/usr/bin/env python3
"""ImageNet-R label-space audit.

Verifies the relationship between:
  - ImageFolder dataset class indices (0..199 alphabetical synset order)
  - imagenet_r_mask (boolean 1000-dim mask over original ImageNet-1K class IDs)
  - timm ViT-B's head output (1000-dim, original ImageNet-1K class order)
"""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import sys, os, json
sys.path.insert(0, _REPO_ROOT)

from torchvision import datasets

# 1. ImageFolder loading
DATA_R = os.environ.get("DATA_R", "")
ds = datasets.ImageFolder(DATA_R)
print('=' * 70)
print('Task 1: ImageNet-R Dataset label mapping')
print('=' * 70)
print(f'len(dataset)         = {len(ds)}')
print(f'len(dataset.classes) = {len(ds.classes)}')
print(f'dataset.classes[:10] = {ds.classes[:10]}')
print(f'dataset.classes[-10:]= {ds.classes[-10:]}')
print()
print(f'class_to_idx (first 10):')
for k, v in list(ds.class_to_idx.items())[:10]:
    print(f'  {k} -> {v}')
print('class_to_idx (last 5):')
for k, v in list(ds.class_to_idx.items())[-5:]:
    print(f'  {k} -> {v}')
print()
print('Sample targets (first 5 samples):')
for i in range(5):
    path, lbl = ds.samples[i]
    print(f'  {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} -> label={lbl}')
print(f'Target range observed: min={min(t for _, t in ds.samples)}, max={max(t for _, t in ds.samples)}')

# 2. imagenet_r_mask audit
from dataset.ImageNetMask import imagenet_r_mask, imagenet_r_wnids, all_wnids
import torch

print()
print('=' * 70)
print('Task 2: imagenet_r_mask audit')
print('=' * 70)

m = imagenet_r_mask
print(f'type: {type(m).__name__}, length: {len(m)}, True count: {sum(m)}')
print(f'first 10 entries: {m[:10]}')
print(f'last  10 entries: {m[-10:]}')
print()

# True positions in mask = ImageNet-1K indices of R classes
true_indices = [i for i, b in enumerate(m) if b]
print(f'imagenet_r_mask True positions count: {len(true_indices)}')
print(f'first 10 True positions (ImageNet-1K class IDs): {true_indices[:10]}')
print(f'last  5 True positions: {true_indices[-5:]}')
print()

# Cross-check: imagenet_r_wnids vs all_wnids
print(f'len(imagenet_r_wnids) = {len(imagenet_r_wnids)}  (type: {type(imagenet_r_wnids).__name__})')
print(f'len(all_wnids)        = {len(all_wnids)}        (type: {type(all_wnids).__name__})')
_r_sorted = sorted(imagenet_r_wnids)
print(f'sorted(imagenet_r_wnids)[:5] = {_r_sorted[:5]}')
print(f'all_wnids[:5]                = {all_wnids[:5]}')
print(f'all_wnids sorted alphabetically? {all_wnids == sorted(all_wnids)}')
print()

# Critical check: does dataset.class_to_idx[wnid] == position of wnid in sorted R subset?
#                And does imagenet_r_mask True positions, in order, give the ImageNet-1K
#                IDs of those same wnids?
print('Critical mapping check:')
print(f'  For dataset class k (alphabetical R wnid), its ImageNet-1K class ID is all_wnids.index(wnid)')
print(f'  Expected: true_indices[k] == all_wnids.index(dataset.classes[k]) for all k in 0..199')
print()

# build dataset class index -> ImageNet-1K class ID
all_wnid_to_idx = {w: i for i, w in enumerate(all_wnids)}
dataset_idx_to_imagenet_idx = {}
ok = True
mismatches = []
for ds_idx, wnid in enumerate(ds.classes):
    if wnid in all_wnid_to_idx:
        imagenet_idx = all_wnid_to_idx[wnid]
        dataset_idx_to_imagenet_idx[ds_idx] = imagenet_idx
        if ds_idx < len(true_indices):
            if true_indices[ds_idx] != imagenet_idx:
                ok = False
                mismatches.append((ds_idx, wnid, true_indices[ds_idx], imagenet_idx))
    else:
        ok = False
        mismatches.append((ds_idx, wnid, None, None))

if ok:
    print(f'  ✓ All {len(ds.classes)} dataset classes correctly aligned with imagenet_r_mask')
    print(f'    dataset class k ↔ ImageNet-1K class true_indices[k] ↔ wnid {ds.classes[0]}, etc.')
else:
    print(f'  ✗ {len(mismatches)} MISMATCHES detected:')
    for ds_idx, wnid, ti, ii in mismatches[:10]:
        print(f'    ds_idx={ds_idx} wnid={wnid} mask_true_indices[k]={ti} all_wnids.index(wnid)={ii}')

# Check for a 201st dataset class (the mask has exactly 200)
if len(ds.classes) != 200:
    print()
    print(f'WARNING: dataset has {len(ds.classes)} classes but imagenet_r_mask has {sum(m)} True positions.')
    extra = [c for c in ds.classes if c not in imagenet_r_wnids]
    print(f'  Extra dataset classes not in imagenet_r_wnids: {extra}')
    missing = [w for w in imagenet_r_wnids if w not in ds.classes]
    print(f'  imagenet_r_wnids not in dataset.classes: {missing}')

# Save audit report
out_md = _os.path.join(_REPO_ROOT, "outputs/imagenet_r_mask_audit.md")
with open(out_md, 'w') as f:
    f.write('# ImageNet-R Label-Space Audit\n\n')
    f.write('## Dataset structure\n\n')
    f.write(f'- Path: `{DATA_R}`\n')
    f.write(f'- Total images: {len(ds)}\n')
    f.write(f'- Number of class subdirectories: {len(ds.classes)}\n')
    f.write(f'- Target range: 0..{max(t for _, t in ds.samples)}\n')
    f.write(f'- Targets are **ImageFolder local indices** (0..{len(ds.classes)-1}), NOT ImageNet-1K original IDs.\n\n')

    f.write('## imagenet_r_mask\n\n')
    f.write(f'- Type: `{type(m).__name__}` of length {len(m)}, with {sum(m)} `True` entries.\n')
    f.write(f'- `True` positions are ImageNet-1K class IDs (0..999) of R-included synsets.\n')
    f.write(f'- First 10 True positions: `{true_indices[:10]}`\n')
    f.write(f'- Last 5 True positions: `{true_indices[-5:]}`\n\n')

    f.write('## Cross-check\n\n')
    if ok:
        f.write(f'✅ **All {len(ds.classes)} dataset classes correctly aligned with `imagenet_r_mask`.**\n\n')
        f.write('Verified: `dataset.class_to_idx[wnid] == k` iff `imagenet_r_mask[true_indices[k]] == True` '
                'and `all_wnids[true_indices[k]] == wnid` for k in 0..199.\n\n')
        f.write('Mapping rule:\n')
        f.write('- Dataset target k (from ImageFolder) corresponds to the k-th `True` position of `imagenet_r_mask`.\n')
        f.write('- That True position is the ImageNet-1K class ID of the same wnid.\n')
        f.write('- So `logits_1000[:, imagenet_r_mask]` produces a 200-dim tensor whose column k matches dataset label k.\n\n')
    else:
        f.write(f'❌ **{len(mismatches)} MISMATCHES** between dataset.class_to_idx and imagenet_r_mask:\n\n')
        for ds_idx, wnid, ti, ii in mismatches:
            f.write(f'- ds_idx={ds_idx} wnid={wnid} mask_true_indices[k]={ti} all_wnids.index(wnid)={ii}\n')

    f.write('\n## First-10 dataset class → ImageNet-1K class ID\n\n')
    f.write('| dataset class k | wnid | ImageNet-1K class ID (via all_wnids.index) | true_indices[k] from mask |\n')
    f.write('|----------------:|------|-------------------------------------------:|--------------------------:|\n')
    for k in range(min(10, len(ds.classes))):
        wnid = ds.classes[k]
        imgnet_id = all_wnid_to_idx.get(wnid, '?')
        mask_id = true_indices[k] if k < len(true_indices) else '?'
        f.write(f'| {k} | {wnid} | {imgnet_id} | {mask_id} |\n')

print(f'\nAudit saved to {out_md}')

# Save mapping JSON
mapping_path = _os.path.join(_REPO_ROOT, "outputs/imagenet_r_dataset_to_imagenet1k.json")
with open(mapping_path, 'w') as f:
    json.dump({
        'note': 'For ImageNet-R: dataset_target k (from ImageFolder, alphabetical) -> ImageNet-1K class id.',
        'len_dataset_classes': len(ds.classes),
        'len_imagenet_r_mask_True': sum(m),
        'mapping': {str(ds_idx): {'wnid': ds.classes[ds_idx], 'imagenet1k_id': all_wnid_to_idx.get(ds.classes[ds_idx], -1)}
                    for ds_idx in range(len(ds.classes))},
        'mask_true_indices': true_indices,
    }, f, indent=2)
print(f'Mapping saved to {mapping_path}')
