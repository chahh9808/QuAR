#!/usr/bin/env python
"""Fine-tune a timm ViT on an ImageFolder dataset (source model for a non-ImageNet TTA benchmark).

Used for DomainNet-126 (source domain: real). Recipe: init from an ImageNet-21k pre-trained
timm ViT-B/16, replace the head, AdamW + cosine schedule with warm-up, AMP, label smoothing 0.1,
RandomResizedCrop(224)+flip for training and Resize(256)+CenterCrop(224) for evaluation, mean/std 0.5
(the same eval preprocessing the TTA loader applies, see dataset/selectedRotateImageFolder.te_transforms).
Saves a plain timm state_dict loadable by --arch_checkpoint. Logs every epoch's val top-1.

Example:
  python tools/prep/finetune_vit_imagefolder.py --root /path/to/data/domainnet126/real_src \
     --num_classes 126 --epochs 5 --out checkpoints/vit_base_patch16_224_domainnet126_real_timm.pth
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); c = n = 0
    for x, y in loader:
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(x.to(device, non_blocking=True)).argmax(1)
        c += (p.cpu() == y).sum().item(); n += y.numel()
    return 100.0 * c / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='ImageFolder root with train/ and val/')
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--arch', default='vit_base_patch16_224')
    ap.add_argument('--init', default='vit_base_patch16_224.augreg_in21k', help='timm pretrained tag to start from')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--warmup_frac', type=float, default=0.1)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    import timm
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    tf_train = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.35, 1.0)), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    train_ds = datasets.ImageFolder(os.path.join(args.root, 'train'), tf_train)
    val_ds = datasets.ImageFolder(os.path.join(args.root, 'val'), tf_eval)
    assert len(train_ds.classes) == args.num_classes, (len(train_ds.classes), args.num_classes)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                           pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)
    log(f'train {len(train_ds)} imgs / val {len(val_ds)} imgs / {len(train_ds.classes)} classes')

    model = timm.create_model(args.init, pretrained=True, num_classes=args.num_classes).to(device)
    log(f'init {args.init} -> head {args.num_classes}; val top-1 before fine-tuning: {evaluate(model, val_dl, device):.2f}')
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n_ in ('cls_token', 'pos_embed') else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}], lr=args.lr)
    total = args.epochs * len(train_dl); warm = int(args.warmup_frac * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler('cuda')
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best = -1.0
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if (i + 1) % 200 == 0:
                log(f'ep {ep} it {i + 1}/{len(train_dl)} loss {run / 200:.3f} lr {sched.get_last_lr()[0]:.2e}'); run = 0.0
        acc = evaluate(model, val_dl, device)
        log(f'epoch {ep}: val top-1 {acc:.2f} ({time.time() - t0:.0f}s)')
        if acc > best:
            best = acc; torch.save(model.state_dict(), args.out); log(f'saved {args.out} (best {best:.2f})')
    log(f'done; best val top-1 {best:.2f}')


if __name__ == '__main__':
    main()
