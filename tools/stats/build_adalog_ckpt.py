#!/usr/bin/env python
"""Build an AdaLog low-bit checkpoint (W4A4 / W3A3) for a fine-tuned, non-ImageNet ViT backbone.

Mirrors quant_library/adalog/test_quant.py `--calibrate --optimize`, the recipe behind the ImageNet
checkpoints/w{4,3}a{4,3}_checkpoint_opt.pt:
  1. wrap the FP backbone with AdaLog quantizers (reparam=True) and calibrate on cfg.calib_size (32)
     images from the source TRAIN split;
  2. wrap_reparamed_modules_in_net;
  3. block reconstruction (BlockReconstructor, cfg.optim_size=1024 train images, quant_act=cfg.train_act);
  4. finish_training (reparam_bias) and save the state_dict, which main.py loads with
     --adalog_checkpoint (wrap reparam=False, load_state_dict strict=False).
Preprocessing: CIFAR Resize((224,224), bilinear); DomainNet-126 Resize(256)+CenterCrop(224); Normalize(0.5), deterministic (AdaLog's default
in-memory calibration set would otherwise apply RandomResizedCrop to 32x32 inputs).

Example:
  python tools/stats/build_adalog_ckpt.py --dataset cifar10 --data /path/to/data/cifar10 \
     --num_classes 10 --arch_checkpoint checkpoints/vit_base_patch16_224_cifar10_nateraw_timm.pth \
     --bits 4 --out checkpoints/w4a4_checkpoint_opt_cifar10.pt
"""
import argparse
import copy
import importlib
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADALOG = os.path.join(REPO, 'quant_library', 'adalog')
# AdaLog's modules import each other as top-level packages (utils, quant_layers, quantizers, configs),
# so its root must come first on sys.path; import them before anything from the repository.
sys.path.insert(0, ADALOG)
import utils.wrap_net as wrap_net          # noqa: E402
import utils.calibrator as calibrator      # noqa: E402
import utils.block_recon as block_recon    # noqa: E402
import utils.datasets as adatasets         # noqa: E402
sys.path.append(REPO)
from quant_library.quant_utils.models import create_backbone  # noqa: E402
from torchvision import transforms  # noqa: E402


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def top1(model, loader, device, max_batches=None):
    model.eval(); c = n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        p = model(x.to(device)).argmax(1).cpu()
        c += (p == y).sum().item(); n += y.numel()
    return 100.0 * c / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cifar10', 'cifar100', 'domainnet126'], required=True)
    ap.add_argument('--data', required=True, help='ImageFolder root with train/ and val/')
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--arch', default='vit_base_patch16_224')
    ap.add_argument('--arch_checkpoint', required=True)
    ap.add_argument('--bits', type=int, choices=[3, 4], required=True)
    ap.add_argument('--seed', type=int, default=5, help='AdaLog test_quant.py default')
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_batch_size', type=int, default=200)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip_optimize', action='store_true', help='calibration only (debug)')
    args = ap.parse_args()
    device = torch.device('cuda')
    seed_all(args.seed)

    cfg_mod = importlib.import_module({4: 'configs.4bit', 3: 'configs.3bit'}[args.bits])
    cfg = cfg_mod.Config()
    cfg.w_bit = cfg.a_bit = cfg.s_bit = args.bits
    log('config:', {k: v for k, v in vars(cfg).items()})

    model = create_backbone(args.arch, num_classes=args.num_classes, checkpoint=args.arch_checkpoint).to(device).eval()
    full_model = copy.deepcopy(model).to(device).eval()

    g = adatasets.ViTImageNetLoaderGenerator(args.data, args.val_batch_size, args.workers, kwargs={"model": model})
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    if args.dataset == 'domainnet126':
        # natural images: the ImageNet-style eval transform used by the TTA loader (te_transforms)
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])
    else:
        tf = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                                 transforms.ToTensor(), norm])
    g.train_transform = tf; g.val_transform = tf      # before train_set / val_set are materialized
    val_loader = g.val_loader()
    acc, n = top1(full_model, val_loader, device); log(f'FP clean top-1: {acc:.2f} ({n})')

    model = wrap_net.wrap_modules_in_net(model, cfg, reparam=True).to(device).eval()
    log('start calibration')
    calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
    calibrator.QuantCalibrator(model, calib_loader).batching_quant_calib()
    model = wrap_net.wrap_reparamed_modules_in_net(model).to(device)
    log('calibration finished')
    acc, n = top1(model, val_loader, device); log(f'W{args.bits}A{args.bits} clean top-1 after calibration: {acc:.2f} ({n})')

    if not args.skip_optimize:
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        log('start block reconstruction')
        block_reconstructor = block_recon.BlockReconstructor(model, full_model, calib_loader)
        block_reconstructor.reconstruct_model(quant_act=cfg.train_act, keep_gpu=cfg.keep_gpu)
        log('block reconstruction finished')
    for _, m in model.named_modules():
        if hasattr(m, 'mode') and hasattr(m, 'reparam_bias'):
            m.reparam_bias()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.out)
    log(f'saved {args.out}')
    acc, n = top1(model, val_loader, device); log(f'W{args.bits}A{args.bits} clean top-1 after optimization: {acc:.2f} ({n})')


if __name__ == '__main__':
    main()
