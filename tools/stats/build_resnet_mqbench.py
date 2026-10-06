"""Quantize ZOA's torchvision-style ResNet-50 with MQBench, exactly as ZOA does.

Why this exists. ZOA quantizes ViTs with PTQ4ViT but CNNs with MQBench
(ZOA_official/quant_model.py L27 vs L41), and its ResNet adaptation code
(models/fuse_resnet.py) is written against the MQBench FX-traced graph: it looks
up `layer{i}_{j}_relu_post_act_fake_quantizer` modules and relies on the traced
graph dropping the `downsample=None` attribute. On an eager PTQ4ViT-quantized
model neither holds, so ZOA cannot run. Rebuilding the artefact with MQBench is
what lets ZOA run unmodified.

Faithfulness. The config and the calibration sequence below are copied from
ZOA_official/quant_library/configs/PTQ4CNN.py: MSEObserver for weights,
EMAMSEObserver for activations, asymmetric per-channel weights, asymmetric
per-tensor activations, BackendType.Academic, calibrate then enable_quantization.
AdaRound reconstruction is skipped, matching upstream, which only enables it for
bit <= 4; at W6A6 upstream takes the same no-reconstruction path.

Environment note. MQBench 0.0.6 was already present in this environment before
this work started (it is in the pre-existing package snapshot), and torch stays
at 2.8.0. Nothing was installed or upgraded.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, _REPO_ROOT)
# ZOA's ResNet-50 is vendored as models/zoa_resnet.py; mqbench is a pip package.

from mqbench.prepare_by_platform import prepare_by_platform, BackendType  # noqa: E402
from mqbench.utils.state import enable_calibration, enable_quantization  # noqa: E402
from models import zoa_resnet as ZOAResNet  # noqa: E402
from quant_library.quant_utils import datasets as qdatasets  # noqa: E402


def build_backbone(arch):
    if arch == 'zoa_resnet50':
        return ZOAResNet.resnet50(pretrained=True)
    import timm
    return timm.create_model(arch, pretrained=True)


def mqbench_config(bit):
    """Verbatim from ZOA_official PTQ4CNN.extra_config, with bit substituted."""
    return {
        'extra_qconfig_dict': {
            'w_observer': 'MSEObserver',
            'a_observer': 'EMAMSEObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
            'w_qscheme': {'bit': bit, 'symmetry': False, 'per_channel': True, 'pot_scale': False},
            'a_qscheme': {'bit': bit, 'symmetry': False, 'per_channel': False, 'pot_scale': False},
        }
    }


@torch.no_grad()
def top1(model, loader, n_images, tag):
    model.eval()
    correct = total = 0
    for x, y in loader:
        if total >= n_images:
            break
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    acc = 100.0 * correct / max(total, 1)
    print(f"  {tag:10s} top-1 = {acc:.2f}  (n={total})", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get("DATA", ""))
    ap.add_argument('--arch', default='zoa_resnet50',
                    help="zoa_resnet50 for ZOA's torchvision-style model, or any timm name")
    ap.add_argument('--bit', type=int, default=6)
    ap.add_argument('--calib_batches', type=int, default=32)
    ap.add_argument('--n_images', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    print(f"=== {args.arch} | MQBench W{args.bit}A{args.bit} | "
          f"Academic backend, {args.calib_batches} calib batches ===", flush=True)

    net_fp = build_backbone(args.arch).cuda().eval()
    print(f"[check] bottleneck exposes .relu = {hasattr(net_fp.layer1[0], 'relu')} "
          f"(ZOA's fuse_resnet requires torchvision naming)")

    # ImageNetLoaderGenerator uses ImageNet mean/std, which is what a torchvision
    # ResNet-50 expects; ZOA passes --use_in1k_norm for the same reason.
    g = qdatasets.ImageNetLoaderGenerator(args.data, 'imagenet', 32, 64, 8)

    torch.manual_seed(args.seed)
    acc_fp = top1(net_fp, g.test_loader(shuffle=True), args.n_images, 'FP')

    model = prepare_by_platform(copy.deepcopy(net_fp), BackendType.Academic,
                                mqbench_config(args.bit)).cuda()
    del net_fp
    torch.cuda.empty_cache()

    enable_calibration(model)
    model.eval()
    print(f"[calib] {args.calib_batches} batches ...", flush=True)
    with torch.no_grad():
        for i, (x, _) in enumerate(g.calib_loader(num=args.calib_batches)):
            if i >= args.calib_batches:
                break
            model(x.cuda())
    enable_quantization(model)

    n_fq = sum(1 for n, _ in model.named_modules() if n.endswith('post_act_fake_quantizer'))
    print(f"[quant] {n_fq} activation fake-quantizers active")

    torch.manual_seed(args.seed)
    acc_q = top1(model, g.test_loader(shuffle=True), args.n_images, f'W{args.bit}A{args.bit}')

    print("\n===== SANITY GATE (MQBench) =====")
    print(f"  FP    clean top-1 = {acc_fp:.2f}")
    print(f"  W{args.bit}A{args.bit}  clean top-1 = {acc_q:.2f}")
    print(f"  drop              = {acc_fp - acc_q:.2f}")

    if args.out:
        # An MQBench GraphModule cannot be pickled whole: prepare_by_platform
        # attaches qconfig factory closures (_add_module_to_qconfig_obs_ctr.<locals>
        # .get_factory_kwargs_based_on_module_device). ZOA hits the same limit and
        # saves only the state dict (quant_model.py L51); rebuild_mqbench_model()
        # below reconstructs the graph and loads it back.
        torch.save(model.state_dict(), args.out)
        print(f"[save] state_dict -> {args.out}")


def rebuild_mqbench_model(ckpt, bit=6, arch='zoa_resnet50'):
    """Reconstruct the traced+quantized graph and load a saved state dict."""
    net = build_backbone(arch)
    model = prepare_by_platform(net, BackendType.Academic, mqbench_config(bit)).cuda()
    enable_quantization(model)
    sd = torch.load(ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    return model.eval()


if __name__ == '__main__':
    main()
