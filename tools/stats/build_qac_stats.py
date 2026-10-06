"""Build per-channel source activation statistics for the local branch.

For each selected pre-quantizer layer, collects:
    mean[layer] : tensor [C]  — per-channel mean of x over source data
    std[layer]  : tensor [C]  — per-channel std  of x over source data

where x is the activation entering the quantizer (forward_pre_hook).

W4A4 (AdaLog):
  python build_qac_stats.py \\
    --data /path/to/imagenet \\
    --adalog_checkpoint w4a4_checkpoint_opt.pt \\
    --quant_w_bit 4 --quant_a_bit 4 \\
    --quar_local_scope all --max_batches 200 \\
    --output qac_stats_w4a4_seed0.pt

W6A6 / W8A8 (PTQ4ViT):
  python build_qac_stats.py \\
    --data /path/to/imagenet \\
    --quant --quant_w_bit 6 --quant_a_bit 6 \\
    --quar_local_scope all --max_batches 200 \\
    --output qac_stats_w6a6_seed0.pt
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import argparse
import os
import sys
import types
import torch
import torch.nn as nn
from importlib import import_module, reload


# ────────────────────────────────────────── helpers

def _is_quant_act(module):
    """Detect quantized activation module (PTQ4ViT or AdaLog)."""
    if torch.is_tensor(getattr(module, "a_interval", None)):
        if getattr(module, "a_bit", 32) < 32 and hasattr(module, "a_qmax"):
            return True
    qz = getattr(module, "a_quantizer", None)
    if qz is not None and getattr(qz, "inited", False):
        if getattr(qz, "n_bits", 32) < 32:
            return True
    return False


def _block_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "blocks":
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _scope_matches(blk, scope):
    if scope in ("last_2_blocks", "qba_valid"):
        return blk in (10, 11)
    if scope == "last_block_only":
        return blk == 11
    if scope == "attn_only":
        return blk in (10, 11)
    if scope == "all":
        return True
    return False


def _exclude_fc2(scope, name):
    if scope in ("qba_valid", "last_2_blocks", "last_block_only"):
        return "fc2" in name
    if scope == "attn_only":
        return "fc1" in name or "fc2" in name
    return False


def load_adalog_net(checkpoint_path, w_bit=4, a_bit=4, arch="vit_base_patch16_224",
                    num_classes=None, arch_checkpoint=None):
    import importlib as _importlib
    _adalog_root = os.path.join(_REPO_ROOT,
                                "quant_library", "adalog")
    _adalog_utils_init = os.path.join(_adalog_root, "utils", "__init__.py")
    if not os.path.exists(_adalog_utils_init):
        open(_adalog_utils_init, "w").close()
    if _adalog_root not in sys.path:
        sys.path.insert(0, _adalog_root)
    _shadow = ("utils", "quant_layers", "quantizers", "configs")
    _saved = {k: sys.modules.pop(k) for k in list(sys.modules.keys())
              if k in _shadow or any(k.startswith(p + ".") for p in _shadow)}
    try:
        _wrap    = _importlib.import_module("utils.wrap_net")
        _cfg_key = {(3, 3): "configs.3bit", (4, 4): "configs.4bit", (6, 6): "configs.6bit"}.get(
            (w_bit, a_bit), "configs.4bit")
        _cfg_mod = _importlib.import_module(_cfg_key)
    finally:
        for k in list(sys.modules.keys()):
            if k in _shadow or any(k.startswith(p + ".") for p in _shadow):
                sys.modules[f"_adalog_{k}"] = sys.modules.pop(k)
        sys.modules.update(_saved)

    adalog_cfg = _cfg_mod.Config()
    adalog_cfg.w_bit = w_bit
    adalog_cfg.a_bit = a_bit
    adalog_cfg.s_bit = a_bit

    from quant_library.quant_utils.models import create_backbone
    net = create_backbone(arch, num_classes=num_classes, checkpoint=arch_checkpoint)
    net = _wrap.wrap_modules_in_net(net, adalog_cfg, reparam=False)
    for _name, _mod in net.named_modules():
        if hasattr(_mod, "mode"):
            _mod.calibrated = True
            _mod.mode = "quant_forward"
        for _qa in ("a_quantizer", "w_quantizer", "A_quantizer", "B_quantizer"):
            if hasattr(_mod, _qa):
                getattr(_mod, _qa).inited = True
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    result = net.load_state_dict(ckpt, strict=False)
    print(f"AdaLog checkpoint loaded: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")
    return net


def load_ptq4vit_net(data_path, w_bit, a_bit,
                     ptq_checkpoint=None, save_ptq_checkpoint=None,
                     arch="vit_base_patch16_224", save_ptq_fullnet=None,
                     num_classes=None, arch_checkpoint=None, calib_transform=None):
    """Load PTQ4ViT quantized ViT-B.

    If ptq_checkpoint is given, skip Hessian calibration and load the saved
    calibrated state dict directly (~10x faster).  If save_ptq_checkpoint is
    given after a fresh calibration, the state dict is saved for future reuse.
    """
    from quant_library.quant_utils.models import get_net
    from quant_library.quant_utils import net_wrap
    import quant_library.quant_utils.datasets as datasets

    def _init_config(name):
        from importlib import import_module, reload
        import quant_library.quant_utils.models as _m
        _, _, files = next(os.walk("./quant_library/configs"))
        cfg = import_module(f"quant_library.configs.{name}")
        reload(cfg)
        return cfg

    quant_cfg = _init_config("PTQ4ViT")
    quant_cfg.bit = w_bit
    for dname, bval in [("w_bit", w_bit), ("a_bit", a_bit),
                        ("A_bit", a_bit), ("B_bit", a_bit)]:
        if hasattr(quant_cfg, dname):
            for k in getattr(quant_cfg, dname):
                getattr(quant_cfg, dname)[k] = bval

    net = get_net(arch, num_classes=num_classes, checkpoint=arch_checkpoint)
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)

    if ptq_checkpoint is not None and os.path.isfile(ptq_checkpoint):
        # Full-net checkpoints (as saved by --save_ptq_fullnet and used by main.py)
        # need the same attention monkey-patch main.py applies before unpickling.
        from quant_library.quant_utils.models import attention_forward as _attn_fwd
        from timm.models.vision_transformer import Attention as _Attn
        _Attn.attention_forward = _attn_fwd
        ckpt = torch.load(ptq_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            net = ckpt
            for _m in net.modules():
                if hasattr(_m, 'mode'):
                    _m.mode = 'quant_forward'
            # Plain-attr tensors that .cuda() does not reach (mirrors main.py).
            net = net.cuda()
            for _m in net.modules():
                for _attr in ('w_interval', 'a_interval', 'A_interval', 'B_interval', 'qzero_z'):
                    _t = getattr(_m, _attr, None)
                    if isinstance(_t, torch.Tensor) and not _t.is_cuda:
                        setattr(_m, _attr, _t.cuda())
                    elif isinstance(_t, (list, tuple)) and any(
                            isinstance(x, torch.Tensor) and not x.is_cuda for x in _t):
                        setattr(_m, _attr, type(_t)(
                            x.cuda() if isinstance(x, torch.Tensor) else x for x in _t))
            print(f"PTQ4ViT full net loaded from '{ptq_checkpoint}'.")
        else:
            result = net.load_state_dict(ckpt, strict=False)
            print(f"PTQ4ViT checkpoint loaded from '{ptq_checkpoint}': "
                  f"missing={len(result.missing_keys)}, "
                  f"unexpected={len(result.unexpected_keys)}")
    else:
        from quant_library.quant_utils.quant_calib import HessianQuantCalibrator
        g = datasets.ViTImageNetLoaderGenerator(data_path, "imagenet", 32, 32, 16,
                                                kwargs={"model": net, "transform": calib_transform})
        calib_loader = g.calib_loader(num=32)
        quant_calibrator = HessianQuantCalibrator(
            net, wrapped_modules, calib_loader, sequential=False, batch_size=4)
        quant_calibrator.batching_quant_calib()
        print("PTQ4ViT Hessian calibration complete.")
        if save_ptq_checkpoint:
            torch.save(net.state_dict(), save_ptq_checkpoint)
            print(f"PTQ4ViT checkpoint saved to '{save_ptq_checkpoint}'.")
        if save_ptq_fullnet:
            torch.save(net, save_ptq_fullnet)
            print(f"PTQ4ViT full-net object saved to '{save_ptq_fullnet}' "
                  f"(main.py-loadable with --ptq_checkpoint + matching --arch).")
    return net


# ────────────────────────────────────────── stats collection

_FP_TARGETS = {
    "qba_valid":       {f"blocks.{b}.{s}" for b in (10, 11)
                        for s in ("attn.qkv", "attn.proj", "mlp.fc1")},
    "last_2_blocks":   {f"blocks.{b}.{s}" for b in (10, 11)
                        for s in ("attn.qkv", "attn.proj", "mlp.fc1")},
    "last_block_only": {f"blocks.11.{s}"
                        for s in ("attn.qkv", "attn.proj", "mlp.fc1")},
    "attn_only":       {f"blocks.{b}.{s}" for b in (10, 11)
                        for s in ("attn.qkv", "attn.proj")},
}


def _get_quant_scale_qmax(module):
    """Return (scale_float, a_qmax_int, n_bits) for a quantizer module, or (None,None,None)."""
    # PTQ4ViT path: a_interval tensor, a_qmax attr
    interval = getattr(module, "a_interval", None)
    if torch.is_tensor(interval):
        a_bit = getattr(module, "a_bit", 8)
        a_qmax = int(getattr(module, "a_qmax", 2 ** (a_bit - 1)))
        scale = float(interval.detach().abs().mean())
        return scale, a_qmax, a_bit
    # AdaLog path: a_quantizer with scale
    qz = getattr(module, "a_quantizer", None)
    if qz is not None:
        n_bits = int(getattr(qz, "n_bits", 8))
        n_levels = 2 ** (n_bits - 1)
        sc = getattr(qz, "scale", None)
        scale = float(sc.detach().abs().mean()) if torch.is_tensor(sc) else 1.0
        return scale, n_levels, n_bits
    return None, None, None


def _get_neg_scale(module):
    """Post-GELU (twin-uniform) interfaces carry a second, negative-side interval.

    PTQ4ViT's PostGeluPTQSLBatchingQuantLinear quantizes x>=0 with a_interval and
    x<0 with a_neg_interval (fixed at GELU's minimum |-0.16997|/a_qmax), so a
    single symmetric grid does not describe its codes. Returns None for the
    ordinary single-scale interfaces.
    """
    neg = getattr(module, "a_neg_interval", None)
    if neg is None:
        return None
    return float(neg.detach().abs().mean()) if torch.is_tensor(neg) else float(abs(neg))


def _compute_code_hist(x_flat, scale, a_qmax, n_bits, neg_scale=None):
    """Return raw (unnormalized) code count tensor [2**n_bits] from flattened activation.

    Caller accumulates raw counts across batches and normalizes once at the end.
    This avoids the equal-weight-per-batch bias that comes from summing normalized
    per-batch histograms when batch sizes differ.
    """
    n_codes = 1 << n_bits
    if scale <= 0:
        scale = 1e-8
    xf = x_flat.float()
    if neg_scale is not None:
        # Mirror PostGeluPTQSLBatchingQuantLinear.quant_input: the positive side
        # rounds on a_interval and clamps to [0, a_qmax-1], the negative side
        # rounds on a_neg_interval and clamps to [-a_qmax, 0]; their sum is the
        # code, over the same 2**n_bits bins as the single-scale case.
        if neg_scale <= 0:
            neg_scale = 1e-8
        pos = (xf / scale).round_().clamp_(0, a_qmax - 1)
        neg = (xf / neg_scale).round_().clamp_(-a_qmax, 0)
        codes = (pos + neg).long()
    else:
        codes = (xf / scale).round_().clamp_(-a_qmax, a_qmax - 1).long()
    codes = codes + a_qmax  # shift to [0, n_codes-1]
    return torch.bincount(codes.flatten().clamp_(0, n_codes - 1), minlength=n_codes).float()


def collect_qac_stats(net, loader, scope, max_batches=-1, fp_mode=False,
                      save_code_hist=False):
    """Collect per-channel (mean, std) for each selected pre-quantizer activation.

    If save_code_hist=True and not fp_mode, also collects source quantizer code
    histograms (requires quantized model with calibrated a_interval / a_quantizer).
    """
    modules = []

    if fp_mode:
        targets = _FP_TARGETS.get(scope, set())
        for name, module in net.named_modules():
            if name in targets:
                modules.append((name, module))
    else:
        for name, module in net.named_modules():
            if not _is_quant_act(module):
                continue
            if scope == "head":
                # The classifier's own activation quantizer, the interface the
                # global CLS branch and NEO both act at. It carries no block
                # index, so every other scope skips it below.
                if name.split(".")[-1] == "head":
                    modules.append((name, module))
                continue
            blk = _block_idx(name)
            if blk is None:
                continue
            if not _scope_matches(blk, scope):
                continue
            if _exclude_fc2(scope, name):
                continue
            modules.append((name, module))

    print(f"Selected {len(modules)} modules for scope='{scope}':")
    for n, _ in modules:
        print(f"  {n}")

    accum = {}        # name -> [sum_x, sum_x2, count]
    hist_accum = {}   # name -> FloatTensor [n_codes]  (running sum, unnormalized)

    def make_hook(layer_name, mod):
        do_hist = save_code_hist and not fp_mode
        neg_scale_v = _get_neg_scale(mod) if do_hist else None
        scale_v, a_qmax_v, n_bits_v = (_get_quant_scale_qmax(mod) if do_hist
                                        else (None, None, None))

        def hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            with torch.no_grad():
                x_flat = x.detach().float().reshape(-1, x.shape[-1])  # [B*N, C]
                s_x  = x_flat.sum(dim=0).cpu()
                s_x2 = x_flat.pow(2).sum(dim=0).cpu()
                cnt  = x_flat.shape[0]
            if layer_name not in accum:
                accum[layer_name] = [s_x, s_x2, cnt]
            else:
                accum[layer_name][0] += s_x
                accum[layer_name][1] += s_x2
                accum[layer_name][2] += cnt

            if do_hist and scale_v is not None:
                with torch.no_grad():
                    h = _compute_code_hist(x.detach().flatten(), scale_v, a_qmax_v,
                                           n_bits_v, neg_scale=neg_scale_v)
                if layer_name not in hist_accum:
                    hist_accum[layer_name] = h.cpu()
                else:
                    hist_accum[layer_name] += h.cpu()
        return hook

    hooks = []
    for name, module in modules:
        h = module.register_forward_pre_hook(make_hook(name, module))
        hooks.append(h)

    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            net(batch[0].cuda())
            if (i + 1) % 50 == 0:
                n_done = i + 1
                n_total = max_batches if max_batches > 0 else len(loader)
                print(f"  batch {n_done}/{n_total}")

    for h in hooks:
        h.remove()

    mean_dict = {}
    std_dict  = {}
    print(f"\n{'Layer':<55} {'C':>5} {'mean_mean':>10} {'mean_std':>10}")
    print("-" * 85)
    for name, (s_x, s_x2, cnt) in accum.items():
        mu  = s_x  / cnt
        var = (s_x2 / cnt) - mu.pow(2)
        var = var.clamp_min(0.0)
        sig = var.sqrt()
        mean_dict[name] = mu
        std_dict[name]  = sig
        print(f"  {name:<53} {mu.shape[0]:>5} {float(mu.mean()):>10.4f} {float(sig.mean()):>10.4f}")

    # Normalize accumulated histograms
    hist_dict = {}
    if hist_accum:
        print(f"\n{'Layer':<55} {'codes':>6} {'peak_code':>10} {'sat_rate':>10}")
        print("-" * 85)
        for name, h_sum in hist_accum.items():
            h_norm = h_sum / (h_sum.sum() + 1e-12)
            hist_dict[name] = h_norm
            n_codes = len(h_norm)
            peak = int(h_norm.argmax()) - n_codes // 2
            sat = float(h_norm[0] + h_norm[-1])
            print(f"  {name:<53} {n_codes:>6} {peak:>10} {sat:>10.4f}")

    return mean_dict, std_dict, hist_dict


# ────────────────────────────────────────── main

def main():
    parser = argparse.ArgumentParser(description="Build the local branch per-channel source activation stats.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adalog_checkpoint", default=None, type=str,
                        help="AdaLog checkpoint (W4A4). Omit for PTQ4ViT path (W6A6/W8A8).")
    parser.add_argument("--quant", default=False, action="store_true",
                        help="Use PTQ4ViT quantization (required when no adalog_checkpoint).")
    parser.add_argument("--ptq_checkpoint", default=None, type=str,
                        help="Pre-calibrated PTQ4ViT state dict. Skips Hessian calibration (~8 min saved).")
    parser.add_argument("--save_ptq_checkpoint", default=None, type=str,
                        help="Save PTQ4ViT state dict after calibration for future reuse.")
    parser.add_argument("--quant_w_bit", default=4, type=int)
    parser.add_argument("--quant_a_bit", default=4, type=int)
    # "all" collects every block including fc2, which is what the statistics the paper ships
    # were built with: the adapter selects its 16 interfaces at load time.
    parser.add_argument("--quar_local_scope", default="all",
                        choices=["last_block_only", "last_2_blocks", "qba_valid", "attn_only", "all",
                                 "head"])
    parser.add_argument("--output", default="qac_stats_w4a4_seed0.pt")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--max_batches", default=200, type=int,
                        help="Max source batches to process (-1 = full set).")
    parser.add_argument("--source_shuffle", default=False, action="store_true",
                        help="shuffle the val source loader so the sampled prefix covers all 1000 classes instead of the first ~256 (val split only).")
    parser.add_argument("--source_split", default="val", choices=["val", "train"],
                        help="Source-statistic split. 'val' is the default used for the "
                             "reported stats; 'train' matches PEA's protocol (seeded random "
                             "subset of ImageNet train, no overlap with the eval images).")
    parser.add_argument("--source_samples", default=10000, type=int,
                        help="Number of images when --source_split train (PEA default 10000).")
    parser.add_argument("--save_code_hist", default=False, action="store_true",
                        help="Also collect and save per-layer source quantizer code histograms "
                             "(requires quantized model; ignored in FP mode).")
    parser.add_argument("--dataset", default="imagenet", choices=["imagenet", "cifar10", "cifar100", "domainnet126"],
                        help="source dataset family (selects the eval/calibration transform; layout stays ImageNet-like)")
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--arch_checkpoint", default=None, type=str,
                        help="timm-format state_dict of a fine-tuned backbone (see tools/models/convert_hf_vit_to_timm.py)")
    parser.add_argument("--arch", default="vit_base_patch16_224", type=str,
                        help="timm backbone (e.g. deit_base_patch16_224, vit_small_patch16_224).")
    parser.add_argument("--save_ptq_fullnet", default=None, type=str,
                        help="Save the calibrated PTQ4ViT net as a full object (main.py-loadable).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    w_bit = args.quant_w_bit
    a_bit = args.quant_a_bit
    use_adalog = args.adalog_checkpoint is not None
    from dataset.selectedRotateImageFolder import eval_transform_for as _eval_tf, calib_transform_override as _calib_tf_fn
    _bb_nc = int(args.num_classes) if int(args.num_classes) != 1000 else None
    _calib_tf = _calib_tf_fn(args)

    print(f"Building local-branch source stats: scope={args.quar_local_scope}, "
          f"W{w_bit}A{a_bit}, adalog={use_adalog}, max_batches={args.max_batches}")

    if use_adalog:
        net = load_adalog_net(args.adalog_checkpoint, w_bit=w_bit, a_bit=a_bit, arch=args.arch,
                              num_classes=_bb_nc, arch_checkpoint=args.arch_checkpoint)
    elif args.quant:
        net = load_ptq4vit_net(args.data, w_bit=w_bit, a_bit=a_bit,
                               ptq_checkpoint=args.ptq_checkpoint,
                               save_ptq_checkpoint=args.save_ptq_checkpoint,
                               arch=args.arch, save_ptq_fullnet=args.save_ptq_fullnet,
                               num_classes=_bb_nc, arch_checkpoint=args.arch_checkpoint,
                               calib_transform=_calib_tf)
    else:
        # FP control: plain pretrained backbone, select modules by name
        from quant_library.quant_utils.models import create_backbone
        net = create_backbone(args.arch, num_classes=_bb_nc, checkpoint=args.arch_checkpoint)
        print(f"FP mode: using plain pretrained {args.arch} (no quantization).")

    net = net.cuda().eval()
    net.requires_grad_(False)

    if getattr(args, "source_split", "val") == "train":
        # Match PEA's protocol: a seeded random subset of the ImageNet TRAIN split
        # (their main_vit.py samples source_stats_samples images from train_dir).
        # Removes both the class-prefix bias and the val/eval overlap of the
        # default path. Default remains 'val' so existing stats are reproducible.
        import os as _os
        from torchvision import datasets as _tvd
        from dataset.selectedRotateImageFolder import te_transforms as _te
        from dataset.selectedRotateImageFolder import arch_normalized as _arch_norm
        _full = _tvd.ImageFolder(_os.path.join(args.data, "train"), _arch_norm(args, _eval_tf(args, _te)))
        _n = min(int(args.source_samples), len(_full))
        _g = torch.Generator().manual_seed(args.seed)
        _sub, _ = torch.utils.data.random_split(_full, [_n, len(_full) - _n], generator=_g)
        source_loader = torch.utils.data.DataLoader(
            _sub, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        source_dataset = None
        print(f"Source split=TRAIN, {_n} random images (seed {args.seed}).")
    else:
        source_args = types.SimpleNamespace(
            data=args.data, corruption="original", level=5,
            batch_size=args.batch_size, workers=args.workers,
            imagenet_5k_eval=False, continual=False, dataset=args.dataset, arch=args.arch,
            if_shuffle=bool(getattr(args, 'source_shuffle', False)), seed=args.seed,
        )
        from dataset.selectedRotateImageFolder import prepare_test_data
        source_dataset, source_loader = prepare_test_data(source_args)
    if source_dataset is not None:
        source_dataset.switch_mode(True, False)
    print(f"Source loader: {len(source_loader)} batches × {args.batch_size}.")

    fp_mode = not use_adalog and not args.quant
    mean_dict, std_dict, hist_dict = collect_qac_stats(
        net, source_loader, scope=args.quar_local_scope, max_batches=args.max_batches,
        fp_mode=fp_mode, save_code_hist=args.save_code_hist)

    payload = {"mean": mean_dict, "std": std_dict}
    if hist_dict:
        payload["code_hist"] = hist_dict
        print(f"Code histograms collected for {len(hist_dict)} layers.")
    torch.save(payload, args.output)
    print(f"\nSaved {len(mean_dict)} layers to '{args.output}'.")


if __name__ == "__main__":
    main()
