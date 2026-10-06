"""Build per-channel source CLS-token statistics for FOA-style CLS shifting.

Collects:
  pre_ln  : CLS token BEFORE final LayerNorm  (net.norm input  [:, 0])
  post_ln : CLS token AFTER  final LayerNorm  (net.head  input [:])

Usage:
  # W8A8 PTQ4ViT
  python build_cls_stats.py --data /path/to/imagenet \
    --quant --quant_w_bit 8 --quant_a_bit 8 \
    --ptq_checkpoint runs/.../ptq4vit_w8a8_timing_ckpt.pt \
    --max_batches 200 --output cls_stats_w8a8_seed0.pt

  # W4A4 AdaLog
  python build_cls_stats.py --data /path/to/imagenet \
    --adalog_checkpoint w4a4_checkpoint_opt.pt \
    --quant_w_bit 4 --quant_a_bit 4 \
    --max_batches 200 --output cls_stats_w4a4_seed0.pt
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

# ─── model loading (reuse from build_qac_stats.py) ────────────────────────────

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


def load_ptq4vit_net(checkpoint_path):
    from quant_library.quant_utils.models import (
        attention_forward as _attn_fwd,
        window_attention_forward as _win_attn_fwd,
    )
    from timm.models.vision_transformer import Attention as _Attn
    from timm.models.swin_transformer import WindowAttention as _WinAttn
    _Attn.attention_forward = _attn_fwd
    _WinAttn.window_attention_forward = _win_attn_fwd
    net = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for _m in net.modules():
        if hasattr(_m, "mode"):
            _m.mode = "quant_forward"
    # move plain-attr tensors to cuda after net.cuda() call
    return net


# ─── CLS stats collection ─────────────────────────────────────────────────────

def collect_cls_stats(net, loader, max_batches=200):
    """Collect per-channel mean/std for CLS token at pre-LN and post-LN positions.

    Returns dict:
      {
        "pre_ln":  {"mean": Tensor[768], "std": Tensor[768]},
        "post_ln": {"mean": Tensor[768], "std": Tensor[768]},
        "n_samples": int,
      }
    """
    pre_ln_feats  = []   # list of (B, 768) tensors
    post_ln_feats = []

    # pre-LN: forward PRE-hook on net.norm → inputs[0][:, 0]
    def _pre_ln_hook(module, inputs):
        x = inputs[0]   # (B, N+1, D)
        pre_ln_feats.append(x[:, 0].detach().cpu().float())

    # post-LN / classifier-input: forward PRE-hook on net.head → inputs[0]
    def _post_ln_hook(module, inputs):
        post_ln_feats.append(inputs[0].detach().cpu().float())

    h1 = net.norm.register_forward_pre_hook(_pre_ln_hook)
    h2 = net.head.register_forward_pre_hook(_post_ln_hook)

    net.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            images = images.cuda()
            net(images)
            if (i + 1) % 20 == 0:
                print(f"  batch {i+1}/{min(max_batches, len(loader))} ...")

    h1.remove()
    h2.remove()

    def _stats(feats):
        all_f = torch.cat(feats, dim=0)    # (N_total, D)
        mean  = all_f.mean(dim=0)           # (D,)
        std   = all_f.std(dim=0)            # (D,)
        return mean, std, all_f.shape[0]

    pre_mean, pre_std, n = _stats(pre_ln_feats)
    post_mean, post_std, _ = _stats(post_ln_feats)

    print(f"  Collected {n} samples (D={pre_mean.shape[0]})")
    print(f"  pre_ln  mean: [{pre_mean.min():.3f}, {pre_mean.max():.3f}]  "
          f"std: [{pre_std.min():.3f}, {pre_std.max():.3f}]")
    print(f"  post_ln mean: [{post_mean.min():.3f}, {post_mean.max():.3f}]  "
          f"std: [{post_std.min():.3f}, {post_std.max():.3f}]")

    return {
        "pre_ln":  {"mean": pre_mean,  "std": pre_std},
        "post_ln": {"mean": post_mean, "std": post_std},
        "n_samples": n,
    }


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--quant", default=False, action="store_true")
    parser.add_argument("--quant_w_bit", default=8, type=int)
    parser.add_argument("--quant_a_bit", default=8, type=int)
    parser.add_argument("--ptq_checkpoint", default=None)
    parser.add_argument("--adalog_checkpoint", default=None)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--max_batches", default=200, type=int)
    parser.add_argument("--source_shuffle", default=False, action="store_true",
                        help="shuffle the val source loader so the sampled prefix covers all 1000 classes instead of the first ~256 (val split only).")
    parser.add_argument("--source_split", default="val", choices=["val", "train"],
                        help="Source-statistic split; 'train' matches PEA's protocol.")
    parser.add_argument("--source_samples", default=10000, type=int,
                        help="Number of images when --source_split train (PEA default 10000).")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="imagenet", choices=["imagenet", "cifar10", "cifar100", "domainnet126"])
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--arch_checkpoint", default=None, type=str,
                        help="timm-format state_dict of a fine-tuned backbone")
    parser.add_argument("--arch", default="vit_base_patch16_224", type=str,
                        help="timm backbone (e.g. deit_base_patch16_224, vit_small_patch16_224).")
    args = parser.parse_args()

    use_adalog = args.adalog_checkpoint is not None
    w_bit, a_bit = args.quant_w_bit, args.quant_a_bit

    from dataset.selectedRotateImageFolder import eval_transform_for as _eval_tf
    _bb_nc = int(args.num_classes) if int(args.num_classes) != 1000 else None

    print(f"Building CLS stats: W{w_bit}A{a_bit}, "
          f"adalog={use_adalog}, max_batches={args.max_batches}")

    if use_adalog:
        net = load_adalog_net(args.adalog_checkpoint, w_bit=w_bit, a_bit=a_bit, arch=args.arch,
                              num_classes=_bb_nc, arch_checkpoint=args.arch_checkpoint)
    elif args.ptq_checkpoint:
        net = load_ptq4vit_net(args.ptq_checkpoint)
    else:
        from quant_library.quant_utils.models import create_backbone
        net = create_backbone(args.arch, num_classes=_bb_nc, checkpoint=args.arch_checkpoint)
        print(f"FP mode: using plain pretrained {args.arch}.")

    net = net.cuda()
    # move plain-attr tensors for PTQ4ViT
    _plain_attrs = ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z")
    for _m in net.modules():
        for _attr in _plain_attrs:
            _t = getattr(_m, _attr, None)
            if isinstance(_t, torch.Tensor) and not _t.is_cuda:
                setattr(_m, _attr, _t.cuda())
    net.eval()
    net.requires_grad_(False)

    if getattr(args, "source_split", "val") == "train":
        # Match PEA's protocol (see build_qac_stats.py for the rationale).
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

    stats = collect_cls_stats(net, source_loader, max_batches=args.max_batches)
    stats["w_bit"] = w_bit
    stats["a_bit"] = a_bit
    stats["max_batches"] = args.max_batches

    storage_bytes = (stats["pre_ln"]["mean"].numel() + stats["pre_ln"]["std"].numel() +
                     stats["post_ln"]["mean"].numel() + stats["post_ln"]["std"].numel()) * 4
    print(f"\nStorage: {storage_bytes} bytes ({storage_bytes/1024:.1f} KB) "
          f"for pre_ln+post_ln mean+std")

    torch.save(stats, args.output)
    print(f"Saved to '{args.output}'.")


if __name__ == "__main__":
    main()
