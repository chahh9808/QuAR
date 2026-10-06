"""Build per-channel source PERCENTILE statistics for AQR (baseline).

For each selected layer, collects source-domain per-channel percentiles
p^S_i (i in {0..n_pct-1}) of the pre-activation x, computed by running source
data through the SAME (quantized) model that will be adapted at test time.

Two placements (must match the runtime --aqr_hook):
  linear_prehook (default): x = activation entering the quantized Linear
                            (forward_pre_hook), keyed by the Linear's name
                            (e.g. blocks.10.attn.qkv). Same sites as QuAR/the local branch.
  ln_output:                x = LayerNorm output (forward_hook), keyed by the
                            norm's name (e.g. blocks.10.norm1). Paper-literal.

Tail calibration ("Average Sample Tails", paper Sec. 3.3/4.4): estimate the
source p_0 / p_100 by sampling `tail_m` values `tail_reps` times and averaging
the per-draw min / max. Enabled by default (--aqr_tail_calib avg_sample);
--aqr_tail_calib standard uses the raw reservoir min/max instead.

Saved payload:
    {"percentiles": {layer: Tensor[L, C]},
     "levels":      Tensor[L]  (in [0, 1]),
     "config":      {...}}

W6A6 (PTQ4ViT), same checkpoint as QuAR:
  python build_aqr_stats.py \\
    --data /path/to/imagenet \\
    --quant --quant_w_bit 6 --quant_a_bit 6 \\
    --ptq_checkpoint runs/.../ptq4vit_w6a6_calibrated.pt \\
    --aqr_scope block8_9_10_11_all --aqr_hook linear_prehook \\
    --max_batches 200 --output aqr_stats_w6a6_block8_9_10_11_all_seed0.pt
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import argparse
import os
import types
import torch
import torch.nn as nn

# Reuse QuAR's model construction so the quantized model is identical to the local branch.
from tools.stats.build_qac_stats import load_ptq4vit_net, load_adalog_net
from tta_library.aqr import quantile_dim0, _blocks_for_scope, _scope_excludes_fc2
from tta_library.quar import _is_quant_act, _block_idx


def load_ptq4vit_fullnet(ptq_checkpoint):
    """Load a full-object PTQ4ViT checkpoint EXACTLY as main.py does (lines
    ~1129-1137), so the quantized net is byte-for-byte identical to QuAR runs.

    These checkpoints are pickled net objects (not state dicts) and require the
    timm Attention class to be monkey-patched with `attention_forward` before
    unpickling."""
    from quant_library.quant_utils.models import (
        attention_forward as _attn_fwd,
        window_attention_forward as _win_attn_fwd,
    )
    from timm.models.vision_transformer import Attention as _Attn
    from timm.models.swin_transformer import WindowAttention as _WinAttn
    _Attn.attention_forward = _attn_fwd
    _WinAttn.window_attention_forward = _win_attn_fwd
    net = torch.load(ptq_checkpoint, map_location="cpu", weights_only=False)
    for _m in net.modules():
        if hasattr(_m, "mode"):
            _m.mode = "quant_forward"
    print(f"PTQ4ViT full-net checkpoint loaded from '{ptq_checkpoint}' "
          f"(main.py-compatible path).")
    return net


def _select_modules(net, scope, hook_point):
    blocks = _blocks_for_scope(scope)
    modules = []
    if hook_point == "ln_output":
        for name, module in net.named_modules():
            if not isinstance(module, nn.LayerNorm):
                continue
            blk = _block_idx(name)
            if blk is None:
                continue
            if blocks is not None and blk not in blocks:
                continue
            modules.append((name, module))
        return modules
    excl_fc2 = _scope_excludes_fc2(scope)
    for name, module in net.named_modules():
        if not _is_quant_act(module):
            continue
        blk = _block_idx(name)
        if blk is None:
            continue
        if blocks is not None and blk not in blocks:
            continue
        if excl_fc2 and "fc2" in name:
            continue
        modules.append((name, module))
    return modules


def avg_sample_tails(res, m, reps, gen):
    """Average Sample Tails: sample `m` values `reps` times, average per-draw
    min / max. res: [K, C] -> (p0[C], p100[C])."""
    K, C = res.shape
    mins, maxs = [], []
    done = 0
    chunk = min(reps, max(1, 100))
    while done < reps:
        r = min(chunk, reps - done)
        idx = torch.randint(0, K, (r, m), generator=gen).to(res.device)  # [r, m]
        samp = res[idx]                                                  # [r, m, C]
        mins.append(samp.min(dim=1).values)                             # [r, C]
        maxs.append(samp.max(dim=1).values)
        done += r
    p0 = torch.cat(mins, 0).mean(0)
    p100 = torch.cat(maxs, 0).mean(0)
    return p0, p100


def collect_aqr_stats(net, loader, scope, hook_point, n_pct, reservoir,
                      tail_calib, tail_m, tail_reps, max_batches, seed):
    modules = _select_modules(net, scope, hook_point)
    print(f"Selected {len(modules)} modules for scope='{scope}', hook='{hook_point}':")
    for n, _ in modules:
        print(f"  {n}")

    n_batches = max_batches if max_batches > 0 else len(loader)
    per_batch_keep = max(1, reservoir // max(1, n_batches))
    print(f"Reservoir={reservoir}, planned batches={n_batches}, "
          f"per-batch keep={per_batch_keep} rows.")

    gen = torch.Generator().manual_seed(seed)
    buffers = {name: [] for name, _ in modules}

    def _subsample(x):
        x_rs = x.detach().float().reshape(-1, x.shape[-1])  # [M, C]
        M = x_rs.shape[0]
        if M <= per_batch_keep:
            return x_rs.cpu()
        idx = torch.randperm(M, generator=gen)[:per_batch_keep].to(x_rs.device)
        return x_rs[idx].cpu()

    def make_pre_hook(name):
        def hook(module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            with torch.no_grad():
                buffers[name].append(_subsample(x))
        return hook

    def make_post_hook(name):
        def hook(module, inputs, output):
            if not torch.is_tensor(output):
                return
            with torch.no_grad():
                buffers[name].append(_subsample(output))
        return hook

    hooks = []
    for name, module in modules:
        if hook_point == "ln_output":
            hooks.append(module.register_forward_hook(make_post_hook(name)))
        else:
            hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))

    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            net(batch[0].cuda())
            if (i + 1) % 50 == 0:
                print(f"  batch {i + 1}/{n_batches}")
    for h in hooks:
        h.remove()

    levels = torch.linspace(0.0, 1.0, n_pct)
    pct_dict = {}
    print(f"\n{'Layer':<45} {'K':>8} {'C':>6} {'p0_mean':>9} {'p50_mean':>9} {'p100_mean':>9}")
    print("-" * 90)
    for name, _ in modules:
        if not buffers[name]:
            continue
        res = torch.cat(buffers[name], dim=0).cuda()  # [K, C]
        pct = quantile_dim0(res, levels.cuda())       # [L, C]
        if tail_calib == "avg_sample":
            p0, p100 = avg_sample_tails(res, tail_m, tail_reps, gen)
            # Clamp to preserve monotonicity of the percentile array.
            pct[0] = torch.minimum(p0, pct[1])
            pct[-1] = torch.maximum(p100, pct[-2])
        pct = pct.cpu()
        pct_dict[name] = pct
        mid = pct.shape[0] // 2
        print(f"  {name:<43} {res.shape[0]:>8} {res.shape[1]:>6} "
              f"{float(pct[0].mean()):>9.4f} {float(pct[mid].mean()):>9.4f} "
              f"{float(pct[-1].mean()):>9.4f}")
        del res
        torch.cuda.empty_cache()

    return pct_dict, levels


def main():
    p = argparse.ArgumentParser(description="Build AQR per-channel source percentiles.")
    p.add_argument("--data", required=True)
    p.add_argument("--adalog_checkpoint", default=None, type=str,
                   help="AdaLog checkpoint (W4A4). Omit for PTQ4ViT (W6A6/W8A8).")
    p.add_argument("--quant", default=False, action="store_true",
                   help="Use PTQ4ViT quantization (required when no adalog_checkpoint).")
    p.add_argument("--ptq_checkpoint", default=None, type=str,
                   help="Pre-calibrated PTQ4ViT state dict. Skips Hessian calibration.")
    p.add_argument("--save_ptq_checkpoint", default=None, type=str)
    p.add_argument("--quant_w_bit", default=6, type=int)
    p.add_argument("--quant_a_bit", default=6, type=int)
    p.add_argument("--arch", default="vit_base_patch16_224", type=str)
    p.add_argument("--aqr_scope", default="block8_9_10_11_all", type=str,
                   help="Scope (block set). See tta_library.aqr._blocks_for_scope.")
    p.add_argument("--aqr_hook", default="linear_prehook",
                   choices=["linear_prehook", "ln_output"],
                   help="Placement; must match runtime --aqr_hook.")
    p.add_argument("--n_pct", default=101, type=int,
                   help="Number of percentile levels (paper: 101, i.e. p0..p100).")
    p.add_argument("--reservoir", default=50000, type=int,
                   help="Max token rows kept per layer for percentile estimation.")
    p.add_argument("--tail_calib", default="avg_sample",
                   choices=["avg_sample", "standard"],
                   help="avg_sample = Average Sample Tails (paper best); "
                        "standard = raw reservoir min/max.")
    p.add_argument("--tail_m", default=100, type=int,
                   help="Sample size per tail draw (paper: 100).")
    p.add_argument("--tail_reps", default=1000, type=int,
                   help="Number of tail draws to average (paper: 1000).")
    p.add_argument("--output", default="aqr_stats_w6a6_block8_9_10_11_all_seed0.pt")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--max_batches", default=200, type=int,
                   help="Max source batches to process (-1 = full set).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    w_bit, a_bit = args.quant_w_bit, args.quant_a_bit
    use_adalog = args.adalog_checkpoint is not None
    print(f"Building AQR source percentiles: scope={args.aqr_scope}, "
          f"hook={args.aqr_hook}, W{w_bit}A{a_bit}, adalog={use_adalog}, "
          f"n_pct={args.n_pct}, tail={args.tail_calib}, max_batches={args.max_batches}")

    if use_adalog:
        net = load_adalog_net(args.adalog_checkpoint, w_bit=w_bit, a_bit=a_bit, arch=args.arch)
    elif args.ptq_checkpoint and os.path.isfile(args.ptq_checkpoint):
        # Full-object checkpoint (as saved by main.py --save_ptq_fullnet): load
        # via the identical main.py path so the model matches QuAR exactly.
        net = load_ptq4vit_fullnet(args.ptq_checkpoint)
    elif args.quant:
        net = load_ptq4vit_net(args.data, w_bit=w_bit, a_bit=a_bit,
                               ptq_checkpoint=args.ptq_checkpoint,
                               save_ptq_checkpoint=args.save_ptq_checkpoint,
                               arch=args.arch)
    else:
        # Full-precision control (paper's native setting): plain pretrained
        # backbone. No quant-act modules exist, so use --aqr_hook ln_output
        # (percentiles at LayerNorm outputs, the paper-literal placement).
        import timm
        net = timm.create_model(args.arch, pretrained=True)
        print(f"FP mode: plain pretrained {args.arch} (no quantization). "
              f"hook={args.aqr_hook} (use ln_output for FP).")

    net = net.cuda()
    # Move plain-attr tensors that .cuda() does not reach (w_interval, a_interval,
    # etc.) -- identical to main.py's post-load device fix (lines ~1150-1157).
    if args.ptq_checkpoint and os.path.isfile(args.ptq_checkpoint):
        _plain_attrs = ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z")
        for _m in net.modules():
            for _attr in _plain_attrs:
                _t = getattr(_m, _attr, None)
                if isinstance(_t, torch.Tensor) and not _t.is_cuda:
                    setattr(_m, _attr, _t.cuda())
    net.eval()
    net.requires_grad_(False)

    source_args = types.SimpleNamespace(
        data=args.data, corruption="original", level=5,
        batch_size=args.batch_size, workers=args.workers,
        imagenet_5k_eval=False, continual=False, if_shuffle=False, seed=args.seed,
    )
    from dataset.selectedRotateImageFolder import prepare_test_data
    source_dataset, source_loader = prepare_test_data(source_args)
    source_dataset.switch_mode(True, False)
    print(f"Source loader: {len(source_loader)} batches x {args.batch_size}.")

    pct_dict, levels = collect_aqr_stats(
        net, source_loader, scope=args.aqr_scope, hook_point=args.aqr_hook,
        n_pct=args.n_pct, reservoir=args.reservoir,
        tail_calib=args.tail_calib, tail_m=args.tail_m, tail_reps=args.tail_reps,
        max_batches=args.max_batches, seed=args.seed)

    payload = {
        "percentiles": pct_dict,
        "levels": levels,
        "config": {
            "scope": args.aqr_scope, "hook": args.aqr_hook, "n_pct": args.n_pct,
            "reservoir": args.reservoir, "tail_calib": args.tail_calib,
            "tail_m": args.tail_m, "tail_reps": args.tail_reps,
            "w_bit": w_bit, "a_bit": a_bit, "arch": args.arch,
            "max_batches": args.max_batches, "seed": args.seed,
        },
    }
    torch.save(payload, args.output)
    print(f"\nSaved {len(pct_dict)} layers to '{args.output}'.")


if __name__ == "__main__":
    main()
