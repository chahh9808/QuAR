#!/usr/bin/env python3
"""Part 4 — the local branch mode timing comparison.

Measures per-batch wall-clock time for 5 the local branch modes on the same GPU/model:
  no_qac, explicit, coupled (scale_offset), decoupled (scale_offset_decoupled),
  output_only.

Usage:
  python run_qac_timing.py \
      --data /path/to/imagenet \
      --bw 8 \
      --scope qba_valid \
      --stats_file runs/.../qac_stats_w8a8_all_layers_seed0.pt \
      --ptq_checkpoint runs/.../ptq4vit_w8a8.pt \
      --n_batches 30 \
      --batch_size 64 \
      --gpu 0
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import sys, os, time, argparse, torch
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

EPS = 1e-6
ALPHA = 1.0
CLIP = 3.0

# ──── model loading (reused from run_qac_mechanism_diag.py) ────────────────────

def load_ptq4vit(data_path, bw, ptq_checkpoint=None):
    from quant_library.quant_utils.models import get_net, attention_forward, window_attention_forward
    from quant_library.quant_utils import net_wrap
    from timm.models.vision_transformer import Attention
    from timm.models.swin_transformer import WindowAttention
    from importlib import import_module, reload

    def _init_config(name):
        cfg = import_module(f"quant_library.configs.{name}")
        reload(cfg)
        return cfg

    quant_cfg = _init_config("PTQ4ViT")
    quant_cfg.bit = bw
    for dname in ("w_bit", "a_bit", "A_bit", "B_bit"):
        if hasattr(quant_cfg, dname):
            for k in getattr(quant_cfg, dname):
                getattr(quant_cfg, dname)[k] = bw

    net = get_net("vit_base_patch16_224")
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)
    Attention.attention_forward = attention_forward
    WindowAttention.window_attention_forward = window_attention_forward

    if ptq_checkpoint and os.path.isfile(ptq_checkpoint):
        net = torch.load(ptq_checkpoint, map_location="cpu", weights_only=False)
        print(f"  Checkpoint loaded: {ptq_checkpoint}")
    else:
        import quant_library.quant_utils.datasets as datasets
        from quant_library.quant_utils.quant_calib import HessianQuantCalibrator
        g = datasets.ViTImageNetLoaderGenerator(data_path, "imagenet", 32, 32, 16,
                                                kwargs={"model": net})
        calib_loader = g.calib_loader(num=32)
        quant_calibrator = HessianQuantCalibrator(
            net, wrapped_modules, calib_loader, sequential=False, batch_size=4)
        quant_calibrator.batching_quant_calib()
        print("  PTQ4ViT Hessian calibration complete.")
        if ptq_checkpoint:
            torch.save(net, ptq_checkpoint)
            print(f"  Checkpoint saved: {ptq_checkpoint}")
    for m in net.modules():
        if hasattr(m, "mode"):
            m.mode = "quant_forward"
    return net


def get_scope_modules(net, scope):
    modules = []
    for name, module in net.named_modules():
        if not hasattr(module, "a_interval"):
            continue
        blk = None
        for b in range(12):
            if f"blocks.{b}." in name:
                blk = b; break
        if blk is None:
            continue
        if "fc2" in name:
            continue
        if scope in ("last_2_blocks", "qba_valid"):
            if blk not in (10, 11):
                continue
        elif scope == "block8_9_10_11_valid":
            if blk not in (8, 9, 10, 11):
                continue
        modules.append((name, module))
    return modules


def get_imagenet_loader(data_path, batch_size, n_batches):
    import torchvision.transforms as T
    import torchvision.datasets as D
    val_dir = os.path.join(data_path, "val")
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = D.ImageFolder(val_dir, transform=tfm)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True)
    return loader


# ──── per-channel the local branch correction helpers ──────────────────────────────────────

def compute_ab_perchan(x, src, module, device):
    sig_s = src["std"].to(device)
    mu_s  = src["mean"].to(device)
    x_rs  = x.reshape(-1, x.shape[-1])
    mu_t  = x_rs.mean(dim=0)
    sig_t = x_rs.std(dim=0)
    sig_ratio = (sig_s / (sig_t + EPS)).clamp(1.0 / CLIP, CLIP)
    a_ch = (1.0 - ALPHA) + ALPHA * sig_ratio
    b_ch = ALPHA * (mu_s - sig_ratio * mu_t)
    n_a = module.a_interval.shape[0]
    C = x.shape[-1]
    gs = max(C // n_a, 1)
    nf = n_a * gs
    return a_ch[:nf].reshape(n_a, gs), b_ch[:nf].reshape(n_a, gs)


def apply_explicit_correction(x, src, module, device):
    sig_s = src["std"].to(device)
    mu_s  = src["mean"].to(device)
    x_rs  = x.reshape(-1, x.shape[-1])
    mu_t  = x_rs.mean(dim=0)
    sig_t = x_rs.std(dim=0)
    sig_ratio = (sig_s / (sig_t + EPS)).clamp(1.0 / CLIP, CLIP)
    a_ch = ((1.0 - ALPHA) + ALPHA * sig_ratio).reshape(1, 1, -1)
    b_ch = (ALPHA * (mu_s - sig_ratio * mu_t)).reshape(1, 1, -1)
    return a_ch * x + b_ch


# ──── timing harness ─────────────────────────────────────────────────────────

def time_mode(net, loader_iter, scope_modules, src_stats, mode, n_batches, device):
    """Run n_batches forward passes and return mean wall-clock time (ms)."""
    times = []
    for i in range(n_batches):
        x_batch, _ = next(loader_iter)
        x_batch = x_batch.to(device, non_blocking=True)

        if mode == "no_qac":
            hooks = []
        elif mode == "explicit":
            def make_explicit_hook(name, module):
                src = src_stats.get(name, {})
                def hook(m, inputs):
                    if not inputs:
                        return None
                    x = inputs[0]
                    if not torch.is_tensor(x):
                        return None
                    with torch.no_grad():
                        x_corr = apply_explicit_correction(x.detach(), src, module, device)
                    return (x_corr,) + inputs[1:]
                return hook
            hooks = [m.register_forward_pre_hook(make_explicit_hook(n, m))
                     for n, m in scope_modules]
        elif mode == "coupled":
            def make_coupled_hook(name, module):
                src = src_stats.get(name, {})
                interval_backup = [None]
                zz_backup = [None]
                def pre_hook(m, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    a_ch, b_ch = compute_ab_perchan(x.detach(), src, m, device)
                    # group-average for coupled
                    a_g = a_ch.mean(dim=1, keepdim=True)
                    b_g = b_ch.mean(dim=1, keepdim=True)
                    iv = m.a_interval
                    interval_backup[0] = iv.clone()
                    iv.div_(a_g.to(iv.device, iv.dtype))
                    z = (b_g / interval_backup[0]).to(iv.device, iv.dtype)
                    zz_backup[0] = getattr(m, "qzero_z", None)
                    m.qzero_z = z
                    return None
                def post_hook(m, inputs, output):
                    if interval_backup[0] is not None:
                        m.a_interval.copy_(interval_backup[0])
                    if zz_backup[0] is None:
                        m.__dict__.pop("qzero_z", None)
                    else:
                        m.qzero_z = zz_backup[0]
                return pre_hook, post_hook
            hooks = []
            for n, m in scope_modules:
                pre_h, post_h = make_coupled_hook(n, m)
                hooks.append(m.register_forward_pre_hook(pre_h))
                hooks.append(m.register_forward_hook(post_h))
        elif mode == "decoupled":
            def make_decoupled_hook(name, module):
                src = src_stats.get(name, {})
                def pre_hook(m, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    a_ch, b_ch = compute_ab_perchan(x.detach(), src, m, device)
                    m.qaffine_gamma = a_ch.to(m.a_interval.device, m.a_interval.dtype)
                    m.qaffine_beta  = b_ch.to(m.a_interval.device, m.a_interval.dtype)
                    return None
                def post_hook(m, inputs, output):
                    m.__dict__.pop("qaffine_gamma", None)
                    m.__dict__.pop("qaffine_beta",  None)
                return pre_hook, post_hook
            hooks = []
            for n, m in scope_modules:
                pre_h, post_h = make_decoupled_hook(n, m)
                hooks.append(m.register_forward_pre_hook(pre_h))
                hooks.append(m.register_forward_hook(post_h))
        elif mode == "output_only":
            def make_output_only_hook(name, module):
                src = src_stats.get(name, {})
                def pre_hook(m, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        return None
                    x = inputs[0]
                    a_ch, b_ch = compute_ab_perchan(x.detach(), src, m, device)
                    m._quar_post_gamma = a_ch.to(m.a_interval.device, m.a_interval.dtype)
                    m._quar_post_beta  = b_ch.to(m.a_interval.device, m.a_interval.dtype)
                    return None
                def post_hook(m, inputs, output):
                    m.__dict__.pop("_quar_post_gamma", None)
                    m.__dict__.pop("_quar_post_beta",  None)
                return pre_hook, post_hook
            hooks = []
            for n, m in scope_modules:
                pre_h, post_h = make_output_only_hook(n, m)
                hooks.append(m.register_forward_pre_hook(pre_h))
                hooks.append(m.register_forward_hook(post_h))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            net(x_batch)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

        for h in hooks:
            h.remove()

    # discard first 3 as warm-up
    steady = times[3:] if len(times) > 3 else times
    return sum(steady) / len(steady), min(steady), max(steady)


# ──── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--bw", type=int, default=8)
    p.add_argument("--scope", default="qba_valid")
    p.add_argument("--stats_file", required=True)
    p.add_argument("--ptq_checkpoint", default=None)
    p.add_argument("--n_batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"the local branch Timing Benchmark — W{args.bw}A{args.bw}, scope={args.scope}")
    print(f"n_batches={args.n_batches}, batch_size={args.batch_size}, GPU={args.gpu}")

    print("\nLoading PTQ4ViT model...")
    net = load_ptq4vit(args.data, args.bw, args.ptq_checkpoint)
    net = net.to(device).eval()

    scope_modules = get_scope_modules(net, args.scope)
    print(f"Scope modules ({len(scope_modules)}):", [n for n, _ in scope_modules])

    src_stats_raw = torch.load(args.stats_file, map_location="cpu", weights_only=False)
    src_stats = {}
    for name, _ in scope_modules:
        if name in src_stats_raw:
            src_stats[name] = src_stats_raw[name]
        elif "mean" in src_stats_raw and name in src_stats_raw["mean"]:
            src_stats[name] = {"mean": src_stats_raw["mean"][name],
                               "std":  src_stats_raw["std"][name]}
    print(f"Source stats matched: {len(src_stats)}/{len(scope_modules)}")

    print("\nLoading data...")
    loader = get_imagenet_loader(args.data, args.batch_size, args.n_batches)
    data_iter = iter(loader)

    # Pre-fetch enough batches to avoid DataLoader overhead
    batches = []
    for _ in range(args.n_batches + 5):
        try:
            batches.append(next(data_iter))
        except StopIteration:
            break

    print(f"\n{'Mode':<20} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Overhead':>10}")
    print("-" * 65)

    results = {}
    modes = ["no_qac", "explicit", "coupled", "decoupled", "output_only"]
    for mode in modes:
        # Use infinite iterator over pre-fetched batches
        from itertools import cycle
        loader_iter = iter(cycle(batches))
        # warm up
        for _ in range(3):
            next(loader_iter)
        mean_t, min_t, max_t = time_mode(
            net, loader_iter, scope_modules, src_stats, mode, args.n_batches, device)
        results[mode] = mean_t
        print(f"{mode:<20} {mean_t:>10.1f} {min_t:>10.1f} {max_t:>10.1f}", end="")
        if mode != "no_qac":
            overhead = mean_t - results["no_qac"]
            print(f" {overhead:>+9.1f}ms")
        else:
            print()

    print("\nRelative overhead vs no_qac:")
    base = results["no_qac"]
    for mode in modes[1:]:
        rel = (results[mode] - base) / base * 100
        print(f"  {mode}: {rel:+.1f}%")

    # Save results
    out_file = f"qac_timing_W{args.bw}A{args.bw}_{args.scope}.txt"
    with open(out_file, "w") as f:
        f.write(f"W{args.bw}A{args.bw}, scope={args.scope}, n_batches={args.n_batches}, batch_size={args.batch_size}\n")
        f.write(f"{'Mode':<20} {'Mean(ms)':>10} {'Overhead(ms)':>14}\n")
        for mode in modes:
            ov = results[mode] - results["no_qac"] if mode != "no_qac" else 0
            f.write(f"{mode:<20} {results[mode]:>10.1f} {ov:>+14.1f}\n")
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
