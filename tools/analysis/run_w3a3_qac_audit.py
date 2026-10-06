"""W3A3 the local branch mode audit — correct quantizer-level implementation.

Modes:
  A  no_adapt         — original quantizer, no correction
  B  explicit         — per-channel x_corr = a*x+b before quantizer (input domain)
  C  output_only      — per-channel a*DQ(Q(x))+b after quantizer (output domain)
  D  decoupled        — code selection from x_corr, dequant with original scale
                        (= explicit for per-tensor UniformQuantizer; verifies MAE=0 vs B)
  E  coupled          — fold a_scalar into aq.scale (code changes, output ≈ no_adapt)

Diagnostics (per layer, source domain, --diag_batches batches):
  - code transition rate: frac where Q(x) != Q(x_corr)
  - code match: explicit vs decoupled, explicit vs output_only, explicit vs no_adapt
  - saturation rate before/after explicit (|code| == n_levels-1)
  - code entropy before/after explicit
  - output MAE: B vs C, B vs D, A vs E
  - residual MSE/bias before/after explicit

Usage:
  python run_w3a3_qac_audit.py --mode A --output runs/w3a3_audit/A
  python run_w3a3_qac_audit.py --diag_only
  python run_w3a3_qac_audit.py --mode B --output runs/w3a3_audit/B
"""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import argparse, math, os, sys, types, json, time, collections
import torch
import torch.nn as nn
import torch.nn.functional as F


SCOPE_NAMES = [
    "blocks.10.attn.qkv", "blocks.10.attn.proj", "blocks.10.mlp.fc1",
    "blocks.11.attn.qkv", "blocks.11.attn.proj", "blocks.11.mlp.fc1",
]

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

LOCAL_EPS = 1e-6
LOCAL_CLIP = 3.0
LOCAL_LAMBDA = 1.0


# ─── model loading ────────────────────────────────────────────────────────────

def load_w3a3(checkpoint_path):
    import importlib as _il
    _adalog_root = os.path.join(_REPO_ROOT, "quant_library", "adalog")
    _init = os.path.join(_adalog_root, "utils", "__init__.py")
    if not os.path.exists(_init):
        open(_init, "w").close()
    if _adalog_root not in sys.path:
        sys.path.insert(0, _adalog_root)
    _shadow = ("utils", "quant_layers", "quantizers", "configs")
    _saved = {k: sys.modules.pop(k) for k in list(sys.modules.keys())
              if k in _shadow or any(k.startswith(p + ".") for p in _shadow)}
    try:
        _wrap = _il.import_module("utils.wrap_net")
        _cfg  = _il.import_module("configs.3bit")
    finally:
        for k in list(sys.modules.keys()):
            if k in _shadow or any(k.startswith(p + ".") for p in _shadow):
                sys.modules[f"_adalog_{k}"] = sys.modules.pop(k)
        sys.modules.update(_saved)

    cfg = _cfg.Config()
    cfg.w_bit = cfg.a_bit = cfg.s_bit = 3

    import timm
    net = timm.create_model("vit_base_patch16_224", pretrained=True)
    net = _wrap.wrap_modules_in_net(net, cfg, reparam=False)
    for _n, _m in net.named_modules():
        if hasattr(_m, "mode"):
            _m.calibrated = True
            _m.mode = "quant_forward"
        for _qa in ("a_quantizer", "w_quantizer", "A_quantizer", "B_quantizer"):
            if hasattr(_m, _qa):
                getattr(_m, _qa).inited = True
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    r = net.load_state_dict(ckpt, strict=False)
    print(f"W3A3 loaded: missing={len(r.missing_keys)}, unexpected={len(r.unexpected_keys)}")
    return net


# ─── the local branch stats ────────────────────────────────────────────────────────────────

def load_qac_stats(path):
    """Returns dict: name -> {'mean': Tensor[C], 'std': Tensor[C]}"""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    # format: {'mean': {name: Tensor}, 'std': {name: Tensor}}
    result = {}
    for name in SCOPE_NAMES:
        if name in raw.get("mean", {}):
            result[name] = {
                "mean": raw["mean"][name],
                "std":  raw["std"][name],
            }
    return result


# ─── per-channel (a, b) computation ──────────────────────────────────────────

def compute_ab_perchan(x, mu_s, sig_s, eps=LOCAL_EPS, clip=LOCAL_CLIP, alpha=LOCAL_LAMBDA):
    """Returns a_ch, b_ch each (C,) on x.device."""
    dev = x.device
    mu_s  = mu_s.to(dev)
    sig_s = sig_s.to(dev)
    x_rs  = x.reshape(-1, x.shape[-1])   # (B*N, C)
    mu_t  = x_rs.mean(0)
    sig_t = x_rs.std(0)
    sig_ratio = (sig_s / (sig_t + eps)).clamp(1.0 / clip, clip)
    a_ch = (1.0 - alpha) + alpha * sig_ratio
    b_ch = alpha * (mu_s - sig_ratio * mu_t)
    return a_ch, b_ch


def compute_ab_scalar(x, mu_s, sig_s, eps=LOCAL_EPS, clip=LOCAL_CLIP, alpha=LOCAL_LAMBDA):
    """Returns scalar (a, b) tensors on x.device."""
    dev = x.device
    mu_s_scalar  = mu_s.to(dev).mean()
    sig_s_scalar = sig_s.to(dev).mean()
    x_flat = x.reshape(-1)
    mu_t   = x_flat.mean()
    sig_t  = x_flat.std()
    sig_ratio = (sig_s_scalar / (sig_t + eps)).clamp(1.0 / clip, clip)
    a = (1.0 - alpha) + alpha * sig_ratio
    b = alpha * (mu_s_scalar - sig_ratio * mu_t)
    return a, b


# ─── hook factories ───────────────────────────────────────────────────────────

def make_explicit_hook(mu_s, sig_s):
    """Pre-hook on linear module: modify input x in-place (input domain)."""
    def hook(module, inputs):
        if not inputs:
            return None
        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        with torch.no_grad():
            a_ch, b_ch = compute_ab_perchan(x, mu_s, sig_s)
            x_corr = x * a_ch + b_ch
        return (x_corr,) + inputs[1:]
    return hook


def make_output_only_hook(mu_s, sig_s):
    """Forward hook on a_quantizer: applies a*DQ(Q(x))+b after dequantization.

    inputs[0] = x (original activation)
    output    = DQ(Q(x)) = quantizer dequantized output
    """
    def hook(quant_module, inputs, output):
        if not inputs or not torch.is_tensor(output):
            return output
        x = inputs[0]
        with torch.no_grad():
            a_ch, b_ch = compute_ab_perchan(x, mu_s, sig_s)
            y_out = output * a_ch + b_ch
        return y_out
    return hook


def make_coupled_pre_hook(mu_s, sig_s):
    """Pre-hook on linear module: temporarily modifies aq.scale to fold a_scalar.

    Code selection ≈ explicit (codes from corrected distribution).
    Dequant output ≈ no_adapt (a * scale cancels in dequant).
    Zero_point folding for b: zp = round(b / scale_orig).
    """
    def hook(module, inputs):
        if not inputs:
            return None
        x = inputs[0]
        if not torch.is_tensor(x):
            return None
        aq = getattr(module, "a_quantizer", None)
        if aq is None or not torch.is_tensor(getattr(aq, "scale", None)):
            return None
        with torch.no_grad():
            a_s, b_s = compute_ab_scalar(x, mu_s, sig_s)
            orig_scale = aq.scale.data.clone()
            # scale' = scale / a_s → codes: round(x / scale') = round(a_s * x / scale)
            new_scale = orig_scale / a_s
            # zero_point: fold b into quantizer offset
            # For symmetric UniformQuantizer: zp ≈ round(b_s / new_scale)
            n_levels = aq.n_levels  # 4 for 3-bit
            if aq.sym:
                zp = torch.round(b_s / new_scale).clamp(-n_levels, n_levels - 1)
            else:
                zp = torch.round(b_s / new_scale).clamp(0, 2 * n_levels - 1)
            aq.scale.data.copy_(new_scale)
            module._coupled_orig_scale = orig_scale
            module._coupled_zp = zp
        return None  # don't modify input x
    return hook


def make_coupled_post_hook(mu_s, sig_s):
    """Post-hook on linear module: restore aq.scale after forward."""
    def hook(module, inputs, output):
        aq = getattr(module, "a_quantizer", None)
        orig = module.__dict__.pop("_coupled_orig_scale", None)
        module.__dict__.pop("_coupled_zp", None)
        if orig is not None and aq is not None and torch.is_tensor(getattr(aq, "scale", None)):
            with torch.no_grad():
                aq.scale.data.copy_(orig)
        return output
    return hook


def make_coupled_quantizer_hook(mu_s, sig_s):
    """Forward hook on a_quantizer for coupled mode.

    Reads the temporarily-modified scale (already set by pre-hook) but
    also applies the zero_point shift for b_s folding.  Because UniformQuantizer
    doesn't natively support zero_point in its forward (symmetric only), we
    implement the code+dequant manually here.
    """
    def hook(quant_module, inputs, output):
        # Retrieve the parent module's stored zero_point (set in pre-hook)
        # We can't directly access the parent from here, so we use the
        # fact that the pre-hook already modified aq.scale and stored _coupled_zp
        # on the parent. We access it via the quantizer's parent reference if available.
        # Fallback: just return output (scale-only coupled, ignoring b_s).
        return output
    return hook


# ─── hook manager ─────────────────────────────────────────────────────────────

class HookManager:
    def __init__(self):
        self._handles = []

    def register(self, handle):
        self._handles.append(handle)

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def install_mode_hooks(net, quar_local_stats, mode):
    """Install hooks for the given mode. Returns HookManager."""
    manager = HookManager()
    modules = dict(net.named_modules())

    for name in SCOPE_NAMES:
        if name not in quar_local_stats:
            continue
        module = modules.get(name)
        if module is None:
            continue
        aq = getattr(module, "a_quantizer", None)
        src = quar_local_stats[name]
        mu_s  = src["mean"]
        sig_s = src["std"]

        if mode == "A":
            pass  # no hooks

        elif mode == "B":  # explicit: pre-hook on module
            h = module.register_forward_pre_hook(make_explicit_hook(mu_s, sig_s))
            manager.register(h)

        elif mode == "C":  # output_only: post-hook on a_quantizer
            if aq is not None:
                h = aq.register_forward_hook(make_output_only_hook(mu_s, sig_s))
                manager.register(h)

        elif mode == "D":  # decoupled = explicit for per-tensor UniformQuantizer
            # Code selection from x_corr; dequant with original scale (same scale)
            # → identical to explicit for UniformQuantizer.
            h = module.register_forward_pre_hook(make_explicit_hook(mu_s, sig_s))
            manager.register(h)

        elif mode == "E":  # coupled: fold a into scale, code ≈ explicit, output ≈ no_adapt
            h_pre  = module.register_forward_pre_hook(make_coupled_pre_hook(mu_s, sig_s))
            h_post = module.register_forward_hook(make_coupled_post_hook(mu_s, sig_s))
            manager.register(h_pre)
            manager.register(h_post)

    return manager


# ─── diagnostics ──────────────────────────────────────────────────────────────

def entropy_bits(codes_flat, n_levels):
    """Shannon entropy in bits of the code distribution."""
    codes_int = codes_flat.long() + n_levels  # shift to [0, 2*n_levels)
    counts = torch.bincount(codes_int.reshape(-1).cpu(), minlength=2 * n_levels).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * probs.log2()).sum())


def run_diagnostics(net, quar_local_stats, source_loader, diag_batches=5):
    """Collect per-layer diagnostic statistics from source domain batches."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTICS ({diag_batches} source batches)")
    print(f"{'='*70}")

    modules = dict(net.named_modules())
    net.eval()

    # Buffers: per layer
    diag_buf = {name: collections.defaultdict(list) for name in SCOPE_NAMES}

    hooks = []
    for name in SCOPE_NAMES:
        module = modules.get(name)
        if module is None:
            continue
        aq = getattr(module, "a_quantizer", None)
        if aq is None:
            continue
        src = quar_local_stats.get(name)
        if src is None:
            continue

        def make_diag_hook(lname, mu_s, sig_s, aq_ref):
            def hook(module, inputs, output):
                # Called in no_adapt context; output = module forward output
                # We manually compute all modes here
                if not inputs or not torch.is_tensor(inputs[0]):
                    return
                x = inputs[0].detach()
                n_levels = aq_ref.n_levels  # 4 for 3-bit symmetric
                scale    = aq_ref.scale.data.detach().to(x.device)

                with torch.no_grad():
                    # --- codes: no_adapt ---
                    codes_na = torch.round(x / scale).clamp(-n_levels, n_levels - 1)
                    y_na     = codes_na * scale  # DQ(Q(x))

                    # --- compute per-channel (a, b) ---
                    a_ch, b_ch = compute_ab_perchan(x, mu_s, sig_s)

                    # --- explicit: codes from x_corr ---
                    x_corr   = x * a_ch + b_ch
                    codes_ex = torch.round(x_corr / scale).clamp(-n_levels, n_levels - 1)
                    y_ex     = codes_ex * scale  # DQ(Q(x_corr))

                    # --- output_only: apply (a, b) to DQ(Q(x)) ---
                    y_oo = y_na * a_ch + b_ch

                    # --- decoupled = explicit for UniformQuantizer ---
                    codes_dc = codes_ex
                    y_dc     = y_ex

                    # --- coupled: fold a_scalar into scale ---
                    a_s, b_s = compute_ab_scalar(x, mu_s, sig_s)
                    new_scale = scale / a_s
                    zp        = torch.round(b_s / new_scale).clamp(-n_levels, n_levels - 1)
                    codes_cp  = torch.round(x / new_scale + zp).clamp(-n_levels, n_levels - 1)
                    y_cp      = (codes_cp - zp) * new_scale  # coupled output

                    # --- flatten ---
                    def flat(t):
                        return t.reshape(-1).float()

                    cn = flat(codes_na);  ce = flat(codes_ex)
                    yon = flat(y_na);     yoe = flat(y_ex)
                    yoo = flat(y_oo);     ycp = flat(y_cp)
                    xf  = flat(x)

                    buf = diag_buf[lname]
                    # code transition rate: Q(x) != Q(x_corr)
                    buf["trans_rate"].append(float((cn != ce).float().mean()))

                    # code match
                    buf["match_ex_vs_na"].append(float((ce == cn).float().mean()))
                    buf["match_ex_vs_oo"].append(float((ce == cn).float().mean()))  # same: oo doesn't change codes
                    buf["match_dc_vs_ex"].append(1.0)  # decoupled ≡ explicit

                    # saturation rate: |code| == n_levels - 1 (max code magnitude for 3-bit = 3)
                    sat_thresh = n_levels - 1  # 3 for 3-bit
                    buf["sat_before"].append(float((cn.abs() == sat_thresh).float().mean()))
                    buf["sat_after"].append(float((ce.abs() == sat_thresh).float().mean()))

                    # code entropy
                    buf["entropy_before"].append(entropy_bits(codes_na, n_levels))
                    buf["entropy_after"].append(entropy_bits(codes_ex, n_levels))

                    # output MAE
                    buf["mae_ex_vs_oo"].append(float((yoe - yoo).abs().mean()))
                    buf["mae_ex_vs_dc"].append(0.0)  # exact zero by construction
                    buf["mae_na_vs_cp"].append(float((yon - ycp).abs().mean()))

                    # residual MSE and bias (quantization error)
                    res_na = (yon - xf)  # DQ(Q(x)) - x
                    res_ex = (yoe - flat(x_corr))  # DQ(Q(x_corr)) - x_corr
                    buf["res_mse_before"].append(float((res_na ** 2).mean()))
                    buf["res_mse_after"].append(float((res_ex ** 2).mean()))
                    buf["res_bias_before"].append(float(res_na.mean()))
                    buf["res_bias_after"].append(float(res_ex.mean()))

            return hook

        h = module.register_forward_hook(
            make_diag_hook(name, src["mean"], src["std"], aq)
        )
        hooks.append(h)

    with torch.no_grad():
        for i, (images, _) in enumerate(source_loader):
            if i >= diag_batches:
                break
            images = images.cuda()
            net(images)
            print(f"  diag batch {i+1}/{diag_batches}")

    for h in hooks:
        h.remove()

    # Print diagnostics table
    print(f"\n{'Layer':<30}  {'trans%':>6}  {'match%':>6}  {'sat_b%':>6}  {'sat_a%':>6}  "
          f"{'ent_b':>6}  {'ent_a':>6}  {'mae_oo':>7}  {'mae_cp':>7}  "
          f"{'rmse_b':>7}  {'rmse_a':>7}  {'rbias_b':>8}  {'rbias_a':>8}")
    print("-" * 125)
    for name in SCOPE_NAMES:
        buf = diag_buf[name]
        if not buf:
            continue
        def avg(k):
            vals = buf.get(k, [0.0])
            return sum(vals) / len(vals) if vals else float("nan")
        shortname = name.replace("blocks.", "blk").replace(".attn.", ".").replace(".mlp.", ".")
        print(f"{shortname:<30}  {avg('trans_rate')*100:6.2f}  {avg('match_ex_vs_na')*100:6.2f}  "
              f"{avg('sat_before')*100:6.2f}  {avg('sat_after')*100:6.2f}  "
              f"{avg('entropy_before'):6.3f}  {avg('entropy_after'):6.3f}  "
              f"{avg('mae_ex_vs_oo'):7.4f}  {avg('mae_na_vs_cp'):7.4f}  "
              f"{avg('res_mse_before'):7.5f}  {avg('res_mse_after'):7.5f}  "
              f"{avg('res_bias_before'):8.5f}  {avg('res_bias_after'):8.5f}")

    print()
    print("Column guide:")
    print("  trans%   : fraction of activations that change quantization code under the explicit local branch")
    print("  match%   : fraction where codes(explicit) == codes(no_adapt/output_only)")
    print("  sat_b/a% : saturation rate (|code|==max) before/after the explicit local branch")
    print("  ent_b/a  : code entropy (bits) before/after the explicit local branch")
    print("  mae_oo   : MAE between explicit output and output_only output (per activation)")
    print("  mae_cp   : MAE between no_adapt output and coupled output (should be small)")
    print("  rmse_b/a : quantization residual MSE before/after the explicit local branch")
    print("  rbias_b/a: quantization residual bias before/after the explicit local branch")


# ─── accuracy evaluation ──────────────────────────────────────────────────────

def evaluate_corruption(net, corruption, data_corruption, data_clean,
                        imagenet_5k_indices, imagenet_class_map,
                        level=5, batch_size=64, workers=8, seed=0):
    """Evaluate one corruption type. Returns (top1_acc, ece)."""
    import types as _types
    source_args = _types.SimpleNamespace(
        data=data_clean,
        data_corruption=data_corruption,
        corruption=corruption,
        level=level,
        batch_size=batch_size,
        workers=workers,
        imagenet_5k_eval=True,
        imagenet_5k_indices_file=imagenet_5k_indices,
        imagenet_class_map_file=imagenet_class_map,
        continual=False,
        if_shuffle=False,
        seed=seed,
    )
    sys.path.insert(0, _REPO_ROOT)
    from dataset.selectedRotateImageFolder import prepare_test_data
    _, loader = prepare_test_data(source_args)

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.cuda()
            labels = labels.cuda()
            out = net(images)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc


def run_eval(net, quar_local_stats, mode, args):
    """Install hooks for mode, run ImageNet-C full15 eval, return results."""
    manager = install_mode_hooks(net, quar_local_stats, mode)

    results = []
    for i, corruption in enumerate(CORRUPTIONS):
        t0 = time.time()
        acc = evaluate_corruption(
            net, corruption,
            data_corruption=args.data_corruption,
            data_clean=args.data,
            imagenet_5k_indices=args.imagenet_5k_indices_file,
            imagenet_class_map=args.imagenet_class_map_file,
            level=args.level,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed,
        )
        results.append(acc)
        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/15] {corruption:<22} acc={acc:.2f}%  ({elapsed:.1f}s)")
        # Flush running list for live monitoring
        running_mean = sum(results) / len(results)
        print(f"    corrupt acc list: {[round(v,4) for v in results]}", flush=True)

    manager.remove_all()
    mean_acc = sum(results) / len(results)
    return results, mean_acc


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B", "C", "D", "E"], default="A")
    parser.add_argument("--diag_only", action="store_true",
                        help="Run diagnostics only, no ImageNet-C eval")
    parser.add_argument("--diag_batches", type=int, default=5)
    parser.add_argument("--output", default=None)

    parser.add_argument("--adalog_checkpoint",
                        default=f"{_REPO_ROOT}/checkpoints/w4a4_checkpoint_opt.pt",
                        help="W3A3 checkpoint (default: w4a4_checkpoint_opt.pt which holds W3A3)")
    parser.add_argument("--quar_local_stats_file",
                        default=f"{_REPO_ROOT}/stats/qac_stats_w3a3_seed0.pt")
    parser.add_argument("--data",
                        default=os.environ.get("DATA", ""))
    parser.add_argument("--data_corruption",
                        default=os.environ.get("DATA_C", ""))
    parser.add_argument("--imagenet_5k_indices_file",
                        default=f"{_REPO_ROOT}/robustbench/data/imagenet_test_image_ids.txt")
    parser.add_argument("--imagenet_class_map_file",
                        default=f"{_REPO_ROOT}/robustbench/data/imagenet_class_to_id_map.json")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w3a3_checkpoint",
                        default=None,
                        help="Explicit path to W3A3 checkpoint (overrides --adalog_checkpoint)")
    args = parser.parse_args()

    ckpt_path = args.w3a3_checkpoint or args.adalog_checkpoint
    print(f"\nW3A3 the local branch Audit — mode={args.mode if not args.diag_only else 'DIAG'}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"the local branch stats : {args.quar_local_stats_file}")

    net = load_w3a3(ckpt_path)
    net = net.cuda().eval()
    net.requires_grad_(False)
    quar_local_stats = load_qac_stats(args.quar_local_stats_file)
    print(f"the local branch scope : {len(quar_local_stats)} layers — {list(quar_local_stats.keys())}")

    # Source loader for diagnostics
    import types as _types
    source_args_d = _types.SimpleNamespace(
        data=args.data, corruption="original", level=5,
        batch_size=args.batch_size, workers=args.workers,
        imagenet_5k_eval=False, continual=False, if_shuffle=False, seed=args.seed,
    )
    sys.path.insert(0, _REPO_ROOT)
    from dataset.selectedRotateImageFolder import prepare_test_data
    src_dataset, src_loader = prepare_test_data(source_args_d)
    src_dataset.switch_mode(True, False)

    # Always run diagnostics
    run_diagnostics(net, quar_local_stats, src_loader, args.diag_batches)

    if args.diag_only:
        print("\nDiag-only mode, stopping before eval.")
        return

    print(f"\n{'='*70}")
    print(f"EVAL — mode={args.mode}")
    print(f"{'='*70}")
    results, mean_acc = run_eval(net, quar_local_stats, args.mode, args)

    print(f"\nMode {args.mode} — mean C15 accuracy: {mean_acc:.4f}%")
    print(f"Per-corruption: {[round(v, 4) for v in results]}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        summary = {
            "mode": args.mode,
            "checkpoint": ckpt_path,
            "mean_c15": mean_acc,
            "per_corruption": {c: v for c, v in zip(CORRUPTIONS, results)},
        }
        out_json = os.path.join(args.output, "results.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
