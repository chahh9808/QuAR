#!/usr/bin/env python3
"""run_qac_alpha_ablation.py

Conservative the local branch strength ablation.

Modes:
  no_adapt           — baseline, no correction
  alpha_1.0          — output_only the local branch, alpha=1.0 (full correction)
  alpha_0.5          — output_only the local branch, alpha=0.5
  alpha_0.25         — output_only the local branch, alpha=0.25
  gated_tau0.05      — statistic-gated the local branch, tau=0.05
  gated_tau0.10      — statistic-gated the local branch, tau=0.10
  gated_tau0.20      — statistic-gated the local branch, tau=0.20

Gated alpha per layer per batch:
  a_ch, b_ch computed with alpha=1.0
  r = mean_abs((a_ch-1)*x + b_ch) / (mean_abs(x) + eps)    [using input x as proxy for y]
  alpha_l = clamp(tau / (r + eps), 0, 1)
  effective: a_eff = (1-alpha_l) + alpha_l * sig_ratio
             b_eff = alpha_l * (mu_s - sig_ratio * mu_t)

Datasets: ImageNet-C full15 (RobustBench 5k), ImageNet-Sketch, ImageNet-R, ImageNet-A
Setting: W6A6, block8_9_10_11_valid, episodic, seed 0, batch 64

Usage:
  python run_qac_alpha_ablation.py \\
      --data_c   {DATA_C} \\
      --data_r   {DATA_R} \\
      --data_s   {DATA_SKETCH} \\
      --data_a   {DATA_A} \\
      --ptq_checkpoint runs/.../ptq4vit_w6a6_calibrated.pt \\
      --stats_file    runs/.../qac_stats_w6a6_all_layers_seed0.pt \\
      --gpu 0
"""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import sys, os, time, argparse, json, math
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as D
from torch.utils.data import DataLoader

sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

EPS   = 1e-6
CLIP  = 3.0
SEVERITY = 5

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
GROUPS = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD  = [0.5, 0.5, 0.5]

_SCRIPT_DIR = _REPO_ROOT
_RB_INDICES  = os.path.join(_SCRIPT_DIR, "robustbench", "data", "imagenet_test_image_ids.txt")
_RB_CLASSMAP = os.path.join(_SCRIPT_DIR, "robustbench", "data", "imagenet_class_to_id_map.json")

with open(_RB_CLASSMAP) as _f:
    _CLASS_TO_IDX = json.load(_f)
with open(_RB_INDICES) as _f:
    _RB_FNAMES = [l.strip() for l in _f if l.strip()]


# ─── model loading ──────────────────────────────────────────────────────────

def _patch_timm_compat():
    import types
    if "timm.layers.attention" not in sys.modules:
        import timm.layers
        from timm.models.vision_transformer import Attention as VitAttention
        m = types.ModuleType("timm.layers.attention")
        m.Attention = VitAttention
        sys.modules["timm.layers.attention"] = m


def load_ptq4vit(bw, ptq_checkpoint, device=None):
    from quant_library.quant_utils.models import get_net, attention_forward, window_attention_forward
    from quant_library.quant_utils import net_wrap
    from timm.models.vision_transformer import Attention
    from timm.models.swin_transformer import WindowAttention
    from importlib import import_module, reload

    def _cfg(name):
        c = import_module(f"quant_library.configs.{name}"); reload(c); return c

    quant_cfg = _cfg("PTQ4ViT")
    quant_cfg.bit = bw
    for d in ("w_bit", "a_bit", "A_bit", "B_bit"):
        if hasattr(quant_cfg, d):
            for k in getattr(quant_cfg, d): getattr(quant_cfg, d)[k] = bw

    net = get_net("vit_base_patch16_224")
    net_wrap.wrap_modules_in_net(net, quant_cfg)
    _patch_timm_compat()
    Attention.attention_forward  = attention_forward
    WindowAttention.window_attention_forward = window_attention_forward

    net = torch.load(ptq_checkpoint, map_location="cpu", weights_only=False)
    print(f"  Checkpoint loaded: {ptq_checkpoint}")
    for m in net.modules():
        if hasattr(m, "mode"): m.mode = "quant_forward"

    if device is not None:
        net = net.to(device)
        for m in net.modules():
            for attr in ("w_interval", "a_interval", "qzero_z", "qzero_alpha",
                         "A_interval", "B_interval"):
                v = m.__dict__.get(attr)
                if torch.is_tensor(v): m.__dict__[attr] = v.to(device)
    return net


def get_scope_modules(net, scope):
    result = []
    for name, mod in net.named_modules():
        if not hasattr(mod, "a_interval"): continue
        blk = None
        for b in range(12):
            if f"blocks.{b}." in name: blk = b; break
        if blk is None or "fc2" in name: continue
        if scope == "block8_9_10_11_valid" and blk not in (8, 9, 10, 11): continue
        if scope == "qba_valid" and blk not in (10, 11): continue
        result.append((name, mod))
    return result


# ─── data loading ───────────────────────────────────────────────────────────

class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        from PIL import Image
        self._items = items; self._tfm = transform; self._Image = Image
    def __len__(self): return len(self._items)
    def __getitem__(self, i):
        path, label = self._items[i]
        img = self._Image.open(path).convert("RGB")
        return self._tfm(img), label


def corruption_loader(data_c, corruption, batch_size, severity=5):
    tfm = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(_NORM_MEAN, _NORM_STD)])
    base = os.path.join(data_c, corruption, str(severity))
    items = []
    for fn in _RB_FNAMES:
        synset = fn.split(os.sep)[0]
        if synset not in _CLASS_TO_IDX: continue
        full = os.path.join(base, fn)
        if os.path.exists(full):
            items.append((full, _CLASS_TO_IDX[synset]))
    if not items:
        raise RuntimeError(f"No images found in {base}")
    return DataLoader(_ListDataset(items, tfm), batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True, drop_last=False)


def _build_mask(data_root):
    from dataset.ImageNetMask import all_wnids
    folder_wnids = {d for d in os.listdir(data_root) if d.startswith("n")}
    return [w in folder_wnids for w in all_wnids]


def get_robustness_loader(dataset_name, data_root, batch_size, seed=0):
    """Returns (loader, mask_or_None). Always shuffled with fixed seed."""
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224),
                     T.ToTensor(), T.Normalize(_NORM_MEAN, _NORM_STD)])
    ds = D.ImageFolder(data_root, transform=tfm)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(ds)).tolist()
    ds = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)
    mask = _build_mask(data_root) if dataset_name in ("imagenet_r", "imagenet_a") else None
    return loader, mask


# ─── the local branch correction helpers ─────────────────────────────────────────────────

def _compute_qac_full(x, src, device, alpha):
    """Returns (a_ch, b_ch) of shape (C,) with given alpha scalar."""
    sig_s = src["std"].to(device).float()
    mu_s  = src["mean"].to(device).float()
    x_rs  = x.reshape(-1, x.shape[-1]).float()
    mu_t  = x_rs.mean(0)
    sig_t  = x_rs.std(0)
    sig_ratio = (sig_s / (sig_t + EPS)).clamp(1.0 / CLIP, CLIP)
    a_ch = (1.0 - alpha) + alpha * sig_ratio
    b_ch = alpha * (mu_s - sig_ratio * mu_t)
    return a_ch, b_ch


def _compute_gated_alpha(x, src, device, tau):
    """Returns (alpha_l, r) where alpha_l is the gated scalar for this batch."""
    sig_s = src["std"].to(device).float()
    mu_s  = src["mean"].to(device).float()
    x_rs  = x.reshape(-1, x.shape[-1]).float()
    mu_t  = x_rs.mean(0)
    sig_t  = x_rs.std(0)
    sig_ratio = (sig_s / (sig_t + EPS)).clamp(1.0 / CLIP, CLIP)
    # alpha=1.0 correction params
    a_1 = sig_ratio
    b_1 = mu_s - sig_ratio * mu_t
    # Correction magnitude: E[|delta|] where delta = (a_1-1)*x + b_1
    delta = (a_1 - 1.0).reshape(1, -1) * x_rs + b_1.reshape(1, -1)  # (N, C)
    r_num = delta.abs().mean()
    r_den = x_rs.abs().mean()
    r = (r_num / (r_den + EPS)).item()
    alpha_l = min(1.0, tau / (r + EPS))
    return alpha_l, r


def _to_group_params(a_ch, b_ch, module):
    """Convert per-channel (C,) params to (n_a, gs) group params."""
    n_a = module.a_interval.shape[0]
    C   = a_ch.shape[0]
    gs  = max(C // n_a, 1)
    nf  = n_a * gs
    return a_ch[:nf].reshape(n_a, gs), b_ch[:nf].reshape(n_a, gs)


# ─── metrics accumulators ───────────────────────────────────────────────────

class ECEMeter:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins; self.confs = []; self.accs = []
    def update(self, logits, targets):
        p = torch.softmax(logits.cpu().float(), 1)
        c, pred = p.max(1)
        self.confs.append(c); self.accs.append(pred == targets.cpu())
    def ece(self):
        if not self.confs: return float("nan")
        confs = torch.cat(self.confs); correct = torch.cat(self.accs).float()
        n = len(confs); ece = 0.0
        for lo, hi in zip(torch.linspace(0, 1, self.n_bins+1)[:-1],
                          torch.linspace(0, 1, self.n_bins+1)[1:]):
            m = (confs > lo) & (confs <= hi)
            if m.sum() == 0: continue
            ece += (m.sum().item() / n) * abs(correct[m].mean().item() - confs[m].mean().item())
        return ece * 100.0


class AlphaTracker:
    """Track per-layer effective alpha and correction magnitude r."""
    def __init__(self, layer_names):
        self.alphas = {n: [] for n in layer_names}
        self.rs     = {n: [] for n in layer_names}

    def record(self, name, alpha, r):
        self.alphas[name].append(alpha)
        self.rs[name].append(r)

    def summary(self):
        out = {}
        for name in self.alphas:
            a_vals = self.alphas[name]
            r_vals = self.rs[name]
            if not a_vals:
                out[name] = {}
                continue
            a_arr = np.array(a_vals); r_arr = np.array(r_vals)
            out[name] = {
                "alpha_mean": float(a_arr.mean()),
                "r_mean":     float(r_arr.mean()),
                "r_p10":      float(np.percentile(r_arr, 10)),
                "r_p50":      float(np.percentile(r_arr, 50)),
                "r_p90":      float(np.percentile(r_arr, 90)),
            }
        return out


# ─── evaluation ─────────────────────────────────────────────────────────────

def evaluate(net, loader, scope_modules, src_stats, mode_cfg, mask, device,
             tracker=None):
    """
    mode_cfg: dict with keys:
      "type": "no_adapt" | "fixed_alpha" | "gated"
      "alpha": float  (for fixed_alpha)
      "tau":   float  (for gated)

    Returns (acc_pct, ece_pct, n_correct, n_total)
    """
    net.eval()
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device) if mask else None
    qmax = 2 ** (scope_modules[0][1].a_bit - 1) if scope_modules else 128
    layer_names = [n for n, _ in scope_modules]

    correct = 0; total = 0
    ece_m = ECEMeter()

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        hooks = []
        mtype = mode_cfg["type"]

        if mtype == "no_adapt":
            pass  # no hooks

        elif mtype == "fixed_alpha":
            alpha = mode_cfg["alpha"]
            for name, m in scope_modules:
                src = src_stats.get(name, {})
                if not src: continue
                def make_pre(n_, m_, src_):
                    def pre(mod, inputs):
                        if not inputs or not torch.is_tensor(inputs[0]): return None
                        x = inputs[0]
                        a_ch, b_ch = _compute_qac_full(x.detach(), src_, device, alpha)
                        ag, bg = _to_group_params(a_ch, b_ch, mod)
                        mod._quar_post_gamma = ag.to(mod.a_interval.device, mod.a_interval.dtype)
                        mod._quar_post_beta  = bg.to(mod.a_interval.device, mod.a_interval.dtype)
                        if tracker is not None:
                            # r for fixed alpha: use the alpha=1.0 params as reference
                            _, r = _compute_gated_alpha(x.detach(), src_, device, tau=1.0)
                            tracker.record(n_, alpha, r)
                        return None
                    return pre
                def make_post(n_, m_):
                    def post(mod, inputs, output):
                        mod.__dict__.pop("_quar_post_gamma", None)
                        mod.__dict__.pop("_quar_post_beta",  None)
                    return post
                hooks.append(m.register_forward_pre_hook(make_pre(name, m, src)))
                hooks.append(m.register_forward_hook(make_post(name, m)))

        elif mtype == "gated":
            tau = mode_cfg["tau"]
            for name, m in scope_modules:
                src = src_stats.get(name, {})
                if not src: continue
                def make_pre(n_, m_, src_):
                    def pre(mod, inputs):
                        if not inputs or not torch.is_tensor(inputs[0]): return None
                        x = inputs[0]
                        alpha_l, r = _compute_gated_alpha(x.detach(), src_, device, tau)
                        a_ch, b_ch = _compute_qac_full(x.detach(), src_, device, alpha_l)
                        ag, bg = _to_group_params(a_ch, b_ch, mod)
                        mod._quar_post_gamma = ag.to(mod.a_interval.device, mod.a_interval.dtype)
                        mod._quar_post_beta  = bg.to(mod.a_interval.device, mod.a_interval.dtype)
                        if tracker is not None:
                            tracker.record(n_, alpha_l, r)
                        return None
                    return pre
                def make_post(n_, m_):
                    def post(mod, inputs, output):
                        mod.__dict__.pop("_quar_post_gamma", None)
                        mod.__dict__.pop("_quar_post_beta",  None)
                    return post
                hooks.append(m.register_forward_pre_hook(make_pre(name, m, src)))
                hooks.append(m.register_forward_hook(make_post(name, m)))

        with torch.no_grad():
            logits = net(x_batch)

        for h in hooks: h.remove()

        if mask_t is not None:
            pred = logits[:, mask_t].argmax(1)
        else:
            pred = logits.argmax(1)

        correct += (pred == y_batch).sum().item()
        total   += y_batch.size(0)
        ece_m.update(logits.cpu(), y_batch.cpu())

    acc = 100.0 * correct / max(total, 1)
    return acc, ece_m.ece(), correct, total


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_c", default=os.environ.get("DATA_C", ""))
    p.add_argument("--data_r", default=os.environ.get("DATA_R", ""))
    p.add_argument("--data_s", default=os.environ.get("DATA_SKETCH", ""))
    p.add_argument("--data_a", default=os.environ.get("DATA_A", ""))
    p.add_argument("--bw",     type=int, default=6)
    p.add_argument("--scope",  default="block8_9_10_11_valid")
    p.add_argument("--ptq_checkpoint",
        default="checkpoints/ptq4vit_w6a6_vitb.pt")
    p.add_argument("--stats_file",
        default="runs/qac_scope_sanity_v2_w6a6_allstats_seed0_4corr_20260522_164459/"
                "qac_stats_w6a6_all_layers_seed0.pt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu",        type=int, default=0)
    p.add_argument("--skip_imagenet_c", action="store_true")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"runs/qac_alpha_ablation_w{args.bw}a{args.bw}_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    def log(msg=""):
        print(msg)
        with open(log_path, "a") as f: f.write(msg + "\n")

    log(f"the local branch Alpha Ablation — W{args.bw}A{args.bw}, scope={args.scope}")
    log(f"batch_size={args.batch_size}, gpu={args.gpu}")
    log(f"ptq_checkpoint={args.ptq_checkpoint}")
    log(f"stats_file={args.stats_file}")
    log(f"output_dir={out_dir}\n")

    log("Loading model...")
    net = load_ptq4vit(args.bw, args.ptq_checkpoint, device=device)
    net.eval()
    scope_modules = get_scope_modules(net, args.scope)
    layer_names = [n for n, _ in scope_modules]
    log(f"Scope: {len(scope_modules)} modules → {layer_names}\n")

    log(f"Loading stats from {args.stats_file}...")
    raw_stats = torch.load(args.stats_file, map_location="cpu", weights_only=False)
    src_stats = {}
    for name, _ in scope_modules:
        if name in raw_stats:
            src_stats[name] = raw_stats[name]
        elif "mean" in raw_stats and name in raw_stats.get("mean", {}):
            src_stats[name] = {"mean": raw_stats["mean"][name],
                               "std":  raw_stats["std"][name]}
    log(f"Stats matched: {len(src_stats)}/{len(scope_modules)}\n")

    # Mode definitions
    modes = [
        ("no_adapt",      {"type": "no_adapt"}),
        ("alpha_1.0",     {"type": "fixed_alpha", "alpha": 1.0}),
        ("alpha_0.5",     {"type": "fixed_alpha", "alpha": 0.5}),
        ("alpha_0.25",    {"type": "fixed_alpha", "alpha": 0.25}),
        ("gated_tau0.05", {"type": "gated", "tau": 0.05}),
        ("gated_tau0.10", {"type": "gated", "tau": 0.10}),
        ("gated_tau0.20", {"type": "gated", "tau": 0.20}),
    ]

    # ── ImageNet-C full15 ───────────────────────────────────────────────────
    results_c = {}   # mode -> list of (corruption, acc, ece)
    trackers_c = {}  # mode -> AlphaTracker

    if not args.skip_imagenet_c and os.path.isdir(args.data_c):
        log("=" * 65)
        log("ImageNet-C full15 (RobustBench 5k, severity 5)")
        log("=" * 65)

        for mode_name, mode_cfg in modes:
            log(f"\n  Mode: {mode_name}")
            results_c[mode_name] = []
            t = AlphaTracker(layer_names)
            trackers_c[mode_name] = t
            t0 = time.time()

            for corruption in CORRUPTIONS:
                loader = corruption_loader(args.data_c, corruption,
                                           args.batch_size, SEVERITY)
                acc, ece, _, _ = evaluate(
                    net, loader, scope_modules, src_stats, mode_cfg,
                    mask=None, device=device,
                    tracker=(t if mode_cfg["type"] != "no_adapt" else None))
                results_c[mode_name].append((corruption, acc, ece))
                log(f"    {corruption:<25} acc={acc:.2f}%  ece={ece:.2f}%")

            mean_acc = np.mean([a for _, a, _ in results_c[mode_name]])
            mean_ece = np.mean([e for _, _, e in results_c[mode_name]])
            elapsed  = time.time() - t0
            log(f"  → mean acc={mean_acc:.2f}%  ece={mean_ece:.2f}%  ({elapsed:.0f}s)")

    # ── Robustness datasets ─────────────────────────────────────────────────
    rob_datasets = []
    if os.path.isdir(args.data_r):
        rob_datasets.append(("imagenet_r", args.data_r, "ImageNet-R"))
    if os.path.isdir(args.data_s):
        rob_datasets.append(("imagenet_sketch", args.data_s, "ImageNet-Sketch"))
    if os.path.isdir(args.data_a):
        rob_datasets.append(("imagenet_a", args.data_a, "ImageNet-A"))

    results_rob = {}   # (ds_label, mode_name) -> (acc, ece)
    trackers_rob = {}  # (ds_label, mode_name) -> AlphaTracker

    for ds_key, ds_path, ds_label in rob_datasets:
        log("\n" + "=" * 65)
        log(f"{ds_label}  ({ds_path})")
        log("=" * 65)

        loader, mask = get_robustness_loader(ds_key, ds_path, args.batch_size, seed=0)
        n_images = len(loader.dataset)
        log(f"  {n_images} images, masked={'yes' if mask else 'no'}")

        for mode_name, mode_cfg in modes:
            log(f"\n  Mode: {mode_name}")
            t = AlphaTracker(layer_names)
            trackers_rob[(ds_label, mode_name)] = t
            t0 = time.time()
            acc, ece, _, _ = evaluate(
                net, loader, scope_modules, src_stats, mode_cfg,
                mask=mask, device=device,
                tracker=(t if mode_cfg["type"] != "no_adapt" else None))
            results_rob[(ds_label, mode_name)] = (acc, ece)
            elapsed = time.time() - t0
            log(f"    acc={acc:.2f}%  ece={ece:.2f}%  ({elapsed:.0f}s)")

    # ── Report ──────────────────────────────────────────────────────────────
    log("\n\n" + "=" * 80)
    log("RESULTS SUMMARY")
    log("=" * 80)

    mode_names = [m for m, _ in modes]

    # ImageNet-C group and mean
    if results_c:
        log("\n── ImageNet-C full15 ──────────────────────────────────────────────────────")
        log(f"{'Mode':<18} {'Mean':>7} " +
            "  ".join(f"{g:>8}" for g in GROUPS.keys()) + "  ECE")
        log("-" * 75)
        ref_acc_c = None
        for mode_name in mode_names:
            if mode_name not in results_c: continue
            rows = {c: a for c, a, _ in results_c[mode_name]}
            eceval = np.mean([e for _, _, e in results_c[mode_name]])
            mean_acc = np.mean(list(rows.values()))
            group_accs = {g: np.mean([rows[c] for c in cs]) for g, cs in GROUPS.items()}
            if ref_acc_c is None: ref_acc_c = mean_acc
            delta_str = f"({mean_acc - ref_acc_c:+.2f})" if mode_name != "no_adapt" else "       "
            log(f"{mode_name:<18} {mean_acc:>7.2f}{delta_str} " +
                "  ".join(f"{group_accs[g]:>8.2f}" for g in GROUPS.keys()) +
                f"  {eceval:.2f}")

    # Per-dataset accuracy table
    log("\n── Robustness Datasets ────────────────────────────────────────────────────")
    ds_labels = [l for _, _, l in rob_datasets]

    header = f"{'Mode':<18}" + "".join(f"  {l[:12]:>12}" for l in ds_labels)
    log(header)
    log("-" * len(header))
    ref_accs_rob = {}
    for mode_name in mode_names:
        row = f"{mode_name:<18}"
        for ds_label in ds_labels:
            k = (ds_label, mode_name)
            if k in results_rob:
                acc, ece = results_rob[k]
                if mode_name == "no_adapt":
                    ref_accs_rob[ds_label] = acc
                    row += f"  {acc:>7.2f}      "
                else:
                    d = acc - ref_accs_rob.get(ds_label, 0)
                    row += f"  {acc:>7.2f}({d:+.2f})"
            else:
                row += f"  {'N/A':>12}"
        log(row)

    # Delta vs alpha_1.0
    log("\n── Delta vs alpha_1.0 (robustness datasets) ───────────────────────────────")
    ref_accs_a1 = {ds: results_rob.get((ds, "alpha_1.0"), (0, 0))[0] for ds in ds_labels}
    for mode_name in mode_names:
        if mode_name in ("no_adapt", "alpha_1.0"): continue
        row = f"{mode_name:<18}"
        for ds_label in ds_labels:
            k = (ds_label, mode_name)
            if k in results_rob:
                acc = results_rob[k][0]
                d = acc - ref_accs_a1.get(ds_label, 0)
                row += f"  {d:>+8.2f}     "
            else:
                row += f"  {'N/A':>12}"
        log(row)

    # ECE table
    log("\n── ECE (%) ────────────────────────────────────────────────────────────────")
    header_ece = f"{'Mode':<18}" + "".join(f"  {l[:12]:>12}" for l in ds_labels)
    log(header_ece)
    log("-" * len(header_ece))
    for mode_name in mode_names:
        row = f"{mode_name:<18}"
        for ds_label in ds_labels:
            k = (ds_label, mode_name)
            row += f"  {results_rob[k][1]:>7.2f}      " if k in results_rob else f"  {'N/A':>12}"
        log(row)

    # Alpha statistics for gated modes
    log("\n── Gated the local branch: Average effective alpha per dataset ─────────────────────────")
    gated_mode_names = [m for m, cfg in modes if cfg["type"] == "gated"]
    log(f"{'Mode':<18}" + "".join(f"  {l[:12]:>12}" for l in ds_labels))
    log("-" * (18 + 14 * len(ds_labels)))
    for mode_name in gated_mode_names:
        row = f"{mode_name:<18}"
        for ds_label in ds_labels:
            t = trackers_rob.get((ds_label, mode_name))
            if t:
                alphas_all = []
                for vals in t.alphas.values(): alphas_all.extend(vals)
                mean_a = np.mean(alphas_all) if alphas_all else float("nan")
                row += f"  {mean_a:>7.4f}      "
            else:
                row += f"  {'N/A':>12}"
        log(row)

    # Correction magnitude r per dataset (from gated trackers)
    log("\n── Correction magnitude r: mean/p10/p50/p90 per dataset ───────────────────")
    ref_mode = "gated_tau0.10"  # use one gated mode's r values as the reference
    for ds_label in ds_labels:
        t = trackers_rob.get((ds_label, ref_mode))
        if not t: continue
        all_r = []
        for vals in t.rs.values(): all_r.extend(vals)
        if not all_r: continue
        arr = np.array(all_r)
        log(f"  {ds_label}: r mean={arr.mean():.4f}  p10={np.percentile(arr,10):.4f}  "
            f"p50={np.percentile(arr,50):.4f}  p90={np.percentile(arr,90):.4f}")

    if results_c:
        ref_mode_c = "gated_tau0.10"
        t = trackers_c.get(ref_mode_c)
        if t:
            all_r = []
            for vals in t.rs.values(): all_r.extend(vals)
            if all_r:
                arr = np.array(all_r)
                log(f"  ImageNet-C: r mean={arr.mean():.4f}  p10={np.percentile(arr,10):.4f}  "
                    f"p50={np.percentile(arr,50):.4f}  p90={np.percentile(arr,90):.4f}")

    # Per-layer alpha statistics for gated modes on each dataset
    log("\n── Per-layer average alpha (gated_tau0.10, all datasets) ──────────────────")
    for ds_label in ds_labels:
        t = trackers_rob.get((ds_label, "gated_tau0.10"))
        if not t: continue
        log(f"  {ds_label}:")
        for name in layer_names:
            a_vals = t.alphas.get(name, [])
            r_vals = t.rs.get(name, [])
            if not a_vals: continue
            log(f"    {name:<45} alpha={np.mean(a_vals):.4f}  r={np.mean(r_vals):.4f}")

    # Key findings
    log("\n── Key findings ───────────────────────────────────────────────────────────")
    if "ImageNet-Sketch" in ds_labels and "ImageNet-A" in ds_labels:
        for mode_name in mode_names:
            if mode_name == "no_adapt": continue
            acc_s  = results_rob.get(("ImageNet-Sketch", mode_name), (0,))[0]
            acc_a  = results_rob.get(("ImageNet-A", mode_name), (0,))[0]
            ref_s  = ref_accs_rob.get("ImageNet-Sketch", 0)
            ref_a  = ref_accs_rob.get("ImageNet-A", 0)
            ref_s1 = ref_accs_a1.get("ImageNet-Sketch", 0)
            ref_a1 = ref_accs_a1.get("ImageNet-A", 0)
            gain_s = acc_s - ref_s
            regr_a = acc_a - ref_a
            diff_s = acc_s - ref_s1
            diff_a = acc_a - ref_a1
            sketch_preserved = (acc_s >= ref_s1 - 0.2)
            a_reduced = (regr_a > (ref_a1 - ref_a + 0.1))  # regression reduced by >0.1pp vs alpha=1.0
            log(f"  {mode_name:<18}: Sketch {gain_s:+.2f}pp vs no_adapt ({diff_s:+.2f} vs α=1.0), "
                f"A {regr_a:+.2f}pp vs no_adapt ({diff_a:+.2f} vs α=1.0)  "
                f"[Sketch✓={sketch_preserved} A✓={a_reduced}]")

    # Save JSON
    out = {}
    if results_c:
        out["imagenet_c"] = {}
        for mode_name in mode_names:
            if mode_name not in results_c: continue
            rows = {c: {"acc": a, "ece": e} for c, a, e in results_c[mode_name]}
            mean_acc = np.mean([a for _, a, _ in results_c[mode_name]])
            mean_ece = np.mean([e for _, _, e in results_c[mode_name]])
            out["imagenet_c"][mode_name] = {"per_corruption": rows,
                                             "mean_acc": mean_acc, "mean_ece": mean_ece}
    for ds_label, mode_name in results_rob:
        acc, ece = results_rob[(ds_label, mode_name)]
        if ds_label not in out: out[ds_label] = {}
        out[ds_label][mode_name] = {"acc": acc, "ece": ece}

    # Add alpha stats to JSON
    for ds_label in ds_labels:
        for mode_name in gated_mode_names:
            t = trackers_rob.get((ds_label, mode_name))
            if not t: continue
            if ds_label not in out: out[ds_label] = {}
            if mode_name not in out[ds_label]: out[ds_label][mode_name] = {}
            out[ds_label][mode_name]["alpha_stats"] = t.summary()

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f: json.dump(out, f, indent=2)
    log(f"\nResults saved → {json_path}")
    log(f"Log saved     → {log_path}")


if __name__ == "__main__":
    main()
