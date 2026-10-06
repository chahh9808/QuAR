#!/usr/bin/env python3
"""run_qac_code_reassignment_diag.py

Code-reassignment stress diagnostic on ImageNet-R, ImageNet-Sketch, ImageNet-A.

Four modes:
  no_adapt      — baseline quantized model, no correction
  output_only   — per-channel affine correction AFTER dequantization (_quar_post_gamma/beta)
  decoupled     — per-channel affine correction INSIDE quantizer (qaffine_gamma/beta),
                  changes codes + output; termed "code_plus_output"
  coupled       — modify a_interval + qzero_z; changes code space but cancels in
                  dequant output; termed "code_only"

Mechanism metrics collected per layer (averaged over batches):
  sat_rate      — fraction of tokens hitting ±qmax clamp
  code_entropy  — Shannon entropy (bits) of per-batch code histogram
  residual_mse  — E[(DQ(Q(x))-x)^2], measuring quant noise on original input
  residual_bias — |mean(DQ(Q(x)) - x)| averaged over channels
  mean_err_rel  — |mean_batch(y) - mu_s| / |mu_s|, per channel mean, then mean over channels
  std_err_rel   — |std_batch(y) - sig_s| / sig_s, same
  transition_rt — fraction of per-token codes that differ from no_adapt (approx, first 20 batches)

Stop condition: if decoupled does NOT beat output_only by >0.2 pp on ImageNet-R or Sketch,
  print a clear warning and skip remaining bitwidths.

Usage:
  python run_qac_code_reassignment_diag.py \\
      --data_r   {DATA_R} \\
      --data_s   {DATA_SKETCH} \\
      --data_a   {DATA_A} \\
      --bw 6 \\
      --ptq_checkpoint runs/.../ptq4vit_w6a6_calibrated.pt \\
      --stats_file    runs/.../qac_stats_w6a6_all_layers_seed0.pt \\
      --scope block8_9_10_11_valid \\
      --batch_size 64 --gpu 0
"""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import sys, os, time, argparse, math, json, torch
import numpy as np
import torchvision.transforms as T
import torchvision.datasets as D
from torch.utils.data import DataLoader

sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

EPS    = 1e-6
ALPHA  = 1.0
CLIP   = 3.0

_NORM_MEAN = [0.5, 0.5, 0.5]
_NORM_STD  = [0.5, 0.5, 0.5]

# ─── model loading ──────────────────────────────────────────────────────────

def _patch_timm_compat():
    """Add timm.layers.attention shim for checkpoints saved with older timm versions."""
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
    net_wrap.wrap_modules_in_net(net, quant_cfg)
    _patch_timm_compat()
    Attention.attention_forward  = attention_forward
    WindowAttention.window_attention_forward = window_attention_forward

    if ptq_checkpoint and os.path.isfile(ptq_checkpoint):
        net = torch.load(ptq_checkpoint, map_location="cpu", weights_only=False)
        print(f"  Checkpoint loaded: {ptq_checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ptq_checkpoint}")

    for m in net.modules():
        if hasattr(m, "mode"):
            m.mode = "quant_forward"

    if device is not None:
        net = net.to(device)
        _tensor_attrs = ("w_interval", "a_interval", "qzero_z", "qzero_alpha",
                         "A_interval", "B_interval")
        for m in net.modules():
            for attr in _tensor_attrs:
                v = m.__dict__.get(attr)
                if torch.is_tensor(v):
                    m.__dict__[attr] = v.to(device)

    return net


# ─── scope modules ──────────────────────────────────────────────────────────

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


# ─── data loading ───────────────────────────────────────────────────────────

def _build_mask(data_root):
    """Build a 1000-class boolean mask from the WNIDs in data_root's class folders."""
    from dataset.ImageNetMask import all_wnids
    folder_wnids = {d for d in os.listdir(data_root) if d.startswith("n")}
    mask = [wnid in folder_wnids for wnid in all_wnids]
    return mask


def get_dataset_loader(dataset_name, data_root, batch_size, max_samples=None, seed=0):
    """Returns (loader, mask_or_None).

    mask is a boolean list length 1000 for 200-class subsets; None for full 1000-class.

    Indices are always shuffled once with the given seed so that batches contain
    a class-mixed sample (matching the episodic-TTA assumption). All evaluation
    modes iterate the same fixed order, which is required for the per-batch
    transition-rate comparison.
    """
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(_NORM_MEAN, _NORM_STD),
    ])
    ds = D.ImageFolder(data_root, transform=tfm)

    # Shuffle once with fixed seed so all modes see identical batch compositions.
    rng = np.random.RandomState(seed)
    n = len(ds)
    indices = rng.permutation(n)
    if max_samples is not None and max_samples < n:
        indices = indices[:max_samples]
    ds = torch.utils.data.Subset(ds, indices.tolist())

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False)

    if dataset_name in ("imagenet_r", "imagenet_a"):
        mask = _build_mask(data_root)
    else:
        mask = None

    return loader, mask


# ─── the local branch helpers ────────────────────────────────────────────────────────────

def compute_qac_params(x, src, device):
    """Returns (a_ch, b_ch) shape (C,) each, and sig_ratio (C,)."""
    sig_s = src["std"].to(device)
    mu_s  = src["mean"].to(device)
    x_rs  = x.reshape(-1, x.shape[-1])
    mu_t  = x_rs.mean(dim=0)
    sig_t = x_rs.std(dim=0)
    sig_ratio = (sig_s / (sig_t + EPS)).clamp(1.0 / CLIP, CLIP)
    a_ch = (1.0 - ALPHA) + ALPHA * sig_ratio
    b_ch = ALPHA * (mu_s - sig_ratio * mu_t)
    return a_ch, b_ch


def _group_params(a_ch, b_ch, module):
    """Reshape (C,) per-channel params to (n_a, crb_acts) per group."""
    n_a = module.a_interval.shape[0]
    C   = a_ch.shape[0]
    gs  = max(C // n_a, 1)
    nf  = n_a * gs
    return a_ch[:nf].reshape(n_a, gs), b_ch[:nf].reshape(n_a, gs)


# ─── mechanism metrics accumulator ─────────────────────────────────────────

class MetricsAccum:
    def __init__(self):
        self.n   = 0
        self.sat = 0.0
        self.ent = 0.0
        self.rmse = 0.0
        self.rbias = 0.0
        self.merr = 0.0
        self.serr = 0.0
        self.tr   = 0.0   # transition rate vs no_adapt
        self.tr_n = 0

    def update(self, x, y, interval, qmax, src_mean, src_std):
        """
        x: (B, ..., C) original input tensor, cpu float32
        y: (B, ..., C) dequantized output tensor, cpu float32
        interval: (n_a, 1) or scalar, on cpu
        qmax: int
        src_mean, src_std: (C,) on cpu
        """
        x = x.float(); y = y.float()
        interval = interval.float().cpu() if torch.is_tensor(interval) else float(interval)

        # Reconstruct codes from y and interval (approximate)
        n_a = interval.shape[0] if torch.is_tensor(interval) else 1
        C = x.shape[-1]
        gs = max(C // n_a, 1)
        nf = n_a * gs

        x_flat = x.reshape(-1, x.shape[-1])[:, :nf]
        y_flat = y.reshape(-1, y.shape[-1])[:, :nf]
        iv_flat = interval.reshape(n_a, 1).repeat(1, gs).reshape(nf)  # (nf,)

        # codes = round(y / iv)  (dequantized output / interval ≈ code)
        codes = (y_flat / (iv_flat + EPS)).round().clamp(-qmax, qmax - 1).long()

        # saturation rate
        sat = ((codes.abs() >= qmax - 1).float().mean()).item()

        # code entropy (global histogram over all tokens & channels)
        code_vals = codes.reshape(-1)
        hist = torch.bincount(code_vals + qmax, minlength=2 * qmax).float()
        hist = hist / (hist.sum() + EPS)
        log_h = torch.log2(hist + EPS)
        ent = -(hist * log_h).sum().item()
        ent = max(ent, 0.0)

        # residual MSE and bias (relative to original input x)
        residual = y_flat - x_flat
        rmse  = (residual ** 2).mean().item()
        rbias = residual.mean(dim=0).abs().mean().item()

        # mean/std alignment vs source stats
        y_all = y.reshape(-1, y.shape[-1])
        mu_t   = y_all.mean(dim=0)[:src_mean.shape[0]]
        sig_t  = y_all.std(dim=0)[:src_std.shape[0]]
        merr = ((mu_t - src_mean).abs() / (src_mean.abs() + EPS)).mean().item()
        serr = ((sig_t - src_std).abs() / (src_std + EPS)).mean().item()

        self.n += 1
        self.sat  += sat
        self.ent  += ent
        self.rmse += rmse
        self.rbias += rbias
        self.merr += merr
        self.serr += serr

    def update_transition(self, y_ref, y_cur, interval, qmax):
        """Compare codes of y_ref (no_adapt) vs y_cur (current mode)."""
        y_ref = y_ref.float().cpu(); y_cur = y_cur.float().cpu()
        interval = interval.float().cpu() if torch.is_tensor(interval) else float(interval)
        n_a = interval.shape[0] if torch.is_tensor(interval) else 1
        C = y_ref.shape[-1]
        gs = max(C // n_a, 1); nf = n_a * gs
        iv_flat = interval.reshape(n_a, 1).repeat(1, gs).reshape(nf)
        y_r = y_ref.reshape(-1, y_ref.shape[-1])[:, :nf]
        y_c = y_cur.reshape(-1, y_cur.shape[-1])[:, :nf]
        q_r = (y_r / (iv_flat + EPS)).round().clamp(-qmax, qmax - 1)
        q_c = (y_c / (iv_flat + EPS)).round().clamp(-qmax, qmax - 1)
        tr = (q_r != q_c).float().mean().item()
        self.tr   += tr
        self.tr_n += 1

    def result(self):
        n = max(self.n, 1)
        return {
            "sat_rate":       self.sat  / n,
            "code_entropy_b": self.ent  / n,
            "residual_mse":   self.rmse / n,
            "residual_bias":  self.rbias / n,
            "mean_err_rel":   self.merr / n,
            "std_err_rel":    self.serr / n,
            "transition_rt":  self.tr / max(self.tr_n, 1),
        }


# ─── ECE metric ─────────────────────────────────────────────────────────────

class ECEMeter:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.confs = []
        self.accs  = []

    def update(self, logits, targets):
        probs = torch.softmax(logits.cpu().float(), dim=1)
        conf, pred = probs.max(dim=1)
        correct = pred == targets.cpu()
        self.confs.append(conf)
        self.accs.append(correct)

    def ece(self):
        if not self.confs:
            return float("nan")
        confs   = torch.cat(self.confs)
        correct = torch.cat(self.accs).float()
        bins = torch.linspace(0.0, 1.0, self.n_bins + 1)
        ece  = 0.0
        n    = len(confs)
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (confs > lo) & (confs <= hi)
            if m.sum() == 0:
                continue
            acc_b  = correct[m].mean().item()
            conf_b = confs[m].mean().item()
            ece   += (m.sum().item() / n) * abs(acc_b - conf_b)
        return ece * 100.0


# ─── evaluation with local-branch hooks ──────────────────────────────────────────────

def evaluate_mode(net, loader, scope_modules, src_stats, mode, mask, device,
                  ref_captures=None, capture_max_batches=20):
    """
    Run one full evaluation pass.

    ref_captures: dict {name: list_of_y_tensors} from no_adapt; used for transition rate.
    capture_max_batches: only capture y for transition rate on first N batches.

    Returns (top1_acc, ece, metrics_per_layer, captures_or_None)
    """
    net.eval()
    mask_t = torch.tensor(mask, dtype=torch.bool, device=device) if mask else None
    qmax = 2 ** (scope_modules[0][1].a_bit - 1) if scope_modules else 128

    correct = 0; total = 0
    ece_m   = ECEMeter()
    accum   = {name: MetricsAccum() for name, _ in scope_modules}
    captures = {name: [] for name, _ in scope_modules}

    for batch_idx, (x_batch, y_batch) in enumerate(loader):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        do_capture = batch_idx < capture_max_batches

        # —— build local-branch hooks ——
        hooks = []
        layer_inputs   = {}  # name -> x before quant_input (cpu, for metrics)
        layer_outputs  = {}  # name -> y after quant_input (cpu, for metrics & transition)

        for name, m in scope_modules:
            src = src_stats.get(name, {})

            # monkeypatch quant_input to capture x→y
            original_qi = m.quant_input

            _cap_x = [None]; _cap_y = [None]
            def make_qi_patch(orig, cx, cy):
                def patched(inp):
                    y = orig(inp)
                    cx[0] = inp.detach().cpu()
                    cy[0] = y.detach().cpu()
                    return y
                return patched

            patch = make_qi_patch(original_qi, _cap_x, _cap_y)
            m.quant_input = patch

            layer_inputs[name]  = _cap_x
            layer_outputs[name] = (_cap_y, _cap_x)

            if mode == "no_adapt":
                pass  # no hooks, only capture via monkeypatched quant_input

            elif mode == "output_only":
                def make_pre(n_, m_):
                    src_ = src_stats.get(n_, {})
                    def pre(mod, inputs):
                        if not inputs or not torch.is_tensor(inputs[0]):
                            return None
                        x = inputs[0]
                        a_ch, b_ch = compute_qac_params(x.detach(), src_, device)
                        ag, bg = _group_params(a_ch, b_ch, mod)
                        mod._quar_post_gamma = ag.to(mod.a_interval.device, mod.a_interval.dtype)
                        mod._quar_post_beta  = bg.to(mod.a_interval.device, mod.a_interval.dtype)
                        return None
                    return pre
                def make_post(n_, m_):
                    def post(mod, inputs, output):
                        mod.__dict__.pop("_quar_post_gamma", None)
                        mod.__dict__.pop("_quar_post_beta",  None)
                    return post
                hooks.append(m.register_forward_pre_hook(make_pre(name, m)))
                hooks.append(m.register_forward_hook(make_post(name, m)))

            elif mode == "decoupled":
                def make_pre(n_, m_):
                    src_ = src_stats.get(n_, {})
                    def pre(mod, inputs):
                        if not inputs or not torch.is_tensor(inputs[0]):
                            return None
                        x = inputs[0]
                        a_ch, b_ch = compute_qac_params(x.detach(), src_, device)
                        ag, bg = _group_params(a_ch, b_ch, mod)
                        mod.qaffine_gamma = ag.to(mod.a_interval.device, mod.a_interval.dtype)
                        mod.qaffine_beta  = bg.to(mod.a_interval.device, mod.a_interval.dtype)
                        return None
                    return pre
                def make_post(n_, m_):
                    def post(mod, inputs, output):
                        mod.__dict__.pop("qaffine_gamma", None)
                        mod.__dict__.pop("qaffine_beta",  None)
                    return post
                hooks.append(m.register_forward_pre_hook(make_pre(name, m)))
                hooks.append(m.register_forward_hook(make_post(name, m)))

            elif mode == "coupled":
                def make_pre(n_, m_):
                    src_ = src_stats.get(n_, {})
                    iv_bk = [None]; zz_bk = [None]
                    def pre(mod, inputs):
                        if not inputs or not torch.is_tensor(inputs[0]):
                            return None
                        x = inputs[0]
                        a_ch, b_ch = compute_qac_params(x.detach(), src_, device)
                        ag, bg = _group_params(a_ch, b_ch, mod)
                        ag_g = ag.mean(dim=1, keepdim=True)
                        bg_g = bg.mean(dim=1, keepdim=True)
                        iv = mod.a_interval
                        iv_bk[0] = iv.clone()
                        iv.div_(ag_g.to(iv.device, iv.dtype))
                        z = (bg_g / iv_bk[0]).to(iv.device, iv.dtype)
                        zz_bk[0] = mod.__dict__.get("qzero_z", None)
                        mod.qzero_z = z
                        return None
                    def post(mod, inputs, output):
                        if iv_bk[0] is not None:
                            mod.a_interval.copy_(iv_bk[0])
                        if zz_bk[0] is None:
                            mod.__dict__.pop("qzero_z", None)
                        else:
                            mod.qzero_z = zz_bk[0]
                    return pre, post
                pre_h, post_h = make_pre(name, m)
                hooks.append(m.register_forward_pre_hook(pre_h))
                hooks.append(m.register_forward_hook(post_h))

        # —— forward pass ——
        with torch.no_grad():
            logits = net(x_batch)

        # —— clean up ——
        for h in hooks:
            h.remove()
        for name, m in scope_modules:
            del m.quant_input  # restore class method

        # —— collect metrics ——
        if mask_t is not None:
            logits_masked = logits[:, mask_t]
            pred = logits_masked.argmax(1)
        else:
            pred = logits.argmax(1)

        correct += (pred == y_batch).sum().item()
        total   += y_batch.size(0)
        ece_m.update(logits.cpu(), y_batch.cpu())

        for name, m in scope_modules:
            src = src_stats.get(name, {})
            if not src:
                continue
            cy = layer_outputs[name][0][0]  # captured y (cpu)
            cx = layer_inputs[name][0]       # captured x (cpu)
            if cy is None or cx is None:
                continue

            iv = m.a_interval.detach().cpu()
            sm = src["mean"].cpu().float()
            ss = src["std"].cpu().float()
            accum[name].update(cx, cy, iv, qmax, sm, ss)

            if do_capture:
                captures[name].append(cy)

            if (ref_captures is not None and name in ref_captures
                    and batch_idx < len(ref_captures[name])):
                y_ref = ref_captures[name][batch_idx]
                accum[name].update_transition(y_ref, cy, iv, qmax)

    acc = 100.0 * correct / max(total, 1)
    ece = ece_m.ece()
    metrics = {name: accum[name].result() for name, _ in scope_modules}

    # return captures only for no_adapt
    out_captures = captures if mode == "no_adapt" else None
    return acc, ece, metrics, out_captures


# ─── summary helpers ─────────────────────────────────────────────────────────

def agg_metrics(metrics_per_layer):
    """Average metrics over all layers."""
    keys = list(next(iter(metrics_per_layer.values())).keys())
    agg  = {}
    for k in keys:
        vals = [metrics_per_layer[n][k] for n in metrics_per_layer]
        agg[k] = float(np.mean(vals))
    return agg


def print_results(dataset_name, results):
    """Pretty-print accuracy and aggregated metrics for all modes."""
    print(f"\n{'='*68}")
    print(f"  {dataset_name}")
    print(f"{'='*68}")
    hdr = f"{'Mode':<18} {'Acc%':>7} {'ECE%':>7} | {'sat':>6} {'H(b)':>6} {'rMSE':>8} {'rbias':>8} {'merr':>7} {'serr':>7} {'tr_rt':>7}"
    print(hdr)
    print("-" * len(hdr))
    for mode, (acc, ece, agg) in results.items():
        sat  = agg.get("sat_rate", float("nan"))
        ent  = agg.get("code_entropy_b", float("nan"))
        rmse = agg.get("residual_mse", float("nan"))
        rb   = agg.get("residual_bias", float("nan"))
        me   = agg.get("mean_err_rel", float("nan"))
        se   = agg.get("std_err_rel", float("nan"))
        tr   = agg.get("transition_rt", float("nan"))
        print(f"{mode:<18} {acc:>7.2f} {ece:>7.2f} | {sat:>6.4f} {ent:>6.2f} {rmse:>8.5f} {rb:>8.5f} {me:>7.4f} {se:>7.4f} {tr:>7.4f}")


def check_stop(results_r, results_s, dataset_name_r, dataset_name_s):
    """Return True if stop condition triggered."""
    acc_oo_r = results_r.get("output_only", (0,))[0]
    acc_dc_r = results_r.get("decoupled",   (0,))[0]
    acc_oo_s = results_s.get("output_only", (0,))[0]
    acc_dc_s = results_s.get("decoupled",   (0,))[0]
    gap_r = acc_dc_r - acc_oo_r
    gap_s = acc_dc_s - acc_oo_s
    print(f"\nStop condition check:")
    print(f"  {dataset_name_r}: decoupled - output_only = {gap_r:+.2f} pp")
    print(f"  {dataset_name_s}: decoupled - output_only = {gap_s:+.2f} pp")
    if gap_r <= 0.2 and gap_s <= 0.2:
        print("  *** STOP: code reassignment does NOT improve over output_only by >0.2pp")
        print("  *** Code reassignment should NOT be pursued as a separate method component.")
        return True
    return False


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_r",   default=os.environ.get("DATA_R", ""))
    p.add_argument("--data_s",   default=os.environ.get("DATA_SKETCH", ""))
    p.add_argument("--data_a",   default=os.environ.get("DATA_A", ""))
    p.add_argument("--bw",       type=int, default=6)
    p.add_argument("--scope",    default="block8_9_10_11_valid")
    p.add_argument("--ptq_checkpoint",
        default="checkpoints/ptq4vit_w6a6_vitb.pt")
    p.add_argument("--stats_file",
        default="runs/qac_scope_sanity_v2_w6a6_allstats_seed0_4corr_20260522_164459/"
                "qac_stats_w6a6_all_layers_seed0.pt")
    p.add_argument("--batch_size",       type=int, default=64)
    p.add_argument("--gpu",              type=int, default=0)
    p.add_argument("--max_samples",      type=int, default=None,
                   help="Cap images per dataset (for quick testing)")
    p.add_argument("--skip_sketch",      action="store_true")
    p.add_argument("--skip_imagenet_a",  action="store_true")
    p.add_argument("--capture_batches",  type=int, default=20,
                   help="Batches used for transition-rate comparison")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    out_dir = f"runs/qac_code_reassign_w{args.bw}a{args.bw}_{args.scope}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"the local branch Code-Reassignment Diagnostic — W{args.bw}A{args.bw}")
    log(f"scope={args.scope}, batch_size={args.batch_size}, gpu={args.gpu}")
    log(f"ptq_checkpoint={args.ptq_checkpoint}")
    log(f"stats_file={args.stats_file}")
    log(f"output_dir={out_dir}\n")

    # —— load model ——
    log("Loading PTQ4ViT model...")
    net = load_ptq4vit(args.bw, args.ptq_checkpoint, device=device)
    net.eval()
    scope_modules = get_scope_modules(net, args.scope)
    log(f"Scope modules ({len(scope_modules)}): {[n for n,_ in scope_modules]}")

    # —— load source stats ——
    log(f"\nLoading source stats from {args.stats_file}...")
    raw_stats = torch.load(args.stats_file, map_location="cpu", weights_only=False)
    src_stats = {}
    for name, _ in scope_modules:
        if name in raw_stats:
            src_stats[name] = raw_stats[name]
        elif "mean" in raw_stats and name in raw_stats.get("mean", {}):
            src_stats[name] = {"mean": raw_stats["mean"][name],
                               "std":  raw_stats["std"][name]}
    log(f"Stats matched: {len(src_stats)}/{len(scope_modules)}")

    # —— datasets ——
    datasets_to_run = []
    if os.path.isdir(args.data_r):
        datasets_to_run.append(("imagenet_r",     args.data_r,  "ImageNet-R"))
    if not args.skip_sketch and os.path.isdir(args.data_s):
        datasets_to_run.append(("imagenet_sketch", args.data_s, "ImageNet-Sketch"))
    if not args.skip_imagenet_a and os.path.isdir(args.data_a):
        datasets_to_run.append(("imagenet_a",     args.data_a,  "ImageNet-A"))

    modes = ["no_adapt", "output_only", "decoupled", "coupled"]

    all_results = {}  # {ds_name: {mode: (acc, ece, agg_metrics)}}

    for ds_key, ds_path, ds_label in datasets_to_run:
        log(f"\n{'─'*60}")
        log(f"Dataset: {ds_label}  ({ds_path})")

        loader, mask = get_dataset_loader(
            ds_key, ds_path, args.batch_size,
            max_samples=args.max_samples, seed=0)
        log(f"  {len(loader.dataset)} images, mask={'yes' if mask else 'no'}")

        ds_results = {}
        ref_caps   = None  # will be set after no_adapt pass

        for mode in modes:
            log(f"\n  Mode: {mode}")
            t0 = time.time()
            acc, ece, per_layer_m, out_caps = evaluate_mode(
                net, loader, scope_modules, src_stats, mode, mask, device,
                ref_captures=ref_caps,
                capture_max_batches=args.capture_batches)
            elapsed = time.time() - t0
            agg = agg_metrics(per_layer_m)
            ds_results[mode] = (acc, ece, agg)
            log(f"    Acc={acc:.2f}%  ECE={ece:.2f}%  ({elapsed:.0f}s)")
            log(f"    sat={agg['sat_rate']:.4f}  H={agg['code_entropy_b']:.2f}b  "
                f"rMSE={agg['residual_mse']:.5f}  rbias={agg['residual_bias']:.5f}  "
                f"merr={agg['mean_err_rel']:.4f}  serr={agg['std_err_rel']:.4f}  "
                f"tr={agg['transition_rt']:.4f}")

            if mode == "no_adapt":
                ref_caps = out_caps  # store for transition rate

        all_results[ds_label] = ds_results
        print_results(ds_label, ds_results)

    # —— check stop condition ——
    if "ImageNet-R" in all_results and "ImageNet-Sketch" in all_results:
        triggered = check_stop(
            all_results["ImageNet-R"],
            all_results["ImageNet-Sketch"],
            "ImageNet-R", "ImageNet-Sketch")
        log(f"\nStop condition triggered: {triggered}")
    elif "ImageNet-R" in all_results:
        r = all_results["ImageNet-R"]
        gap = r.get("decoupled", (0,))[0] - r.get("output_only", (0,))[0]
        log(f"\nImageNet-R gap (decoupled - output_only): {gap:+.2f} pp")
        if gap <= 0.2:
            log("  *** STOP: decoupled does NOT beat output_only by >0.2pp on ImageNet-R")

    # —— summary table ——
    log("\n" + "=" * 68)
    log("SUMMARY — Accuracy (%)")
    log(f"{'Dataset':<20} " + "  ".join(f"{m:<14}" for m in modes))
    log("-" * 80)
    for ds_label, ds_results in all_results.items():
        row = f"{ds_label:<20}"
        for mode in modes:
            if mode in ds_results:
                acc = ds_results[mode][0]
                row += f"  {acc:>7.2f}       "
            else:
                row += f"  {'N/A':>7}       "
        log(row)

    # —— save JSON ——
    out_json = {}
    for ds_label, ds_results in all_results.items():
        out_json[ds_label] = {}
        for mode, (acc, ece, agg) in ds_results.items():
            out_json[ds_label][mode] = {"acc": acc, "ece": ece, **agg}

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    log(f"\nResults saved to {json_path}")
    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
