#!/usr/bin/env python3
"""Part 1 — Decoupled folded-the local branch diagnostic.

Tests directly on PTQSLBatchingQuantLinear.quant_input() with synthetic activations
drawn from real source stats. No Hessian calibration needed (~seconds to run).

Compares:
  A. the explicit local branch       — pre-apply (a*x + b), then quant with original interval
  B. scale_offset_coupled  — modify interval and qzero_z, then quant; dequant≈x
  C. scale_offset_decoupled — set qaffine_gamma/beta, then quant; dequant=explicit

Reports per the local branch layer:
  - MAE and max-diff of dq vs explicit
  - output mean/std vs explicit
  - coupled MAE for comparison

Expected:
  coupled:   MAE(vs explicit) >> 0, output≈x (correction cancelled by dequant)
  decoupled: MAE(vs explicit) ≈ 0,  output≈the explicit local branch (output-equivalent)

Pass criterion: decoupled mean MAE < 1e-4 on all tested layers.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import sys, os, torch, numpy as np

sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

STATS_W8A8 = ("runs/qac_bw_scope_sanity_w8a8_w4a4_4corr_20260523_002929/"
              "qac_stats_w8a8_all_layers_seed0.pt")
STATS_W6A6 = ("runs/qac_scope_sanity_v2_w6a6_allstats_seed0_4corr_20260522_164459/"
              "qac_stats_w6a6_all_layers_seed0.pt")

ALPHA = 1.0
EPS   = 1e-6
CLIP  = 3.0
BATCH = 4
TOKENS = 197   # 196 patches + cls


# ── build minimal PTQSLBatchingQuantLinear without Hessian calib ───────────────

def make_quant_module(C, a_bit, n_a, interval_val, device):
    """Create a minimal batching quant linear with known a_interval."""
    from quant_library.quant_layers.linear import PTQSLBatchingQuantLinear
    m = PTQSLBatchingQuantLinear(
        in_features=C, out_features=C,
        mode="quant", w_bit=a_bit, a_bit=a_bit, n_a=n_a,
    ).to(device)
    m.a_interval = interval_val.to(device)
    return m


# ── the local branch parameter computation ──────────────────────────────────────────────────

def compute_ab_grouped(x, src, n_a, C, device, dtype):
    mu_s  = src["mean"].to(device); sig_s = src["std"].to(device)
    x_rs  = x.reshape(-1, C)
    mu_t  = x_rs.mean(0); sig_t = x_rs.std(0)
    sig_r = (sig_s / (sig_t + EPS)).clamp(1/CLIP, CLIP)
    a_ch  = (1 - ALPHA) + ALPHA * sig_r
    b_ch  = ALPHA * (mu_s - sig_r * mu_t)
    gs    = max(C // n_a, 1); nf = n_a * gs
    a_g   = a_ch[:nf].reshape(n_a, gs).mean(1, keepdim=True)
    b_g   = b_ch[:nf].reshape(n_a, gs).mean(1, keepdim=True)
    return a_g.to(device, dtype), b_g.to(device, dtype)


# ── three modes of quant_input ─────────────────────────────────────────────────

def run_explicit(m, x, a_g, b_g, C):
    """Apply explicit correction then quantize with original interval."""
    n_a = a_g.shape[0]; gs = max(C // n_a, 1); nf = n_a * gs
    a_exp = a_g.expand(-1, gs).reshape(nf)
    b_exp = b_g.expand(-1, gs).reshape(nf)
    x_corr = x.clone()
    x_corr[..., :nf] = x[..., :nf] * a_exp + b_exp
    with torch.no_grad():
        dq = m.quant_input(x_corr)
    return dq


def run_coupled(m, x, a_g, b_g):
    """Coupled: interval /= a_g, qzero_z = b_g/interval_orig."""
    iv_orig = m.a_interval.clone()
    saved_z = getattr(m, "qzero_z", None)
    with torch.no_grad():
        m.a_interval.div_(a_g)
        m.qzero_z = b_g / iv_orig
        dq = m.quant_input(x)
        m.a_interval.copy_(iv_orig)
        if saved_z is None: m.__dict__.pop("qzero_z", None)
        else: m.qzero_z = saved_z
    return dq


def run_decoupled(m, x, a_g, b_g):
    """Grouped decoupled: qaffine_gamma/beta shape (n_a, 1) — group-averaged correction."""
    n_a = a_g.shape[0]
    with torch.no_grad():
        m.qaffine_gamma = a_g.reshape(n_a, 1)
        m.qaffine_beta  = b_g.reshape(n_a, 1)
        dq = m.quant_input(x)
        del m.__dict__["qaffine_gamma"]
        del m.__dict__["qaffine_beta"]
    return dq


def run_decoupled_perchan(m, x, a_g, b_g, C):
    """Per-channel decoupled: qaffine_gamma/beta shape (n_a, gs) — matches explicit granularity."""
    n_a = a_g.shape[0]; gs = max(C // n_a, 1); nf = n_a * gs
    a_exp = a_g.expand(-1, gs)   # (n_a, gs)
    b_exp = b_g.expand(-1, gs)
    with torch.no_grad():
        m.qaffine_gamma = a_exp
        m.qaffine_beta  = b_exp
        dq = m.quant_input(x)
        del m.__dict__["qaffine_gamma"]
        del m.__dict__["qaffine_beta"]
    return dq


# ── per-layer test ─────────────────────────────────────────────────────────────

def compute_ab_perchan(x, src, n_a, C, device, dtype):
    """Per-channel (a, b) — shape (n_a, gs), matches _quar_correct granularity."""
    mu_s  = src["mean"].to(device); sig_s = src["std"].to(device)
    x_rs  = x.reshape(-1, C)
    mu_t  = x_rs.mean(0); sig_t = x_rs.std(0)
    sig_r = (sig_s / (sig_t + EPS)).clamp(1/CLIP, CLIP)
    a_ch  = (1 - ALPHA) + ALPHA * sig_r
    b_ch  = ALPHA * (mu_s - sig_r * mu_t)
    gs    = max(C // n_a, 1); nf = n_a * gs
    a_pc  = a_ch[:nf].reshape(n_a, gs)
    b_pc  = b_ch[:nf].reshape(n_a, gs)
    return a_pc.to(device, dtype), b_pc.to(device, dtype)


def run_explicit_perchan(m, x, a_pc, b_pc, C):
    """Apply per-channel correction then quantize — matches _quar_correct exactly."""
    n_a = a_pc.shape[0]; gs = a_pc.shape[1]; nf = n_a * gs
    a_exp = a_pc.reshape(nf)
    b_exp = b_pc.reshape(nf)
    x_corr = x.clone()
    x_corr[..., :nf] = x[..., :nf] * a_exp + b_exp
    with torch.no_grad():
        dq = m.quant_input(x_corr)
    return dq


def test_layer(layer_name, src, a_bit, n_a, device, seed=0):
    torch.manual_seed(seed)
    C = src["mean"].shape[0]
    mu_s  = src["mean"].to(device); sig_s = src["std"].to(device)

    # Synthetic activation drawn from domain-shifted distribution (0.7× std)
    x = (torch.randn(BATCH, TOKENS, C, device=device) * sig_s * 0.7 + mu_s).float()

    # Calibrate interval from x (same as PTQ4ViT calibration heuristic)
    gs  = max(C // n_a, 1); nf = n_a * gs
    x_rs  = x.reshape(-1, C)
    x_g   = x_rs[:, :nf].reshape(-1, n_a, gs)
    a_qmax = 2 ** (a_bit - 1)
    interval = (x_g.abs().amax(dim=(0, 2), keepdim=False) / (a_qmax - 0.5)
                ).unsqueeze(-1).float()   # (n_a, 1)

    m = make_quant_module(C, a_bit, n_a, interval, device)
    dtype = interval.dtype

    # Grouped params (n_a, 1)
    a_g, b_g = compute_ab_grouped(x.detach(), src, n_a, C, device, dtype)
    # Per-channel params (n_a, gs)
    a_pc, b_pc = compute_ab_perchan(x.detach(), src, n_a, C, device, dtype)

    dq_exp_grp = run_explicit(m, x, a_g, b_g, C)          # grouped explicit
    dq_exp_pc  = run_explicit_perchan(m, x, a_pc, b_pc, C) # per-channel explicit
    dq_coup    = run_coupled(m, x, a_g, b_g)
    dq_dec_grp = run_decoupled(m, x, a_g, b_g)             # grouped decoupled (old)
    dq_dec_pc  = run_decoupled_perchan(m, x, a_pc, b_pc, C) # per-channel decoupled (new)

    mae_coup    = (dq_coup    - dq_exp_pc).abs().mean().item()
    mae_dec_grp = (dq_dec_grp - dq_exp_grp).abs().mean().item()  # should be ~0
    mae_dec_pc  = (dq_dec_pc  - dq_exp_pc).abs().mean().item()   # should be ~0
    max_dec_pc  = (dq_dec_pc  - dq_exp_pc).abs().max().item()

    return {
        "mae_coupled":      mae_coup,
        "mae_dec_grouped":  mae_dec_grp,
        "mae_dec_perchan":  mae_dec_pc,
        "max_dec_perchan":  max_dec_pc,
        "mu_exp_pc":  dq_exp_pc.mean().item(),
        "mu_dec_pc":  dq_dec_pc.mean().item(),
        "pass_grouped":  mae_dec_grp < 1e-4,
        "pass_perchan":  mae_dec_pc  < 1e-4,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run_bw(bw, stats_file, scope_layers, n_a=64, device="cpu"):
    print(f"\n{'='*80}")
    print(f"W{bw}A{bw} — Per-channel decoupled vs grouped decoupled vs explicit")
    print(f"{'='*80}")

    raw = torch.load(stats_file, map_location="cpu")
    stats = {k: {"mean": raw["mean"][k], "std": raw["std"][k]} for k in raw["mean"]}

    available = [l for l in scope_layers if l in stats]
    if not available:
        print("  No matching layers in stats file.")
        return False

    print(f"\n  {'Layer':<38} {'MAE(coup)':>10} {'MAE(dec_g)':>11} {'MAE(dec_pc)':>12}  Pass_g  Pass_pc")
    print("  " + "-"*90)

    all_pass_grp = True; all_pass_pc = True
    all_mae_coup = []; all_mae_grp = []; all_mae_pc = []

    for layer in available:
        r = test_layer(layer, stats[layer], bw, n_a, device)
        fg = "✓" if r["pass_grouped"]  else "✗"
        fp = "✓" if r["pass_perchan"]  else "✗"
        if not r["pass_grouped"]: all_pass_grp = False
        if not r["pass_perchan"]: all_pass_pc  = False
        all_mae_coup.append(r["mae_coupled"])
        all_mae_grp.append(r["mae_dec_grouped"])
        all_mae_pc.append(r["mae_dec_perchan"])
        print(f"  {layer:<38} {r['mae_coupled']:>10.6f} {r['mae_dec_grouped']:>11.6f} "
              f"{r['mae_dec_perchan']:>12.6f}    {fg}       {fp}")

    print(f"\n  Summary W{bw}A{bw}:")
    print(f"    coupled           mean MAE vs per-chan exp : {np.mean(all_mae_coup):.6f}  (large >> 0 expected)")
    print(f"    grouped decoupled mean MAE vs grouped exp  : {np.mean(all_mae_grp):.6f}  (should be ≈ 0)")
    print(f"    perchan decoupled mean MAE vs perchan exp  : {np.mean(all_mae_pc):.6f}  (should be ≈ 0)")
    print(f"  Grouped result : {'PASS ✓' if all_pass_grp else 'FAIL ✗'}")
    print(f"  Perchan result : {'PASS ✓' if all_pass_pc  else 'FAIL ✗'}")
    return all_pass_grp and all_pass_pc


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Folded-the local branch Decoupled Diagnostic — Part 1v2 (per-channel qaffine unit test)")
    print("Device:", device)

    qba_layers = [f"blocks.{b}.{s}"
                  for b in (10, 11)
                  for s in ("attn.qkv", "attn.proj", "mlp.fc1")]

    p8 = run_bw(8, STATS_W8A8, qba_layers, n_a=64, device=device)
    p6 = run_bw(6, STATS_W6A6, qba_layers, n_a=64, device=device)

    print("\n" + "=" * 70)
    if p8 and p6:
        print("OVERALL: PASS — per-channel decoupled is output-equivalent to per-channel explicit.")
    else:
        print("OVERALL: FAIL — fix implementation before Part 2b.")
        if not p8: print("  → W8A8 per-channel decoupled FAILED")
        if not p6: print("  → W6A6 per-channel decoupled FAILED")
    print("=" * 70)
    return 0 if (p8 and p6) else 1


if __name__ == "__main__":
    sys.exit(main())
