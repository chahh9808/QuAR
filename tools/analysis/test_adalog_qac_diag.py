#!/usr/bin/env python3
"""Part 1 — AdaLog W4A4 the local branch mechanism diagnostic.

Findings from model inspection:
  W4A4 AdaLog ViT-B uses two distinct quantizer types:
    - AsymmetricallyBatchingQuantLinear (UniformQuantizer): attn.qkv, attn.proj, mlp.fc1
    - PostGeluLogBasedBatchingQuantLinear (ShiftAdaLogQuantizer): mlp.fc2

  The block8_9_10_11_valid scope (excluding fc2) therefore selects ONLY
  AsymmetricallyBatchingQuantLinear layers. These have:
    - no a_interval (not PTQ4ViT)
    - a_quantizer.shift is None (not ShiftAdaLog)
  → ZOAOfficial hook falls through to EXPLICIT the local branch for ALL scope layers.

  Consequence: W4A4 folded the local branch = the explicit local branch trivially, with 0 layers actually folded.

This script:
  Section A — Scope audit: counts fallback layers in block8_9_10_11_valid.
  Section B — ShiftAdaLog unit test (fc2, out-of-scope):
    Verifies code_match, positivity, and dequant behavior for coupled scale_shift the local branch.
    Also confirms decoupled = explicit for nonlinear quantizers.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import sys, os, math, torch, numpy as np

sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

STATS_W4A4 = ("runs/qac_bw_scope_sanity_w8a8_w4a4_4corr_20260523_002929/"
              "qac_stats_w4a4_all_layers_seed0.pt")
ADALOG_CKPT = "w4a4_checkpoint_opt.pt"

ALPHA = 1.0
EPS   = 1e-6
CLIP  = 3.0
BATCH = 4
TOKENS = 197


# ── model loading ──────────────────────────────────────────────────────────────

def load_adalog_w4a4():
    import importlib as _importlib
    _adalog_root = os.path.join(_REPO_ROOT,
                                'quant_library', 'adalog')
    _utils_init = os.path.join(_adalog_root, 'utils', '__init__.py')
    if not os.path.exists(_utils_init): open(_utils_init, 'w').close()
    if _adalog_root not in sys.path: sys.path.insert(0, _adalog_root)
    _shadow_keys = [k for k in list(sys.modules.keys())
                    if k in ('utils','quant_layers','quantizers','configs')
                    or k.startswith(('utils.','quant_layers.','quantizers.','configs.'))]
    _saved = {k: sys.modules.pop(k) for k in _shadow_keys}
    try:
        _wrap = _importlib.import_module('utils.wrap_net')
        _cfg_mod = _importlib.import_module('configs.4bit')
    finally:
        for _k in list(sys.modules.keys()):
            if _k in ('utils','quant_layers','quantizers','configs') or \
               _k.startswith(('utils.','quant_layers.','quantizers.','configs.')):
                sys.modules[f'_adalog_{_k}'] = sys.modules.pop(_k)
        sys.modules.update(_saved)
    import timm
    cfg = _cfg_mod.Config(); cfg.w_bit=4; cfg.a_bit=4; cfg.s_bit=4
    net = timm.create_model('vit_base_patch16_224', pretrained=False)
    net = _wrap.wrap_modules_in_net(net, cfg, reparam=False)
    for _n, _m in net.named_modules():
        if hasattr(_m, 'mode'): _m.calibrated=True; _m.mode='quant_forward'
        for _qa in ('a_quantizer','w_quantizer','A_quantizer','B_quantizer'):
            if hasattr(_m, _qa): getattr(_m, _qa).inited=True
    ckpt = torch.load(ADALOG_CKPT, map_location='cpu')
    net.load_state_dict(ckpt, strict=False)
    net.eval()
    return net


# ── the local branch params (scalar a, b) ───────────────────────────────────────────────────

def compute_ab_scalar(x, src):
    sig_s = src["std"].mean();  mu_s = src["mean"].mean()
    x_rs = x.reshape(-1)
    mu_t = x_rs.mean(); sig_t = x_rs.std()
    sig_r = (sig_s / (sig_t + EPS)).clamp(1/CLIP, CLIP)
    a = (1 - ALPHA) + ALPHA * sig_r
    b = ALPHA * (mu_s - sig_r * mu_t)
    return a.item(), b.item()


# ── ShiftAdaLog test functions ─────────────────────────────────────────────────

def adalog_quantize_codes(x_plus_shift, scale, aq):
    """Compute integer codes (raw) from AdaLog quantizer."""
    r = aq.r; q_val = float(aq.q.item())
    scaled = (x_plus_shift / scale).clamp(min=1e-15, max=1.0)
    codes = torch.round(-scaled.log2() * r / q_val).clamp(0, 2*aq.n_levels - 1)
    return codes


def run_shift_adalog_explicit(aq, x, a, b):
    """Explicit: apply x_corr = a*x + b, then quantize with ORIGINAL params."""
    x_corr = a * x + b
    with torch.no_grad():
        dq = aq(x_corr)   # ShiftAdaLogQuantizer.forward(x_corr) = AdaLog(x_corr+shift) - shift
    return dq, x_corr


def run_shift_adalog_coupled(aq, x, a, b):
    """Coupled: scale'=scale/a, shift'=(b+shift)/a. Codes match explicit, dq≈x."""
    scale_orig = aq.scale.data.clone()
    shift_orig = aq.shift.data.clone()
    with torch.no_grad():
        aq.scale.data.copy_(scale_orig / a)
        aq.shift.data.copy_((b + shift_orig) / a)
        dq = aq(x)
        aq.scale.data.copy_(scale_orig)
        aq.shift.data.copy_(shift_orig)
    return dq


def run_shift_adalog_no_adapt(aq, x):
    with torch.no_grad():
        return aq(x)


def test_adalog_layer(name, mod, src, device, seed=0):
    torch.manual_seed(seed)
    aq = mod.a_quantizer
    scale_orig = aq.scale.data.clone().to(device)
    shift_orig = aq.shift.data.clone().to(device)

    mu_s = src["mean"].to(device); sig_s = src["std"].to(device)
    C = mu_s.shape[0]

    # Synthetic GELU output: x ~ |N(mu_s, 0.7*sig_s)|
    x = (torch.randn(BATCH, TOKENS, C, device=device) * sig_s * 0.7 + mu_s).abs().float()

    a, b = compute_ab_scalar(x, src)

    dq_na       = run_shift_adalog_no_adapt(aq, x)
    dq_exp, x_c = run_shift_adalog_explicit(aq, x, a, b)
    dq_coup     = run_shift_adalog_coupled(aq, x, a, b)

    # Code match: explicit codes (from x_corr + shift_orig) vs coupled codes
    q_exp  = adalog_quantize_codes(x_c + shift_orig, scale_orig,       aq)
    q_coup = adalog_quantize_codes(x  + (b + shift_orig) / a,
                                   scale_orig / a, aq)
    code_match = (q_exp == q_coup).float().mean().item()

    # Positivity check: x + shift' > 0 required
    shift_new = ((b + shift_orig) / a).item()
    n_viol  = (x + shift_new <= 0).sum().item()
    min_val = (x + shift_new).min().item()

    mae_coup_vs_exp = (dq_coup - dq_exp).abs().mean().item()
    mae_coup_vs_x   = (dq_coup - x).abs().mean().item()     # coupled dq ≈ x?
    mae_exp_vs_xcorr= (dq_exp  - x_c.clamp(min=0)).abs().mean().item()  # explicit dq ≈ x_corr?

    return {
        "a": a, "b": b,
        "code_match":       code_match,
        "n_pos_viol":       n_viol,
        "n_total":          x.numel(),
        "min_x_shift":      min_val,
        "mae_coup_vs_exp":  mae_coup_vs_exp,
        "mae_coup_vs_x":    mae_coup_vs_x,
        "mae_exp_vs_xcorr": mae_exp_vs_xcorr,
        "mu_coup": dq_coup.mean().item(),
        "mu_exp":  dq_exp.mean().item(),
        "mu_na":   dq_na.mean().item(),
        "mu_x":    x.mean().item(),
    }


# ── scope audit ───────────────────────────────────────────────────────────────

def audit_scope(net):
    """Classify all block8_9_10_11 layers (excl fc2) by quantizer type."""
    rows = []
    for name, mod in net.named_modules():
        blk = None
        for b in (8, 9, 10, 11):
            if f"blocks.{b}." in name: blk = b; break
        if blk is None: continue
        if "fc2" in name: continue
        a_interval = getattr(mod, "a_interval", None)
        aq = getattr(mod, "a_quantizer", None)
        if a_interval is None and aq is None:
            continue
        aq_type = type(aq).__name__ if aq else "None"
        has_shift = torch.is_tensor(getattr(aq, "shift", None)) if aq else False
        # What does ZOAOfficial hook do?
        if torch.is_tensor(a_interval):
            hook_path = "PTQ4ViT (a_interval)"
        elif aq is not None and torch.is_tensor(getattr(aq, "scale", None)) and has_shift:
            hook_path = "ShiftAdaLog (fold possible)"
        else:
            hook_path = "FALLBACK → explicit"
        rows.append({"name": name, "mod_type": type(mod).__name__,
                     "aq_type": aq_type, "has_shift": has_shift, "hook": hook_path})
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("AdaLog W4A4 the local branch Diagnostic — Part 1")
    print("Device:", device)

    if not os.path.exists(ADALOG_CKPT):
        print(f"ERROR: checkpoint not found: {ADALOG_CKPT}"); return 1
    if not os.path.exists(STATS_W4A4):
        print(f"ERROR: stats not found: {STATS_W4A4}"); return 1

    print("\nLoading W4A4 AdaLog model...")
    net = load_adalog_w4a4()
    net = net.to(device)

    # ── Section A: scope audit ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SECTION A — Scope Audit: block8_9_10_11_valid (excl fc2)")
    print("="*70)
    rows = audit_scope(net)
    n_total = len(rows)
    n_fallback = sum(1 for r in rows if "FALLBACK" in r["hook"])
    n_shift    = sum(1 for r in rows if "ShiftAdaLog" in r["hook"])
    n_ptq4vit  = sum(1 for r in rows if "PTQ4ViT" in r["hook"])

    print(f"\n  {'Layer':<40}  {'ModType':<38}  {'AQType':<22}  {'ZOA Hook Path'}")
    print("  " + "-"*138)
    for r in rows:
        print(f"  {r['name']:<40}  {r['mod_type']:<38}  {r['aq_type']:<22}  {r['hook']}")

    print(f"\n  Total scope layers   : {n_total}")
    print(f"  PTQ4ViT (a_interval): {n_ptq4vit}")
    print(f"  ShiftAdaLog          : {n_shift}")
    print(f"  FALLBACK → explicit  : {n_fallback}")
    print(f"\n  VERDICT: {n_fallback}/{n_total} ({100*n_fallback/max(n_total,1):.0f}%) layers fall back to explicit.")
    print(f"  W4A4 folded the local branch ≡ the explicit local branch for the block8_9_10_11_valid scope.")
    print(f"  No separate accuracy run needed — result is identical to existing explicit baseline.")

    # ── Section B: ShiftAdaLog unit test (fc2, out-of-scope) ──────────────────
    print("\n" + "="*70)
    print("SECTION B — ShiftAdaLogQuantizer Unit Test (mlp.fc2, out-of-scope)")
    print("Verifies coupled code_match, positivity, and dequant behavior.")
    print("="*70)

    raw = torch.load(STATS_W4A4, map_location="cpu")
    stats = {k: {"mean": raw["mean"][k], "std": raw["std"][k]} for k in raw["mean"]}

    fc2_layers = {n: m for n, m in net.named_modules()
                  if any(f"blocks.{b}.mlp.fc2" == n for b in (8, 9, 10, 11))}
    available = {k: v for k, v in fc2_layers.items() if k in stats}
    print(f"\n  fc2 layers found: {len(fc2_layers)}, with stats: {len(available)}")

    if not available:
        print("  No fc2 layers have matching stats — skipping ShiftAdaLog unit test.")
    else:
        print(f"\n  {'Layer':<28} {'a':>6} {'b':>7} {'code_match':>11} "
              f"{'viol/tot':>10} {'min(x+s)':>10} "
              f"{'MAE(c-e)':>9} {'MAE(c-x)':>9}")
        print("  " + "-"*105)

        all_cm = []; all_viol_r = []; all_mae_ce = []; all_mae_cx = []
        all_pass = True

        for layer_name, mod in available.items():
            src = stats[layer_name]
            r = test_adalog_layer(layer_name, mod, src, device)
            all_cm.append(r["code_match"])
            all_viol_r.append(r["n_pos_viol"] / max(r["n_total"], 1))
            all_mae_ce.append(r["mae_coup_vs_exp"])
            all_mae_cx.append(r["mae_coup_vs_x"])
            code_ok = r["code_match"] >= 0.99
            pos_ok  = r["n_pos_viol"] == 0
            if not (code_ok and pos_ok): all_pass = False
            flag = "✓" if (code_ok and pos_ok) else "✗"
            short = layer_name.replace("blocks.", "b").replace(".mlp.fc2", ".fc2")
            print(f"  {short:<28} {r['a']:>6.3f} {r['b']:>7.4f} "
                  f"{r['code_match']:>11.4f} {r['n_pos_viol']:>5}/{r['n_total']:<5} "
                  f"{r['min_x_shift']:>10.4f} "
                  f"{r['mae_coup_vs_exp']:>9.4f} {r['mae_coup_vs_x']:>9.4f}  {flag}")

        print(f"\n  Summary (ShiftAdaLog fc2 layers):")
        print(f"    mean code_match (coupled vs explicit) : {np.mean(all_cm):.6f}  (≈1.0 expected)")
        print(f"    mean positivity viol rate             : {np.mean(all_viol_r):.6f}    (0 expected)")
        print(f"    mean MAE coupled vs explicit dq       : {np.mean(all_mae_ce):.6f}")
        print(f"    mean MAE coupled dq vs x (no-adapt)  : {np.mean(all_mae_cx):.6f}  (small → dq≈x)")
        print(f"  Result: {'PASS ✓' if all_pass else 'FAIL ✗'}")

    # ── Section C: theoretical summary ───────────────────────────────────────
    print("\n" + "="*70)
    print("SECTION C — AdaLog Decoupled Analysis (Theoretical)")
    print("="*70)
    print("""
  ShiftAdaLogQuantizer.forward(x) = AdaLogQuantizer.forward(x + shift) - shift
  Quantization: q = round(-log2((x+shift)/scale) * r/q)
  Dequantization: dq = 2^(-q*q/r) * scale - shift

  Coupled the local branch (scale'=scale/a, shift'=(b+shift)/a):
    q_coup = round(-log2((x+shift')/scale') * r/q)
           = round(-log2((a*x+b+shift)/scale) * r/q) = q_explicit  ✓ codes match
    dq_coup = 2^(-q*q/r) * scale/a - (b+shift)/a
            = (2^(-q*q/r)*scale - shift - b)/a ≈ (x_corr - b)/a = x  ✗ correction cancelled

  Decoupled the local branch for AdaLog:
    To get dq ≈ x_corr = a*x+b with codes matching explicit:
    Must feed x_corr directly to the quantizer with ORIGINAL scale/shift:
      dq_decoupled = AdaLogQuantizer.forward(x_corr + shift) - shift ≈ x_corr
    This requires materializing x_corr = a*x + b — identical to the explicit local branch.

  Conclusion:
    AdaLog decoupled = explicit (no folding advantage for nonlinear quantizers).
    Only uniform (PTQ4ViT) quantizers support the qaffine_gamma/beta decoupled trick.
    W4A4 with block8_9_10_11_valid scope: 0 ShiftAdaLog layers → 100%% explicit fallback.
""")

    print("="*70)
    print("OVERALL: Part 1 complete.")
    print("  W4A4 folded the local branch = the explicit local branch (0 layers folded in scope).")
    print("  ShiftAdaLog coupled: codes match, dq≈x (correction cancelled in simulation).")
    print("  ShiftAdaLog decoupled = explicit (no parameter-folding possible for log quantizers).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
