#!/usr/bin/env python3
"""ONNX QDQ folding feasibility for Ours' pre-quant affine recalibration.

STANDALONE / READ-ONLY w.r.t. the method+eval codebase. Does NOT import or modify
main.py / zoa.py / generate_table*.py. Only loads the W8A8 checkpoint
(read-only forward) to grab one activation tensor for Phase 2.

Tests whether  x -> Mul(a) -> Add(b) -> QuantizeLinear(s,z) -> DequantizeLinear(s,z)
(explicit) can be folded into  x -> QuantizeLinear(s_eff,z_eff) -> DequantizeLinear(s,z)
(folded), using the REAL ONNX QuantizeLinear/DequantizeLinear ops in ONNX Runtime.

Math:  s_eff = s/a ;  z_eff = z + b/s  (must be an INTEGER zero-point).
Exact iff a>0, b/s in Z, and z_eff in dtype range. DequantizeLinear keeps the
ORIGINAL (s,z) (NOT s_eff/z_eff). Writes outputs/qdq_folding_feasibility.json.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import os, sys, json
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

OUT_JSON = f"{_REPO_ROOT}/outputs/qdq_folding_feasibility.json"
ORT_OPT = ort.SessionOptions(); ORT_OPT.log_severity_level = 3
QRANGE = {"int8": (-128, 127), "uint8": (0, 255)}
TP = {"int8": TensorProto.INT8, "uint8": TensorProto.UINT8}


_NP = {TensorProto.FLOAT: np.float32, TensorProto.INT8: np.int8, TensorProto.UINT8: np.uint8}


def _init(name, arr, tp=TensorProto.FLOAT):
    return numpy_helper.from_array(np.asarray(arr).astype(_NP[tp]), name)


def _qdq_nodes(prefix, inp, out, s_name, z_name, axis, per_channel):
    """QuantizeLinear -> DequantizeLinear sharing s,z. Returns (nodes, code_tensor_name)."""
    code = f"{prefix}_q"
    qattr = {"axis": axis} if per_channel else {}
    n1 = helper.make_node("QuantizeLinear", [inp, s_name, z_name], [code], **qattr)
    n2 = helper.make_node("DequantizeLinear", [code, s_name, z_name], [out], **(qattr))
    return [n1, n2], code


def build_explicit(x_shape, A, B, S, Z, axis, per_channel, qdtype):
    """x -> Mul(A) -> Add(B) -> Q(S,Z) -> DQ(S,Z). Outputs: out(float), code(int)."""
    g_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    inits = [_init("A", A), _init("B", B), _init("S", S), _init("Z", Z, TP[qdtype])]
    nodes = [helper.make_node("Mul", ["x", "A"], ["t1"]),
             helper.make_node("Add", ["t1", "B"], ["t2"])]
    qattr = {"axis": axis} if per_channel else {}
    nodes += [helper.make_node("QuantizeLinear", ["t2", "S", "Z"], ["code"], **qattr),
              helper.make_node("DequantizeLinear", ["code", "S", "Z"], ["out"], **qattr)]
    outs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("code", TP[qdtype], x_shape)]
    g = helper.make_graph(nodes, "explicit", [g_in], outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    return ort.InferenceSession(m.SerializeToString(), sess_options=ORT_OPT,
                                providers=["CPUExecutionProvider"])


def build_folded(x_shape, S_eff, Z_eff, S, Z, axis, per_channel, qdtype):
    """x -> Q(S_eff,Z_eff) -> DQ(S,Z[original!]). Outputs: out(float), code(int)."""
    g_in = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    inits = [_init("Seff", S_eff), _init("Zeff", Z_eff, TP[qdtype]),
             _init("S", S), _init("Z", Z, TP[qdtype])]
    qattr = {"axis": axis} if per_channel else {}
    nodes = [helper.make_node("QuantizeLinear", ["x", "Seff", "Zeff"], ["code"], **qattr),
             helper.make_node("DequantizeLinear", ["code", "S", "Z"], ["out"], **qattr)]
    outs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("code", TP[qdtype], x_shape)]
    g = helper.make_graph(nodes, "folded", [g_in], outs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    return ort.InferenceSession(m.SerializeToString(), sess_options=ORT_OPT,
                                providers=["CPUExecutionProvider"])


def fold_params(a, b, s, z, qdtype):
    """s_eff=s/a ; z_eff=z+round(b/s). Returns (s_eff, z_eff_int, flags)."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    s = np.asarray(s, np.float32); z = np.asarray(z)
    s_eff = (s / a).astype(np.float32)
    bs = b / s
    bs_int = np.round(bs)
    z_eff = (z + bs_int).astype(np.int64)
    qmin, qmax = QRANGE[qdtype]
    flags = {
        "a_all_positive": bool(np.all(a > 0)),
        "bs_max_frac_from_int": float(np.max(np.abs(bs - bs_int))),
        "bs_all_integer": bool(np.max(np.abs(bs - bs_int)) < 1e-6),
        "z_eff_in_range": bool(np.all((z_eff >= qmin) & (z_eff <= qmax))),
    }
    z_eff = np.clip(z_eff, qmin, qmax).astype({"int8": np.int8, "uint8": np.uint8}[qdtype])
    return s_eff, z_eff, flags


def run_case(name, x, a, b, s, z, axis, per_channel, qdtype, expect):
    qmin, qmax = QRANGE[qdtype]
    znp = {"int8": np.int8, "uint8": np.uint8}[qdtype]
    Z = np.asarray(z).astype(znp)
    s_eff, z_eff, flags = fold_params(a, b, s, z, qdtype)
    xs = list(x.shape)
    se = build_explicit(xs, np.asarray(a, np.float32), np.asarray(b, np.float32),
                        np.asarray(s, np.float32), Z, axis, per_channel, qdtype)
    sf = build_folded(xs, s_eff, z_eff, np.asarray(s, np.float32), Z, axis, per_channel, qdtype)
    out_e, code_e = se.run(None, {"x": x.astype(np.float32)})
    out_f, code_f = sf.run(None, {"x": x.astype(np.float32)})
    diff = np.abs(out_e.astype(np.float64) - out_f.astype(np.float64))
    code_agree = float(np.mean(code_e.astype(np.int64) == code_f.astype(np.int64)))
    max_d, mean_d = float(diff.max()), float(diff.mean())
    math_exact = flags["a_all_positive"] and flags["bs_all_integer"] and flags["z_eff_in_range"]
    if max_d < 1e-5 and code_agree > 0.999:
        concl = "exact"
    elif max_d < 0.5 and code_agree > 0.5:
        concl = "approximate"
    else:
        concl = "invalid"
    return {
        "name": name, "per_channel": per_channel, "dtype": qdtype, "symmetric": bool(np.all(Z == 0)),
        "expect": expect, "math_exact_conditions_met": math_exact, "flags": flags,
        "max_abs_diff": round(max_d, 8), "mean_abs_diff": round(mean_d, 8),
        "code_agreement": round(code_agree, 6), "conclusion": concl,
    }


def phase1():
    rng = np.random.default_rng(0)
    N, C = 256, 8
    cases = []
    # per-tensor x (N,1) so axis irrelevant; use (N,C) with broadcast scalars for per-tensor
    x = rng.standard_normal((N, C)).astype(np.float32) * 2.0
    s0 = 0.04
    # A: per-tensor scalar, symmetric, scale-only (b=0) -> expect exact
    cases.append(run_case("A_pertensor_sym_scaleonly", x, a=1.25, b=0.0, s=s0, z=0,
                          axis=1, per_channel=False, qdtype="int8", expect="exact"))
    # B: per-tensor, symmetric, full affine (continuous b) -> expect approximate (shift breaks)
    cases.append(run_case("B_pertensor_sym_fullaffine", x, a=1.25, b=0.37, s=s0, z=0,
                          axis=1, per_channel=False, qdtype="int8", expect="approximate"))
    # C: per-tensor, ASYMMETRIC (z=12), full affine -> expect approximate
    cases.append(run_case("C_pertensor_asym_fullaffine", x, a=1.25, b=0.37, s=s0, z=12,
                          axis=1, per_channel=False, qdtype="int8", expect="approximate"))
    # F: per-tensor, symmetric, full affine but b/s integer (b=0.04*5=0.20) -> expect exact
    cases.append(run_case("F_pertensor_sym_bs_integer", x, a=1.25, b=0.20, s=s0, z=0,
                          axis=1, per_channel=False, qdtype="int8", expect="exact"))
    # D: per-channel a (b=0), symmetric -> expect exact BUT requires per-axis activation quant
    ac = (0.8 + 0.5 * rng.random(C)).astype(np.float32)
    cases.append(run_case("D_perchannel_sym_scaleonly", x, a=ac, b=np.zeros(C, np.float32),
                          s=s0, z=0, axis=1, per_channel=True, qdtype="int8", expect="exact (needs per-axis act quant)"))
    # E: per-channel a,b continuous, asymmetric -> expect approximate/invalid (Ours-like)
    bc = (rng.standard_normal(C) * 0.3).astype(np.float32)
    cases.append(run_case("E_perchannel_asym_fullaffine", x, a=ac, b=bc, s=s0, z=10,
                          axis=1, per_channel=True, qdtype="int8", expect="approximate/invalid"))
    return cases


def phase2():
    """Real W8A8 activation at one Ours local site + real a,b,s,z."""
    import torch
    res = {"site": "blocks.11.attn.qkv (input activation)"}
    try:
        from quant_library.quant_utils.models import attention_forward, window_attention_forward
        from timm.models.vision_transformer import Attention
        from timm.models.swin_transformer import WindowAttention
        Attention.attention_forward = attention_forward
        WindowAttention.window_attention_forward = window_attention_forward
        net = torch.load(f"{_REPO_ROOT}/checkpoints/ptq4vit_w8a8_vitb.pt",
                         map_location="cpu", weights_only=False)
        for m in net.modules():
            if hasattr(m, "mode"): m.mode = "quant_forward"
        net.eval(); net.requires_grad_(False)
        # find the target module + hook its input
        target = dict(net.named_modules()).get("blocks.11.attn.qkv")
        if target is None:
            res["status"] = "module-not-found"; return res
        cap = {}
        h = target.register_forward_pre_hook(lambda mod, inp: cap.setdefault("x", inp[0].detach()))
        with torch.no_grad():
            net(torch.randn(1, 3, 224, 224))   # random input (representative activation capture)
        h.remove()
        x = cap["x"].reshape(-1, cap["x"].shape[-1]).float().numpy()   # [tokens, C]
        C = x.shape[-1]
        a_iv = getattr(target, "a_interval", None)
        a_qmax = int(getattr(target, "a_qmax", 128))
        res["activation_shape"] = list(x.shape)
        res["a_interval_shape"] = list(a_iv.shape) if torch.is_tensor(a_iv) else None
        res["a_qmax"] = a_qmax
        s = float(a_iv.detach().abs().mean()) if torch.is_tensor(a_iv) else 0.04
        res["activation_quantizer"] = "per-tensor (symmetric, z=0)" if (a_iv is None or a_iv.numel() == 1) \
            else f"per-group n_a={a_iv.shape[0]} (treated as per-tensor scalar s={s:.5g} for fold test)"
        # real per-channel correction a_ch,b_ch from quar_local_stats source vs captured target (alpha=1, mean_std)
        qac = torch.load(f"{_REPO_ROOT}/stats/qac_stats_w8a8_seed0.pt", map_location="cpu")
        key = "blocks.11.attn.qkv"
        mu_s = local_stats["mean"][key].float().numpy(); sig_s = local_stats["std"][key].float().numpy()
        mu_t = x.mean(0); sig_t = x.std(0)
        eps, clip, alpha = 1e-6, 3.0, 1.0
        sig_ratio = np.clip(sig_s / (sig_t + eps), 1.0 / clip, clip)
        a_ch = ((1 - alpha) + alpha * sig_ratio).astype(np.float32)
        b_ch = (alpha * (mu_s - sig_ratio * mu_t)).astype(np.float32)
        z = 0
        # int8 symmetric; per-channel fold
        out = run_case("phase2_w8a8_qkv", x.astype(np.float32), a=a_ch, b=b_ch, s=s, z=z,
                       axis=1, per_channel=True, qdtype="int8", expect="approximate/invalid (Ours)")
        bs = b_ch / s
        out["bs_fraction_nonint"] = round(float(np.mean(np.abs(bs - np.round(bs)) > 1e-3)), 4)
        out["per_channel_correction_exactly_representable"] = bool(out["conclusion"] == "exact")
        out["per_tensor_quantizer_blocks_exact_fold"] = bool(
            (a_iv is None or a_iv.numel() <= 1) and out["conclusion"] != "exact")
        out["a_ch_range"] = [round(float(a_ch.min()), 4), round(float(a_ch.max()), 4)]
        out["b_ch_range"] = [round(float(b_ch.min()), 4), round(float(b_ch.max()), 4)]
        out["s_pertensor"] = round(s, 6)
        res.update(out); res["status"] = "ok"
        res["note"] = "activation captured from a random input through the W8A8 model (representative intermediate tensor); folding feasibility is distribution-independent."
    except Exception as e:
        import traceback
        res["status"] = f"FAILED: {e}"; res["trace"] = traceback.format_exc()[-1500:]
    return res


def main():
    p1 = phase1()
    p2 = phase2()
    # gate: phase 3 only if both pass exactly for the Ours-like (per-channel full-affine) case
    p1_ours_like = next((c for c in p1 if c["name"].startswith("E_")), None)
    ours_exact = (p1_ours_like and p1_ours_like["conclusion"] == "exact") and (p2.get("conclusion") == "exact")
    p3 = {"attempted": bool(ours_exact),
          "reason": ("skipped: Phase 1/2 show exact folding is INVALID for the channel-wise + "
                     "continuous-shift Ours case; per the protocol, full-graph folding is not attempted."
                     if not ours_exact else "Phase 1/2 passed; full-graph attempt would follow.")}
    result = {
        "_about": "ONNX QDQ folding feasibility for Ours' pre-quant affine. Real ONNX QuantizeLinear/DequantizeLinear in ONNX Runtime. Folding feasibility test, not an accuracy benchmark.",
        "fold_math": "s_eff=s/a ; z_eff=z+round(b/s) (must be integer zero-point); DequantizeLinear keeps ORIGINAL (s,z).",
        "phase1_single_node": p1,
        "phase2_real_activation": p2,
        "phase3_full_graph": p3,
    }
    os.makedirs(f"{_REPO_ROOT}/outputs", exist_ok=True)
    json.dump(result, open(OUT_JSON, "w"), indent=2)
    print(json.dumps({"phase1": [(c["name"], c["conclusion"], c["max_abs_diff"], c["code_agreement"]) for c in p1],
                      "phase2": (p2.get("conclusion"), p2.get("max_abs_diff"), p2.get("code_agreement"),
                                 p2.get("per_tensor_quantizer_blocks_exact_fold")),
                      "phase3_attempted": p3["attempted"]}, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    sys.path.insert(0, _REPO_ROOT); os.chdir(_REPO_ROOT)
    main()
