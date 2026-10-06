#!/usr/bin/env python3
"""Generate the QDQ folding feasibility report + tex from
outputs/qdq_folding_feasibility.json (produced by tools/analysis/qdq_folding_feasibility.py).

STANDALONE / read-only w.r.t. method+eval code. Writes only:
  outputs/qdq_folding_feasibility_report.md
  outputs/qdq_folding_feasibility.tex
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import json, os

OUT = f"{_REPO_ROOT}/outputs"
d = json.load(open(f"{OUT}/qdq_folding_feasibility.json"))
P1 = d["phase1_single_node"]; P2 = d["phase2_real_activation"]; P3 = d["phase3_full_graph"]


def case(name):
    return next((c for c in P1 if c["name"].startswith(name)), {})


def emit_tex():
    rows = []
    label = {"A_": "per-tensor, sym, scale-only", "B_": "per-tensor, sym, full affine",
             "C_": "per-tensor, asym ($z{\\neq}0$), full affine", "F_": "per-tensor, sym, $b/s\\in\\mathbb{Z}$",
             "D_": "per-channel, sym, scale-only", "E_": "per-channel, asym, full affine (Ours-like)"}
    for k in ["A_", "B_", "C_", "F_", "D_", "E_"]:
        c = case(k)
        rows.append(f"  {label[k]} & {c['code_agreement']:.4f} & {c['max_abs_diff']:.3g} & {c['conclusion']} \\\\")
    p2 = P2
    rows.append("\\midrule")
    rows.append(f"  real W8A8 (blocks.11.attn.qkv), per-ch.\\ affine & {p2['code_agreement']:.3f} & "
                f"{p2['max_abs_diff']:.3g} & {p2['conclusion']} \\\\")
    return """% QDQ folding feasibility for Ours' pre-quant affine recalibration.
\\begin{table}[t]
\\centering
\\caption{ONNX QDQ folding feasibility for Ours' pre-quant affine recalibration, using the real ONNX \\texttt{QuantizeLinear}/\\texttt{DequantizeLinear} ops in ONNX Runtime. Folding $x\\!\\to\\!\\mathrm{Mul}(a)\\!\\to\\!\\mathrm{Add}(b)\\!\\to\\!Q(s,z)\\!\\to\\!DQ(s,z)$ into $x\\!\\to\\!Q(s/a,\\,z{+}b/s)\\!\\to\\!DQ(s,z)$. Code agreement = fraction of matching integer codes vs.\\ the explicit graph; max diff in dequantized output. Folding feasibility test, not an accuracy benchmark.}
\\label{tab:qdq_folding}
\\begin{tabular}{l|rrl}
\\toprule
Case & Code agree. & Max out.\\ diff & Folding \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\\\[0.4em]
{\\footnotesize Scale ($a$) folds exactly; the shift ($b$) folds exactly only when $b/s$ is an integer zero-point (and, under round-half-to-even, leaves rare one-code tie differences for odd $b/s$). Ours' per-channel, continuous $(a,b)$ over a per-tensor activation quantizer is \\emph{not} exactly foldable; the explicit \\texttt{Mul}/\\texttt{Add} wrapper is the correct deployable representation.}
\\end{table}
"""


def emit_report():
    A, B, C, F = case("A_"), case("B_"), case("C_"), case("F_")
    D, E = case("D_"), case("E_")
    L = []
    L.append("# ONNX QDQ Folding Feasibility — Ours' Pre-quant Affine Recalibration\n")
    L.append("**Goal:** test whether Ours' explicit pre-quant affine wrapper (`Mul(a)`→`Add(b)` "
             "before the activation quantizer) can be **folded into `QuantizeLinear` parameters** in a "
             "standard ONNX QDQ graph. Folding feasibility test, not an accuracy benchmark. **ECE not "
             "reported.** Read-only w.r.t. the method/eval code (new standalone scripts only).\n")
    L.append("**Method:** real ONNX `QuantizeLinear`/`DequantizeLinear` ops executed in ONNX Runtime "
             "1.19.2 (CPU). Folding tested: "
             "`x→Mul(a)→Add(b)→Q(s,z)→DQ(s,z)` (explicit) ⇒ `x→Q(s_eff,z_eff)→DQ(s,z)` (folded), with "
             "**`s_eff = s/a`** and **`z_eff = z + b/s`** (integer zero-point), and the "
             "**DequantizeLinear keeping the ORIGINAL `(s,z)`** (not `s_eff,z_eff`).\n")

    L.append("## Phase 1 — single-node QDQ folding equivalence\n")
    L.append("| Case | a,b,s,z | b/s integer? | code agree | max out diff | conclusion |")
    L.append("|------|---------|--------------|-----------:|-------------:|------------|")
    def row(c, desc):
        return (f"| {desc} | — | {c['flags']['bs_all_integer']} | {c['code_agreement']:.4f} | "
                f"{c['max_abs_diff']:.3g} | **{c['conclusion']}** |")
    L.append(row(A, "per-tensor, symmetric, scale-only (b=0)"))
    L.append(row(B, "per-tensor, symmetric, full affine"))
    L.append(row(C, "per-tensor, asymmetric (z≠0), full affine"))
    L.append(row(F, "per-tensor, symmetric, b/s = integer (=5)"))
    L.append(row(D, "per-channel, symmetric, scale-only"))
    L.append(row(E, "per-channel, asymmetric, full affine (Ours-like)"))
    L.append("")
    L.append("**Readings.**")
    L.append(f"- **Scale folds exactly.** Scale-only cases are exact: per-tensor (A: code {A['code_agreement']:.3f}, "
             f"max diff {A['max_abs_diff']:.3g}) and per-channel (D: {D['code_agreement']:.3f}, {D['max_abs_diff']:.3g}). "
             f"`s_eff = s/a` is exact for `a>0`.")
    L.append(f"- **Continuous shift breaks exactness.** With a non-integer `b/s`, the shift cannot become an "
             f"integer zero-point: B/C drop to code agreement {B['code_agreement']:.3f} (max one-step diff "
             f"{B['max_abs_diff']:.3g}); the Ours-like per-channel case E is {E['code_agreement']:.3f}.")
    L.append(f"- **Even exact-integer `b/s` is only near-exact under round-half-to-even.** Case F (`b/s=5`, an "
             f"odd integer) gives code agreement {F['code_agreement']:.3f} (mean diff {F['mean_abs_diff']:.3g}): "
             f"ONNX `QuantizeLinear` uses round-half-to-even, which is **not shift-invariant under odd integer "
             f"offsets**, so rare half-integer ties flip by one code. Perfect equivalence needs `b/s` to be an "
             f"**even** integer (or round-half-away semantics).")
    L.append("- **Per-channel scale ⇒ per-axis activation quantizer.** D is exact only because `Q` is built "
             "per-axis (axis=1); folding a per-channel `a` into a per-tensor activation quantizer requires "
             "converting that quantizer to per-axis.\n")

    L.append("## Phase 2 — representative activation tensor (real W8A8)\n")
    L.append(f"Site: **{P2['site']}**; activation shape {P2['activation_shape']}; "
             f"activation quantizer: **{P2['activation_quantizer']}** (`a_interval` shape "
             f"{P2['a_interval_shape']}, a_qmax {P2['a_qmax']}, s≈{P2['s_pertensor']}). "
             f"Real per-channel Ours correction: a_ch∈{P2['a_ch_range']}, b_ch∈{P2['b_ch_range']}.\n")
    L.append(f"- `b_ch/s` is **non-integer for {P2['bs_fraction_nonint']*100:.0f}% of channels**.")
    L.append(f"- Folded vs explicit: **code agreement {P2['code_agreement']:.3f}**, max out diff "
             f"{P2['max_abs_diff']:.3g}, mean {P2['mean_abs_diff']:.3g} → **{P2['conclusion']}** (not exact).")
    L.append(f"- per-channel correction exactly representable: **{P2['per_channel_correction_exactly_representable']}**; "
             f"per-tensor activation quantizer blocks exact fold: **{P2['per_tensor_quantizer_blocks_exact_fold']}**.")
    L.append(f"- _{P2['note']}_\n")

    L.append("## Phase 3 — optional full-graph attempt\n")
    L.append(f"**{'Attempted' if P3['attempted'] else 'Skipped'}.** {P3['reason']}\n")

    L.append("## Answers\n")
    L.append("**1. Can explicit pre-quant affine recalibration be exactly folded into `QuantizeLinear`?** "
             "**Partially.** The **scale** folds exactly (`s_eff=s/a`, verified exact in A and D). The "
             "**shift** can only be absorbed as an **integer zero-point** (`z_eff=z+b/s`); it is exact only "
             "when `b/s` is an integer (ideally even). For a general affine it is **not** exactly foldable.\n")
    L.append("**2. Under what assumptions?** `a>0`; `b/s ∈ ℤ` (and even, for exact round-half-to-even "
             "equivalence — otherwise rare one-code tie differences); `z_eff=z+b/s` within the quantized "
             "dtype range; the **DequantizeLinear keeps the original `(s,z)`** (using `s_eff,z_eff` in DQ is "
             "wrong); and per-channel `a` requires a per-axis activation quantizer (next answer).\n")
    L.append("**3. Does it require per-axis activation quantization?** **Yes, when the scale correction is "
             "per-channel** (as in Ours). Folding a per-channel `a` turns `s_eff=s/a` per-channel, so the "
             "activation `QuantizeLinear` must be per-axis. The W8A8 model's activation quantizer is "
             "**per-tensor** (`a_interval` shape [1,1]), so this would require changing the quantizer "
             "granularity — and runtimes generally prefer per-tensor activation quantization for int8 "
             "fusion.\n")
    L.append("**4. Does zero-point rounding/range break exact folding?** **Yes.** The shift only enters as an "
             "integer zero-point; a continuous `b` makes `z_eff=z+b/s` non-integer (Phase 2: 100% of "
             "channels) → broken. Even integer `b/s` is only near-exact under round-half-to-even (case F). "
             "Large `b` can also push `z_eff` out of the int8 range (an additional failure mode).\n")
    L.append("**5. Is folding feasible for our actual Ours local branch?** **No — exact folding is invalid for "
             "Ours.** Ours applies a **per-channel, continuous** `(a,b)` at a **per-tensor** activation "
             "quantizer; Phase 2 on a real W8A8 site gives only "
             f"{P2['code_agreement']:.0%} code agreement (not exact), with 100% non-integer `b/s` and a "
             "per-tensor quantizer that blocks the fold. **Folding is valid only in the scalar/per-tensor "
             "scale-only sub-cases, not in our channel-wise Ours case.**\n")
    L.append("**6. If exact folding fails, is an explicit `Mul`/`Add` wrapper still the correct deployable "
             "representation?** **Yes.** The explicit pre-quant affine (`Mul`/`Add`) is exactly representable "
             "and ONNX-Runtime-executable (verified in the prior ONNX feasibility task, exact match), needs "
             "no per-axis activation quantization, no integer-zero-point constraint, and avoids the "
             "mismatched-grid `Q`/`DQ` issue. **The explicit affine wrapper is the supported deployable "
             "representation; folding into `QuantizeLinear` is not.**\n")

    L.append("## Limitations / caveats\n")
    L.append("- Even the **exact** scale-only fold produces a **mismatched `Q`/`DQ` pair** (Q uses `s_eff`, "
             "DQ uses `s`), which is **not** a canonical shared-scale QDQ pair that runtimes fuse into int8 "
             "kernels — so it is numerically exact but not a clean deployable QDQ.\n")
    L.append("- ONNX Runtime CPU was used for correctness/representability only — **no** latency, memory, or "
             "hardware claim; **no full deployment claim**.\n")
    L.append("- Phase 2 activations were captured from a random input through the W8A8 model (representative "
             "intermediate tensor); folding feasibility (integer `b/s`, per-channel vs per-tensor) is "
             "distribution-independent.\n")
    L.append("Sources: `outputs/qdq_folding_feasibility.json`; tool `tools/analysis/qdq_folding_feasibility.py`. "
             "Related: `outputs/onnx_runtime_deployment_feasibility_report.md`, "
             "`outputs/deployment_compatibility_audit.md`.\n")
    return "\n".join(L)


def main():
    open(f"{OUT}/qdq_folding_feasibility.tex", "w").write(emit_tex())
    open(f"{OUT}/qdq_folding_feasibility_report.md", "w").write(emit_report())
    print("Wrote outputs/qdq_folding_feasibility_report.md and .tex")


if __name__ == "__main__":
    main()
