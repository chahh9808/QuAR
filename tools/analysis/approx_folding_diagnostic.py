#!/usr/bin/env python3
"""Approximate QDQ-folding performance diagnostic for Ours' local pre-quant affine.

STANDALONE / READ-ONLY w.r.t. the method+eval code. Does NOT modify main.py /
zoa.py / cls_shift.py / generate_table*.py. It IMPORTS the real Ours
builders (build_no_adapt_qac_wrapper, build_cls_shift_wrapper) so the "explicit"
variant is the genuine Ours. For the folded variants it adds its own hooks that
recover the EXACT per-channel (a,b) the real the local branch produced (linear regression of
the local branch-corrected activation on the original activation) and re-express them as an
APPROXIMATE quantizer fold  s_eff=s/a, z_eff=round(b/s)  (DequantizeLinear keeps
the original s,z). Folding is therefore the ONLY difference vs explicit Ours
(identical running moments). Per-tensor activation quantizers (verified) + z=0.

Variants: noadapt | explicit | folded_axis (per-channel s_eff,z_eff)
        | folded_tensor (per-tensor compressed s_eff,z_eff).

Usage:
  python tools/analysis/approx_folding_diagnostic.py --variant explicit --gpu 0 [--max_corr N]
  python tools/analysis/approx_folding_diagnostic.py --mode codeagree --gpu 0
Writes partial results to $TMPDIR (the report generator aggregates).
 No exact-folding / deployment claims.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import os, sys, re, json, argparse, logging
import numpy as np
import torch

sys.path.insert(0, _REPO_ROOT); os.chdir(_REPO_ROOT)
TMP = os.environ.get("TMPDIR", "/tmp")
os.makedirs(TMP, exist_ok=True)
logging.basicConfig(level=logging.ERROR); LOG = logging.getLogger("afd")
CORR15 = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
          'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
          'elastic_transform', 'pixelate', 'jpeg_compression']
SITES = [f"blocks.{b}.{s}" for b in (10, 11) for s in ("attn.qkv", "attn.proj", "mlp.fc1")]
QMIN, QMAX = -128, 127


def make_args():
    txt = open(f"{_REPO_ROOT}/runs/table1_w8a8_20260528_045244/OURS.log", errors="ignore").read()
    ns = re.search(r"Namespace\((.*)\)", txt).group(1)
    a = eval("argparse.Namespace(" + ns + ")", {"argparse": argparse})
    a.batch_size = 64; a.workers = 8; a.seed = 0; a.if_shuffle = False
    a.imagenet_5k_eval = True; a.max_eval_batches = -1
    return a


def load_model():
    from quant_library.quant_utils.models import attention_forward, window_attention_forward
    from timm.models.vision_transformer import Attention
    from timm.models.swin_transformer import WindowAttention
    Attention.attention_forward = attention_forward
    WindowAttention.window_attention_forward = window_attention_forward
    net = torch.load(f"{_REPO_ROOT}/checkpoints/ptq4vit_w8a8_vitb.pt",
                     map_location="cpu", weights_only=False)
    for m in net.modules():
        if hasattr(m, "mode"): m.mode = "quant_forward"
    net.eval().requires_grad_(False); net.cuda()
    for m in net.modules():
        for attr in ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z"):
            t = getattr(m, attr, None)
            if torch.is_tensor(t) and not t.is_cuda: setattr(m, attr, t.cuda())
    return net


def site_scale(mod):
    iv = getattr(mod, "a_interval", None)
    return float(iv.detach().reshape(-1)[0]) if torch.is_tensor(iv) else None


def add_folded_hooks(net, mode, code_log=None):
    """mode in {'axis','tensor'}. Recover (a,b) from (x, x_out=a*x+b), refold.
    code_log: optional dict to record explicit vs folded integer codes per site."""
    mods = dict(net.named_modules())
    cap = {}
    handles = []
    for name in SITES:
        m = mods[name]; s = site_scale(m)

        def mk_cap(nm):
            def h(mod, inp):
                cap[nm] = inp[0].detach()
            return h

        def mk_refold(nm, sval):
            def h(mod, inp):
                x = cap[nm]; x_out = inp[0]
                xf = x.reshape(-1, x.shape[-1]).float()
                of = x_out.reshape(-1, x_out.shape[-1]).float()
                xm = xf.mean(0); om = of.mean(0)
                var = ((xf - xm) ** 2).sum(0).clamp_min(1e-8)
                a = (((xf - xm) * (of - om)).sum(0) / var).clamp_min(1e-6)   # per-channel slope
                b = om - a * xm
                if mode == "tensor":
                    a = a.mean().expand_as(a); b = b.mean().expand_as(b)
                s_eff = sval / a
                z_eff = torch.round(b / sval).clamp(QMIN, QMAX)
                q = torch.clamp(torch.round(x / s_eff) + z_eff, QMIN, QMAX)
                folded = q * sval                       # dequant with original (s, z=0)
                if code_log is not None:
                    # explicit codes = round(x_out / s); folded codes = round(folded / s)
                    ce = torch.round(x_out.detach() / sval).clamp(QMIN, QMAX)
                    cf = torch.round(folded.detach() / sval).clamp(QMIN, QMAX)
                    code_log.setdefault(nm, []).append(float((ce == cf).float().mean()))
                return (folded.to(x_out.dtype),) + tuple(inp[1:])
            return h

        handles.append(m.register_forward_pre_hook(mk_cap(name), prepend=True))
        handles.append(m.register_forward_pre_hook(mk_refold(name, s)))
    return handles


def build_variant(args, net, variant, code_log=None):
    """Returns (callable_model, reset_fn, hook_handles)."""
    from tta_library.zoa import build_no_adapt_qac_wrapper
    from tta_library.quar.global_cls import build_cls_shift_wrapper
    if variant == "noadapt":
        return net, (lambda: None), []
    localw = build_no_adapt_qac_wrapper(args, net, LOG)
    adapt = build_cls_shift_wrapper(args, net, LOG, local_wrapper=localw)
    handles = []
    if variant in ("folded_axis", "folded_tensor"):
        handles = add_folded_hooks(net, "axis" if variant == "folded_axis" else "tensor", code_log)

    def reset():
        for obj in (adapt, localw):
            if hasattr(obj, "reset"):
                try: obj.reset()
                except Exception: pass
    return adapt, reset, handles


def remove(handles):
    for h in handles: h.remove()


def run_full(variant, gpu, max_corr):
    torch.cuda.set_device(gpu)
    args = make_args()
    net = load_model()
    model, reset, handles = build_variant(args, net, variant)
    from dataset.selectedRotateImageFolder import prepare_test_data
    per_corr = {}
    argmax_all = {}
    corrs = CORR15[:max_corr] if max_corr > 0 else CORR15
    for c in corrs:
        args.corruption = c
        reset()
        ds, loader = prepare_test_data(args)
        preds, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                out = model(x.cuda())
                preds.append(out.argmax(1).cpu().numpy().astype(np.int16))
                ys.append(y.numpy().astype(np.int16))
        preds = np.concatenate(preds); ys = np.concatenate(ys)
        acc = float((preds == ys).mean() * 100)
        per_corr[c] = round(acc, 4)
        argmax_all[c] = preds
        print(f"  [{variant} gpu{gpu}] {c}: {acc:.3f}", flush=True)
    remove(handles)
    mean15 = round(float(np.mean([per_corr[c] for c in corrs])), 4) if len(corrs) == 15 else None
    out = {"variant": variant, "per_corruption": per_corr, "mean15": mean15,
           "n_corr": len(corrs)}
    json.dump(out, open(f"{TMP}/afd_{variant}.json", "w"), indent=2)
    np.savez(f"{TMP}/afd_argmax_{variant}.npz", ys=ys if len(corrs) else np.array([]),
             **{c: argmax_all[c] for c in corrs})
    print(json.dumps(out, indent=2))


def run_codeagree(gpu):
    """Run explicit + folded_axis + folded_tensor on a shared small subset:
    top-1 agreement, mean/max logit diff vs explicit, and per-site code agreement."""
    torch.cuda.set_device(gpu)
    args = make_args(); args.max_eval_batches = 8
    from dataset.selectedRotateImageFolder import prepare_test_data
    subset_corr = ["gaussian_noise", "motion_blur", "contrast"]
    res = {"subset": subset_corr, "batches_per_corr": 8}

    def collect(variant):
        net = load_model(); code_log = {} if variant != "explicit" else None
        model, reset, handles = build_variant(args, net, variant, code_log)
        logits = {}
        for c in subset_corr:
            args.corruption = c; reset()
            ds, loader = prepare_test_data(args)
            L = []
            with torch.no_grad():
                for i, (x, y) in enumerate(loader):
                    if i >= 8: break
                    L.append(model(x.cuda()).cpu().numpy())
            logits[c] = np.concatenate(L)
        remove(handles)
        del net; torch.cuda.empty_cache()
        return logits, code_log

    le, _ = collect("explicit")
    la, ca = collect("folded_axis")
    lt, ct = collect("folded_tensor")

    def agree(lx):
        top1 = float(np.mean([np.mean(le[c].argmax(1) == lx[c].argmax(1)) for c in subset_corr]) * 100)
        md = float(np.mean([np.abs(le[c] - lx[c]).mean() for c in subset_corr]))
        mx = float(np.max([np.abs(le[c] - lx[c]).max() for c in subset_corr]))
        return {"top1_agreement_pct": round(top1, 3), "mean_abs_logit_diff": round(md, 6),
                "max_abs_logit_diff": round(mx, 6)}

    def codeagr(cl):
        per = {k: round(float(np.mean(v)), 4) for k, v in cl.items()}
        return {"per_site": per, "avg": round(float(np.mean(list(per.values()))), 4)}

    res["folded_axis"] = {**agree(la), "code_agreement": codeagr(ca)}
    res["folded_tensor"] = {**agree(lt), "code_agreement": codeagr(ct)}
    json.dump(res, open(f"{TMP}/afd_agreement.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["noadapt", "explicit", "folded_axis", "folded_tensor"])
    ap.add_argument("--mode", choices=["codeagree"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_corr", type=int, default=-1)
    a = ap.parse_args()
    if a.mode == "codeagree":
        run_codeagree(a.gpu)
    else:
        run_full(a.variant, a.gpu, a.max_corr)
