"""Measure the decision-side half of the amplification mechanism.

Proposition~\\ref{prop:acc} argues that a prediction survives quantization while the propagated
logit perturbation stays inside the unquantized margin, that shift shrinks that margin while the
perturbation is unchanged, and that the perturbation itself grows as the step coarsens. The
resolution half of that story is measured by tab:levelsgroup and the within-group rescale test;
this tool measures the decision half, the margin and the propagated perturbation.

This runs the full-precision and the quantized network over the same inputs and reports, per
condition:

  margin        the unquantized top-1 minus top-2 logit gap, which the shift should shrink
  |dz|          the L2 norm of the quantized-minus-unquantized logit vector, which should be
                roughly invariant to the shift and should grow as the step coarsens
  gap_shift     the shift the perturbation applies to the top-2 gap itself, which is what actually
                decides a flip
  flip_pred     the fraction with margin < gap_shift, the mechanism's predicted flip rate
  flip_actual   the fraction whose argmax differs between the two networks

The claim is supported if the margin shrinks from clean to corrupted, |dz| does not, flip_actual
grows, and flip_pred tracks flip_actual.

Usage
  DATA=... DATA_C=... python tools/analysis/measure_margin_interaction.py --bw w6a6
  ... --corruptions original,gaussian_noise      # a quick check
  ... --bw w3a3 --max_batches 8                  # a cheaper pass
Writes outputs/margin_interaction_<bw>.json and prints a summary.
"""
# This file sits two levels below the repository root, so the root is named explicitly rather
# than derived from the working directory, matching the other tools under tools/analysis/.
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import os
import statistics as st
import types

import torch

DATA = os.environ.get("DATA", "")
DATA_C = os.environ.get("DATA_C", "")
IDS = f"{_REPO_ROOT}/robustbench/data/imagenet_test_image_ids.txt"
CMAP = f"{_REPO_ROOT}/robustbench/data/imagenet_class_to_id_map.json"
OUT = f"{_REPO_ROOT}/outputs"

CORR15 = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
          "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
          "elastic_transform", "pixelate", "jpeg_compression"]

# The protocol quantizer per bit-width, matching run_cumulative_maintable.sh.
PTQ_W6A6 = f"{_REPO_ROOT}/checkpoints/ptq4vit_w6a6_vitb.pt"
PTQ_W8A8 = f"{_REPO_ROOT}/checkpoints/ptq4vit_w8a8_vitb.pt"
ADALOG = {"w4a4": f"{_REPO_ROOT}/checkpoints/w4a4_checkpoint_opt.pt",
          "w3a3": f"{_REPO_ROOT}/checkpoints/w3a3_checkpoint_opt.pt"}


def build_fp(arch):
    import timm
    net = timm.create_model(arch, pretrained=True)
    net.eval().requires_grad_(False)
    return net.cuda()


def build_quantized(bw, arch):
    """Load the calibrated quantized net the way the launchers do."""
    if bw in ("w6a6", "w8a8"):
        from tools.stats.build_setsize_stats import load_w6a6_fullnet
        ckpt = PTQ_W6A6 if bw == "w6a6" else PTQ_W8A8
        net = load_w6a6_fullnet(ckpt)
    else:
        from tools.stats.build_qac_stats import load_adalog_net
        w = int(bw[1]); a = int(bw[3])
        net = load_adalog_net(ADALOG[bw], w_bit=w, a_bit=a, arch=arch)
    net = net.cuda()
    # PTQ4ViT keeps calibration tensors as attributes, which have to follow the net to the GPU.
    for m in net.modules():
        for attr in ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z"):
            t = getattr(m, attr, None)
            if isinstance(t, torch.Tensor) and not t.is_cuda:
                setattr(m, attr, t.cuda())
    net.eval().requires_grad_(False)
    return net


def loader_for(corruption, batch_size, workers):
    from dataset.selectedRotateImageFolder import prepare_test_data
    args = types.SimpleNamespace(
        data=DATA, data_corruption=DATA_C, corruption=corruption, level=5,
        batch_size=batch_size, workers=workers, imagenet_5k_eval=True, continual=False,
        if_shuffle=True, seed=0, imagenet_5k_indices_file=IDS, imagenet_class_map_file=CMAP)
    ds, loader = prepare_test_data(args)
    if hasattr(ds, "switch_mode"):
        ds.switch_mode(True, False)
    return loader


def condition(fp_net, q_net, corruption, batch_size, workers, max_batches):
    """One (corruption or 'original') pass over the subset with both networks."""
    margins, dznorms, gapshifts, znorms, relperts = [], [], [], [], []
    n = flip_actual = flip_pred = 0
    correct_fp = correct_q = 0
    # 'flips more labels' means correct under full precision and wrong once quantized. The
    # symmetric disagreement also counts flips the other way, so both are reported.
    flip_harm = flip_help = 0
    loader = loader_for(corruption, batch_size, workers)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = batch[0].cuda(non_blocking=True)
            y = batch[1].cuda(non_blocking=True)
            z = fp_net(x).float()
            zq = q_net(x).float()
            # Some wrappers return the masked 200-class head on rendition splits; this script
            # only runs ImageNet-C and clean, where both heads are 1000-dim.
            if z.shape != zq.shape:
                raise RuntimeError("logit shapes differ: %s vs %s" % (tuple(z.shape), tuple(zq.shape)))
            top2 = z.topk(2, dim=1)
            i1, i2 = top2.indices[:, 0], top2.indices[:, 1]
            margin = top2.values[:, 0] - top2.values[:, 1]
            dz = zq - z
            # What decides a flip is how much the perturbation closes the top-2 gap, not its norm.
            gap_shift = dz.gather(1, i2.unsqueeze(1)).squeeze(1) - dz.gather(1, i1.unsqueeze(1)).squeeze(1)
            zn = z.norm(dim=1)
            dzn = dz.norm(dim=1)
            margins += margin.tolist()
            dznorms += dzn.tolist()
            znorms += zn.tolist()
            # The perturbation has to be read against the logit scale it perturbs: corruption
            # changes that scale too, so the absolute norm alone cannot separate the two.
            relperts += (dzn / zn.clamp_min(1e-6)).tolist()
            gapshifts += gap_shift.tolist()
            flip_pred += int((margin < gap_shift).sum())
            flip_actual += int((zq.argmax(1) != i1).sum())
            correct_fp += int((i1 == y).sum())
            qpred = zq.argmax(1)
            correct_q += int((qpred == y).sum())
            fp_ok = i1 == y
            q_ok = qpred == y
            flip_harm += int((fp_ok & ~q_ok).sum())
            flip_help += int((~fp_ok & q_ok).sum())
            n += x.shape[0]
    if not n:
        return None
    return {"n": n,
            "acc_fp": 100.0 * correct_fp / n, "acc_quant": 100.0 * correct_q / n,
            "margin_mean": st.mean(margins), "margin_median": st.median(margins),
            "dz_norm_mean": st.mean(dznorms),
            "z_norm_mean": st.mean(znorms),
            "rel_pert_mean": st.mean(relperts),
            "gap_shift_mean": st.mean(gapshifts),
            "flip_pred": flip_pred / n, "flip_actual": flip_actual / n,
            "flip_harmful": flip_harm / n, "flip_helpful": flip_help / n}



# --- activation-level quantization error at the 16 recalibrated interfaces -------------------
# prop:acc splits E[e^2] into a granular term Delta^2/12, fixed by the frozen step, and a clipping
# term that grows only once the activation leaves the calibrated range. The element count is the
# same on clean and corrupted data, so the prediction is that ||e_l|| is essentially constant
# unless the clipped fraction moves. Measuring it separates that claim from the logit-level
# perturbation, which propagation gain can inflate on its own.
def _quantize(xf, scale, qmax, neg_scale):
    """Reproduce the repo's own quantizer arithmetic (build_qac_stats._compute_code_hist)."""
    if scale <= 0:
        scale = 1e-8
    if neg_scale is not None:
        if neg_scale <= 0:
            neg_scale = 1e-8
        pos_raw = (xf / scale).round()
        neg_raw = (xf / neg_scale).round()
        pos = pos_raw.clamp(0, qmax - 1)
        neg = neg_raw.clamp(-qmax, 0)
        q = pos * scale + neg * neg_scale
        clipped = ((pos_raw > qmax - 1) | (neg_raw < -qmax)).float().mean()
        return q, clipped
    raw = (xf / scale).round()
    q = raw.clamp(-qmax, qmax - 1) * scale
    clipped = ((raw > qmax - 1) | (raw < -qmax)).float().mean()
    return q, clipped


def select_interfaces(net):
    """The 16 local interfaces the method recalibrates: blocks 8-11 x {qkv, proj, fc1, fc2}."""
    from tools.stats.build_qac_stats import _is_quant_act, _block_idx
    out = []
    for name, m in net.named_modules():
        if not _is_quant_act(m):
            continue
        if _block_idx(name) not in (8, 9, 10, 11):
            continue
        if not any(k in name for k in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")):
            continue
        out.append((name, m))
    return out


def activation_error(q_net, corruption, batch_size, workers, max_batches):
    """Accumulate ||e_l||, ||h_l|| and the clipped fraction per interface over one condition."""
    from tools.stats.build_qac_stats import _get_quant_scale_qmax, _get_neg_scale
    sel = select_interfaces(q_net)
    if not sel:
        raise RuntimeError("no quantized interfaces found in blocks 8-11")
    acc = {n: {"e2": 0.0, "h2": 0.0, "clip": 0.0, "b": 0} for n, _ in sel}

    def mk(name, mod):
        scale, qmax, nbits = _get_quant_scale_qmax(mod)
        neg = _get_neg_scale(mod)

        def hook(m, inp):
            if scale is None or not inp:
                return
            xf = inp[0].detach().float()
            q, clipped = _quantize(xf, scale, qmax, neg)
            a = acc[name]
            a["e2"] += float((q - xf).pow(2).sum())
            a["h2"] += float(xf.pow(2).sum())
            a["clip"] += float(clipped)
            a["b"] += 1
        return hook

    handles = [m.register_forward_pre_hook(mk(n, m)) for n, m in sel]
    loader = loader_for(corruption, batch_size, workers)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            q_net(batch[0].cuda(non_blocking=True))
    for h in handles:
        h.remove()

    per = {}
    for n, a in acc.items():
        if not a["b"]:
            continue
        per[n] = {"e_norm": a["e2"] ** 0.5, "h_norm": a["h2"] ** 0.5,
                  "rel": (a["e2"] / max(a["h2"], 1e-12)) ** 0.5,
                  "clipped_frac": a["clip"] / a["b"]}
    return {"per_interface": per,
            "e_norm_mean": st.mean(v["e_norm"] for v in per.values()),
            "rel_mean": st.mean(v["rel"] for v in per.values()),
            "clipped_frac_mean": st.mean(v["clipped_frac"] for v in per.values())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bw", default="w6a6", choices=["w8a8", "w6a6", "w4a4", "w3a3"])
    p.add_argument("--arch", default="vit_base_patch16_224")
    p.add_argument("--corruptions", default="",
                   help="comma-separated; default is 'original' plus all 15 ImageNet-C corruptions")
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--max_batches", default=0, type=int, help="0 runs the whole subset")
    p.add_argument("--activation_error", action="store_true",
                   help="measure ||e_l|| at the 16 interfaces instead of the logit comparison")
    args = p.parse_args()

    if not DATA or not DATA_C:
        raise SystemExit("set DATA to the ImageNet root and DATA_C to the ImageNet-C root")

    corrs = [c for c in args.corruptions.split(",") if c] or (["original"] + CORR15)
    print("bit-width %s, %d conditions" % (args.bw, len(corrs)), flush=True)

    q_net = build_quantized(args.bw, args.arch)
    fp_net = None if args.activation_error else build_fp(args.arch)

    res = {}
    for c in corrs:
        if args.activation_error:
            r = activation_error(q_net, c, args.batch_size, args.workers, args.max_batches)
            print("  [%-18s] |e_l| %10.1f  |e|/|h| %.4f  clipped %.4f%%"
                  % (c, r["e_norm_mean"], r["rel_mean"], 100 * r["clipped_frac_mean"]), flush=True)
            res[c] = r
            continue
        r = condition(fp_net, q_net, c, args.batch_size, args.workers, args.max_batches)
        if r is None:
            print("  [%s] no data" % c, flush=True)
            continue
        res[c] = r
        print("  [%-18s] accFP %5.2f accQ %5.2f | margin %6.3f  |dz| %6.3f  gap_shift %+6.3f  flip_pred %5.1f%%  flip_actual %5.1f%%"
              % (c, r["acc_fp"], r["acc_quant"], r["margin_mean"], r["dz_norm_mean"],
                 r["gap_shift_mean"], 100 * r["flip_pred"], 100 * r["flip_actual"]), flush=True)
        print("  %20s harmful %5.2f%%  helpful %5.2f%%  net %+5.2f pp"
              % ("", 100 * r["flip_harmful"], 100 * r["flip_helpful"],
                 100 * (r["flip_helpful"] - r["flip_harmful"])), flush=True)

    os.makedirs(OUT, exist_ok=True)
    tag = "activation_error" if args.activation_error else "margin_interaction"
    path = os.path.join(OUT, "%s_%s.json" % (tag, args.bw))
    payload = {"bitwidth": args.bw, "arch": args.arch, "batch_size": args.batch_size,
               "max_batches": args.max_batches, "conditions": res}
    corrupted = [v for k, v in res.items() if k != "original"]
    if args.activation_error and "original" in res and corrupted:
        cl = res["original"]
        payload["summary"] = {
            "e_norm_clean": cl["e_norm_mean"],
            "e_norm_corrupted": st.mean(v["e_norm_mean"] for v in corrupted),
            "rel_clean": cl["rel_mean"],
            "rel_corrupted": st.mean(v["rel_mean"] for v in corrupted),
            "clipped_clean": cl["clipped_frac_mean"],
            "clipped_corrupted": st.mean(v["clipped_frac_mean"] for v in corrupted),
        }
    elif "original" in res and corrupted:
        clean = res["original"]
        payload["summary"] = {
            "margin_clean": clean["margin_mean"],
            "margin_corrupted": st.mean(v["margin_mean"] for v in corrupted),
            "dz_norm_clean": clean["dz_norm_mean"],
            "dz_norm_corrupted": st.mean(v["dz_norm_mean"] for v in corrupted),
            "z_norm_clean": clean["z_norm_mean"],
            "z_norm_corrupted": st.mean(v["z_norm_mean"] for v in corrupted),
            "rel_pert_clean": clean["rel_pert_mean"],
            "rel_pert_corrupted": st.mean(v["rel_pert_mean"] for v in corrupted),
            "flip_actual_clean": clean["flip_actual"],
            "flip_actual_corrupted": st.mean(v["flip_actual"] for v in corrupted),
            "flip_pred_clean": clean["flip_pred"],
            "flip_pred_corrupted": st.mean(v["flip_pred"] for v in corrupted),
            "flip_harmful_clean": clean["flip_harmful"],
            "flip_harmful_corrupted": st.mean(v["flip_harmful"] for v in corrupted),
            "flip_helpful_clean": clean["flip_helpful"],
            "flip_helpful_corrupted": st.mean(v["flip_helpful"] for v in corrupted),
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote %s" % path, flush=True)

    s = payload.get("summary")
    if s and args.activation_error:
        print("\n=== activation-level quantization error, clean vs corrupted (%s) ===" % args.bw, flush=True)
        print("  |e_l|        %10.1f -> %10.1f  (%+.1f%%)" % (
            s["e_norm_clean"], s["e_norm_corrupted"],
            100 * (s["e_norm_corrupted"] / s["e_norm_clean"] - 1)), flush=True)
        print("  |e|/|h|      %10.4f -> %10.4f  (%+.1f%%)" % (
            s["rel_clean"], s["rel_corrupted"],
            100 * (s["rel_corrupted"] / s["rel_clean"] - 1)), flush=True)
        print("  clipped      %9.4f%% -> %9.4f%%" % (
            100 * s["clipped_clean"], 100 * s["clipped_corrupted"]), flush=True)
    elif s:
        print("\n=== clean vs corrupted (%s) ===" % args.bw, flush=True)
        print("  margin       %6.3f -> %6.3f  (%+.1f%%)" % (
            s["margin_clean"], s["margin_corrupted"],
            100 * (s["margin_corrupted"] / s["margin_clean"] - 1)), flush=True)
        print("  |dz|         %6.3f -> %6.3f  (%+.1f%%)" % (
            s["dz_norm_clean"], s["dz_norm_corrupted"],
            100 * (s["dz_norm_corrupted"] / s["dz_norm_clean"] - 1)), flush=True)
        print("  |z|          %6.3f -> %6.3f  (%+.1f%%)" % (
            s["z_norm_clean"], s["z_norm_corrupted"],
            100 * (s["z_norm_corrupted"] / s["z_norm_clean"] - 1)), flush=True)
        print("  |dz|/|z|     %6.4f -> %6.4f  (%+.1f%%)" % (
            s["rel_pert_clean"], s["rel_pert_corrupted"],
            100 * (s["rel_pert_corrupted"] / s["rel_pert_clean"] - 1)), flush=True)
        print("  flip actual  %5.1f%% -> %5.1f%%" % (
            100 * s["flip_actual_clean"], 100 * s["flip_actual_corrupted"]), flush=True)
        print("  flip pred    %5.1f%% -> %5.1f%%" % (
            100 * s["flip_pred_clean"], 100 * s["flip_pred_corrupted"]), flush=True)


if __name__ == "__main__":
    main()
