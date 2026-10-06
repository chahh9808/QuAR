"""No-Adapt and Ours (simulated recalibration) code-distance at the post-LN B3 interfaces,
using the SAME pooled methodology as probe_zoa_codedist.py, on the SAME 3 corruptions, so
ZOA / No-Adapt / Ours are apples-to-apples."""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import sys, types, torch, statistics as st
sys.path.insert(0, _REPO_ROOT)
DATA = os.environ.get("DATA", "")
DATA_C = os.environ.get("DATA_C", "")
PTQ = f"{_REPO_ROOT}/checkpoints/ptq4vit_w6a6_vitb.pt"
IDS = f"{_REPO_ROOT}/robustbench/data/imagenet_test_image_ids.txt"
CMAP = f"{_REPO_ROOT}/robustbench/data/imagenet_class_to_id_map.json"
CORRUPTIONS = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur","motion_blur","zoom_blur","snow","frost","fog","brightness","contrast","elastic_transform","pixelate","jpeg_compression"]; MAXB = 25; EPS = 1e-6

from tools.stats.build_setsize_stats import load_w6a6_fullnet
from tools.stats.build_qac_stats import _is_quant_act, _get_quant_scale_qmax, _compute_code_hist, _block_idx
net = load_w6a6_fullnet(PTQ).cuda()
for _m in net.modules():
    for _a in ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z"):
        _t = getattr(_m, _a, None)
        if isinstance(_t, torch.Tensor) and not _t.is_cuda: setattr(_m, _a, _t.cuda())
net.eval(); net.requires_grad_(False)

stats = torch.load(f"{_REPO_ROOT}/stats/qac_stats_w6a6_codehist_all.pt", map_location="cpu")
src, MU, SIG = stats["code_hist"], stats["mean"], stats["std"]
sel = []
for name, m in net.named_modules():
    if not _is_quant_act(m): continue
    if _block_idx(name) not in (8, 9, 10, 11) or "fc2" in name: continue
    if not any(k in name for k in ("attn.qkv", "attn.proj", "mlp.fc1")): continue
    if name in src and name in MU: sel.append((name, m))
print(f"selected {len(sel)} post-LN B3 interfaces", flush=True)

acc_na, acc_ours = {}, {}
def mk(name, m):
    sc, qmax, nb = _get_quant_scale_qmax(m)
    mu_s = MU[name].cuda().float(); sig_s = SIG[name].cuda().float()
    def hook(mod, inp):
        if sc is None or not inp: return
        x = inp[0]
        if not torch.is_tensor(x): return
        with torch.no_grad():
            xf = x.detach().float().reshape(-1, x.shape[-1])      # [BN, C]
            # No-Adapt
            h0 = _compute_code_hist(x.detach().flatten(), sc, qmax, nb).cpu()
            acc_na[name] = h0 if name not in acc_na else acc_na[name] + h0
            # Ours: per-batch recenter + rescale to source (matches diag x_corr)
            mu_t = xf.mean(0); sig_t = xf.std(0)
            xc = (xf - mu_t) * (sig_s / (sig_t + EPS)) + mu_s
            h1 = _compute_code_hist(xc.flatten(), sc, qmax, nb).cpu()
            acc_ours[name] = h1 if name not in acc_ours else acc_ours[name] + h1
    return hook

from dataset.selectedRotateImageFolder import prepare_test_data
def l1(acc):
    ds_ = []
    for n, _ in sel:
        if n not in acc: continue
        ht = acc[n] / (acc[n].sum() + 1e-12)
        hs = src[n].float(); hs = hs / (hs.sum() + 1e-12)
        ds_.append(float((ht - hs).abs().sum()))
    return st.mean(ds_) if ds_ else float("nan")

res = {}
for corr in CORRUPTIONS:
    acc_na.clear(); acc_ours.clear()
    hs = [m.register_forward_pre_hook(mk(n, m)) for n, m in sel]
    sa = types.SimpleNamespace(data=DATA, data_corruption=DATA_C, corruption=corr, level=5,
                               batch_size=64, workers=8, imagenet_5k_eval=True, continual=False,
                               if_shuffle=True, seed=0, imagenet_5k_indices_file=IDS,
                               imagenet_class_map_file=CMAP)
    ds, loader = prepare_test_data(sa)
    if hasattr(ds, "switch_mode"): ds.switch_mode(True, False)
    with torch.no_grad():
        for i, dl in enumerate(loader):
            if i >= MAXB: break
            net(dl[0].cuda())
    for h in hs: h.remove()
    res[corr] = (l1(acc_na), l1(acc_ours))
    print(f"[{corr}] No-Adapt={res[corr][0]:.4f}  Ours={res[corr][1]:.4f}", flush=True)
print("\n=== pooled code-dist (same method as ZOA probe) ===", flush=True)
print(f"  No-Adapt MEAN: {st.mean(v[0] for v in res.values()):.4f}", flush=True)
print(f"  Ours     MEAN: {st.mean(v[1] for v in res.values()):.4f}", flush=True)
