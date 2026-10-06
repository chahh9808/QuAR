"""EPISODIC ZOA code-distance at post-LN B3 interfaces: rebuild a FRESH ZOA adapter per
corruption (reset adaptation state, matching the episodic protocol of tab:main / the teaser),
then measure the pooled quantizer code-distribution distance to source. Fixes the continual
divergence of probe_zoa_codedist.py (which reused one adapter across all 15 corruptions)."""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import sys, types, tempfile, torch, statistics as st
sys.path.insert(0, _REPO_ROOT)
DATA = os.environ.get("DATA", "")
DATA_C = os.environ.get("DATA_C", "")
PTQ = f"{_REPO_ROOT}/checkpoints/ptq4vit_w6a6_vitb.pt"
IDS = f"{_REPO_ROOT}/robustbench/data/imagenet_test_image_ids.txt"
CMAP = f"{_REPO_ROOT}/robustbench/data/imagenet_class_to_id_map.json"
CORRUPTIONS = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
               "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
               "elastic_transform","pixelate","jpeg_compression"]
MAXB = 25
tmpd = tempfile.mkdtemp()
sys.argv = ["main.py","--data",DATA,"--data_corruption",DATA_C,"--level","5",
            "--quant","--quant_w_bit","6","--quant_a_bit","6","--ptq_checkpoint",PTQ,
            "--algorithm","zoa","--official_zoa_stats_file",f"{_REPO_ROOT}/zoa_stats/w6a6.pt",
            "--batch_size","64","--workers","8","--seed","0","--output",tmpd,
            "--imagenet_5k_eval","--imagenet_5k_indices_file",IDS,
            "--imagenet_class_map_file",CMAP,"--corruption",CORRUPTIONS[0]]
import main, logging
logging.basicConfig(level=logging.ERROR); main.logger = logging.getLogger("probe_zoa_epi")
args = main.get_args()
from tools.stats.build_setsize_stats import load_w6a6_fullnet
from tools.stats.build_qac_stats import _is_quant_act, _get_quant_scale_qmax, _compute_code_hist, _block_idx
from dataset.selectedRotateImageFolder import prepare_test_data
src = torch.load(f"{_REPO_ROOT}/stats/qac_stats_w6a6_codehist_all.pt", map_location="cpu").get("code_hist", {})

def fresh_zoa():
    net = load_w6a6_fullnet(PTQ).cuda()
    for _m in net.modules():
        for _a in ("w_interval","a_interval","A_interval","B_interval","qzero_z"):
            _t = getattr(_m,_a,None)
            if isinstance(_t,torch.Tensor) and not _t.is_cuda: setattr(_m,_a,_t.cuda())
    model = main.build_official_zoa_adapter(args, net, args.output)
    sel = []
    for name,m in net.named_modules():
        if not _is_quant_act(m): continue
        if _block_idx(name) not in (8,9,10,11) or "fc2" in name: continue
        if not any(k in name for k in ("attn.qkv","attn.proj","mlp.fc1")): continue
        if name in src: sel.append((name,m))
    return net, model, sel

results = {}
for corr in CORRUPTIONS:
    net, model, sel = fresh_zoa()       # <-- EPISODIC: fresh adapter per corruption
    accum = {}
    def mk(name, m):
        sc,qmax,nb = _get_quant_scale_qmax(m)
        def hook(mod, inp):
            if sc is None or not inp: return
            x = inp[0]
            if not torch.is_tensor(x): return
            with torch.no_grad():
                h = _compute_code_hist(x.detach().flatten(), sc, qmax, nb).cpu()
            accum[name] = h if name not in accum else accum[name]+h
        return hook
    hs = [m.register_forward_pre_hook(mk(n,m)) for n,m in sel]
    sa = types.SimpleNamespace(data=DATA, data_corruption=DATA_C, corruption=corr, level=5,
                               batch_size=64, workers=8, imagenet_5k_eval=True, continual=False,
                               if_shuffle=True, seed=0, imagenet_5k_indices_file=IDS,
                               imagenet_class_map_file=CMAP)
    ds, loader = prepare_test_data(sa)
    if hasattr(ds,"switch_mode"): ds.switch_mode(True, False)
    with torch.no_grad():
        for i, dl in enumerate(loader):
            if i >= MAXB: break
            model(dl[0].cuda())
    for h in hs: h.remove()
    dists = []
    for n,_ in sel:
        if n not in accum: continue
        ht = accum[n]/(accum[n].sum()+1e-12); hsrc = src[n].float(); hsrc = hsrc/(hsrc.sum()+1e-12)
        dists.append(float((ht-hsrc).abs().sum()))
    results[corr] = st.mean(dists) if dists else float("nan")
    print(f"[{corr}] EPISODIC ZOA code-dist = {results[corr]:.4f}", flush=True)
    del net, model
    torch.cuda.empty_cache()
print(f"\nEPISODIC ZOA MEAN (15 corr) = {st.mean(list(results.values())):.4f}", flush=True)
