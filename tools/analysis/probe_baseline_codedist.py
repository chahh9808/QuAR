"""Measure ANY baseline's post-LN B3 code-distance to source, reusing main.main()'s exact
adapter setup + episodic per-corruption reset. Monkeypatches main.validate_adapt to register
code-dist hooks; runs main.main() once for the given --algorithm. Usage:
  python probe_baseline_codedist.py <algorithm>   e.g. fozo | foa | t3a | no_adapt
Pooled code counts per corruption; L1 to source (qac_stats_w6a6_codehist_all.pt)."""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import sys, re, tempfile, torch
sys.path.insert(0, _REPO_ROOT)
DATA=os.environ.get("DATA", ""); DATA_C=os.environ.get("DATA_C", "")
PTQ=f"{_REPO_ROOT}/checkpoints/ptq4vit_w6a6_vitb.pt"
IDS=f"{_REPO_ROOT}/robustbench/data/imagenet_test_image_ids.txt"; CMAP=f"{_REPO_ROOT}/robustbench/data/imagenet_class_to_id_map.json"
ALG = sys.argv[1] if len(sys.argv)>1 else "fozo"
MAXB = "25"; tmpd=tempfile.mkdtemp()
argv=["main.py","--data",DATA,"--data_corruption",DATA_C,"--level","5",
      "--quant","--quant_w_bit","6","--quant_a_bit","6","--ptq_checkpoint",PTQ,
      "--algorithm",ALG,"--batch_size","64","--workers","8","--seed","0","--output",tmpd,
      "--corruption","all","--imagenet_5k_eval","--imagenet_5k_indices_file",IDS,
      "--imagenet_class_map_file",CMAP,"--max_eval_batches",MAXB]
if ALG=="foa": argv += ["--popsize","1"]   # match the paper's minimum-forward FOA
sys.argv=argv
import main, logging, statistics as st
logging.basicConfig(level=logging.ERROR)
from tools.stats.build_qac_stats import _is_quant_act, _get_quant_scale_qmax, _compute_code_hist
SRC = torch.load(f"{_REPO_ROOT}/stats/qac_stats_w6a6_codehist_all.pt", map_location="cpu").get("code_hist", {})
KEY = re.compile(r"(blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.fc1))")  # post-LN B3 (excl fc2)
def canon(name):
    m=KEY.search(name);
    if not m: return None
    k=m.group(1); b=int(k.split('.')[1])
    return k if b in (8,9,10,11) else None

RESULTS={}
_orig = main.validate_adapt
def patched(val_loader, model, args):
    accum={}
    sel=[]
    for name,mod in model.named_modules():
        if not _is_quant_act(mod): continue
        k=canon(name)
        if k is None or k not in SRC: continue
        sel.append((k,mod))
    def mk(k,mod):
        sc,qmax,nb=_get_quant_scale_qmax(mod)
        def hook(m,inp):
            if sc is None or not inp: return
            x=inp[0]
            if not torch.is_tensor(x): return
            with torch.no_grad():
                h=_compute_code_hist(x.detach().flatten(),sc,qmax,nb).cpu()
            accum[k]=h if k not in accum else accum[k]+h
        return hook
    hs=[mod.register_forward_pre_hook(mk(k,mod)) for k,mod in sel]
    out=_orig(val_loader, model, args)
    for h in hs: h.remove()
    dists=[]
    for k in set(kk for kk,_ in sel):
        if k not in accum: continue
        ht=accum[k]/(accum[k].sum()+1e-12); hs2=SRC[k].float(); hs2=hs2/(hs2.sum()+1e-12)
        dists.append(float((ht-hs2).abs().sum()))
    d=st.mean(dists) if dists else float("nan")
    RESULTS[getattr(args,"corruption","?")]=d
    print(f"[CODEDIST {ALG}] {getattr(args,'corruption','?')} = {d:.4f}  (n={len(dists)})", flush=True)
    return out
main.validate_adapt = patched
try:
    main.main()
except SystemExit:
    pass
print(f"\n=== {ALG} post-LN B3 code-dist (pooled, episodic via main) ===", flush=True)
for c,d in RESULTS.items(): print(f"  {c}: {d:.4f}", flush=True)
if RESULTS: print(f"  MEAN = {st.mean(list(RESULTS.values())):.4f}", flush=True)
