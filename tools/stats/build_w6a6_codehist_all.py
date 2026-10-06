"""Build W6A6 source code-histogram + moment stats at FULL scope (all 48 interfaces,
incl. fc2) from the deployed calibrated PTQ4ViT net, for the B3-scope code/saturation
diagnostic."""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import sys
sys.path.insert(0, _REPO_ROOT)
import argparse, types, torch
from tools.stats.build_setsize_stats import load_w6a6_fullnet
from tools.stats.build_qac_stats import collect_qac_stats

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ptq_checkpoint", required=True)
ap.add_argument("--max_batches", default=200, type=int)
ap.add_argument("--output", required=True)
args = ap.parse_args()

torch.manual_seed(0); torch.cuda.manual_seed(0)
net = load_w6a6_fullnet(args.ptq_checkpoint).cuda()
for _m in net.modules():
    for _a in ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z"):
        _t = getattr(_m, _a, None)
        if isinstance(_t, torch.Tensor) and not _t.is_cuda:
            setattr(_m, _a, _t.cuda())
net.eval(); net.requires_grad_(False)

sa = types.SimpleNamespace(data=args.data, corruption="original", level=5,
                           batch_size=64, workers=8, imagenet_5k_eval=False,
                           continual=False, if_shuffle=False, seed=0)
from dataset.selectedRotateImageFolder import prepare_test_data
ds, loader = prepare_test_data(sa); ds.switch_mode(True, False)
print(f"source loader: {len(loader)} batches", flush=True)

mean_d, std_d, hist_d = collect_qac_stats(net, loader, scope="all",
                                          max_batches=args.max_batches,
                                          fp_mode=False, save_code_hist=True)
payload = {"mean": mean_d, "std": std_d}
if hist_d:
    payload["code_hist"] = hist_d
torch.save(payload, args.output)
print(f"saved {len(mean_d)} layers ({len(hist_d)} with code_hist) -> {args.output}", flush=True)
