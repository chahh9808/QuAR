"""Build W6A6 source stats (local branch all-layers + cls) at varying source set sizes.

For the source-moment estimation set-size sensitivity (paper App. sensitivity table).
Loads the W6A6 PTQ4ViT net EXACTLY as main.py does (monkeypatch + torch.load full-net),
then reuses collect_qac_stats / collect_cls_stats so the output format matches the
existing 200-batch stats consumed by main.py.

Usage:
  CUDA_VISIBLE_DEVICES=0 python build_setsize_stats.py \
    --data /path/to/imagenet \
    --ptq_checkpoint runs/.../ptq4vit_w6a6_calibrated.pt \
    --set_sizes 8 16 32 64 128 \
    --out_prefix qac_stats_w6a6_all_layers_seed0 \
    --cls_out_prefix cls_stats_w6a6_seed0
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import argparse
import types
import torch

from tools.stats.build_qac_stats import collect_qac_stats
from tools.stats.build_cls_stats import collect_cls_stats


def load_w6a6_fullnet(ptq_checkpoint):
    """Load the calibrated PTQ4ViT W6A6 net exactly as main.py (lines ~1049-1056)."""
    from quant_library.quant_utils.models import (
        attention_forward as _attn_fwd,
        window_attention_forward as _win_attn_fwd,
    )
    from timm.models.vision_transformer import Attention as _Attn
    from timm.models.swin_transformer import WindowAttention as _WinAttn
    _Attn.attention_forward = _attn_fwd
    _WinAttn.window_attention_forward = _win_attn_fwd
    net = torch.load(ptq_checkpoint, map_location="cpu", weights_only=False)
    for _m in net.modules():
        if hasattr(_m, "mode"):
            _m.mode = "quant_forward"
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ptq_checkpoint", required=True)
    ap.add_argument("--quant_w_bit", default=6, type=int)
    ap.add_argument("--quant_a_bit", default=6, type=int)
    ap.add_argument("--quar_local_scope", default="all")
    ap.add_argument("--set_sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out_prefix", default="qac_stats_w6a6_all_layers_seed0")
    ap.add_argument("--cls_out_prefix", default="cls_stats_w6a6_seed0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = load_w6a6_fullnet(args.ptq_checkpoint)
    net = net.cuda()
    # move plain-attr tensors to cuda (PTQ4ViT stores intervals as plain attrs)
    _plain_attrs = ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z")
    for _m in net.modules():
        for _attr in _plain_attrs:
            _t = getattr(_m, _attr, None)
            if isinstance(_t, torch.Tensor) and not _t.is_cuda:
                setattr(_m, _attr, _t.cuda())
    net.eval()
    net.requires_grad_(False)

    source_args = types.SimpleNamespace(
        data=args.data, corruption="original", level=5,
        batch_size=args.batch_size, workers=args.workers,
        imagenet_5k_eval=False, continual=False, if_shuffle=False, seed=args.seed,
    )
    from dataset.selectedRotateImageFolder import prepare_test_data
    source_dataset, source_loader = prepare_test_data(source_args)
    source_dataset.switch_mode(True, False)
    print(f"Source loader: {len(source_loader)} batches x {args.batch_size}.", flush=True)

    for N in args.set_sizes:
        print(f"\n{'='*70}\n  SET SIZE N={N} batches ({N*args.batch_size} images)\n{'='*70}", flush=True)

        # --- the local branch all-layer stats ---
        mean_d, std_d, hist_d = collect_qac_stats(
            net, source_loader, scope=args.quar_local_scope, max_batches=N,
            fp_mode=False, save_code_hist=False)
        local_out = f"{args.out_prefix}_nb{N}.pt"
        torch.save({"mean": mean_d, "std": std_d}, local_out)
        print(f"  saved the local branch ({len(mean_d)} layers) -> {local_out}", flush=True)

        # --- CLS stats ---
        cls = collect_cls_stats(net, source_loader, max_batches=N)
        cls["w_bit"] = args.quant_w_bit
        cls["a_bit"] = args.quant_a_bit
        cls["max_batches"] = N
        cls_out = f"{args.cls_out_prefix}_nb{N}.pt"
        torch.save(cls, cls_out)
        print(f"  saved CLS (n={cls['n_samples']}) -> {cls_out}", flush=True)

    print("\nALL SET SIZES DONE.", flush=True)


if __name__ == "__main__":
    main()
