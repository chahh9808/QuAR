#!/usr/bin/env python3
"""Build all-layers (blocks 0-11 x {qkv,proj,fc1,fc2}, scope='all') local-branch source stats for a SAVED
PTQ4ViT full-net checkpoint, applying the same Attention.attention_forward monkeypatch main.py uses
(build_qac_stats.py omits it, so torch.load of the saved ptq4vit net fails). Source = clean ImageNet
(corruption='original'), matching how the AdaLog all-layers stats were built. Reuses
build_qac_stats.collect_qac_stats so the stats are computed identically.

Usage: python build_ptq4vit_alllayers_stats.py --data <imagenet> --ptq_checkpoint X.pt --w 4 --a 4 --output Y.pt
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import argparse, types
import torch
import build_qac_stats as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ptq_checkpoint", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=200)
    args = ap.parse_args()

    # monkeypatch (identical to main.py) so the saved full-net unpickles
    from quant_library.quant_utils.models import attention_forward as _attn_fwd, window_attention_forward as _win_attn_fwd
    from timm.models.vision_transformer import Attention as _Attn
    from timm.models.swin_transformer import WindowAttention as _WinAttn
    _Attn.attention_forward = _attn_fwd
    _WinAttn.window_attention_forward = _win_attn_fwd

    print(f"Loading PTQ4ViT full-net {args.ptq_checkpoint} (W{args.w}A{args.a}) ...")
    net = torch.load(args.ptq_checkpoint, map_location="cpu", weights_only=False)
    for m in net.modules():
        if hasattr(m, "mode"):
            m.mode = "quant_forward"
    net = net.cuda()
    # Move plain-attr quantizer tensors that .cuda() doesn't reach (identical to main.py)
    for m in net.modules():
        for attr in ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z"):
            t = getattr(m, attr, None)
            if isinstance(t, torch.Tensor) and not t.is_cuda:
                setattr(m, attr, t.cuda())
    net.eval()
    net.requires_grad_(False)

    source_args = types.SimpleNamespace(
        data=args.data, corruption="original", level=5,
        batch_size=args.batch_size, workers=args.workers,
        imagenet_5k_eval=False, continual=False, if_shuffle=False, seed=args.seed,
    )
    from dataset.selectedRotateImageFolder import prepare_test_data
    ds, loader = prepare_test_data(source_args)
    ds.switch_mode(True, False)
    print(f"Source loader: {len(loader)} batches x {args.batch_size} (corruption=original).")

    mean_dict, std_dict, _ = B.collect_qac_stats(net, loader, scope="all",
                                                 max_batches=args.max_batches, fp_mode=False)
    torch.save({"mean": mean_dict, "std": std_dict}, args.output)
    print(f"Saved {len(mean_dict)} layers to {args.output}.")


if __name__ == "__main__":
    main()
