"""Source per-channel (mean, std) of the RESIDUAL STREAM, i.e. the output of whole transformer
blocks, on the PTQ4ViT full-net checkpoint. Placement control for the local branch: the same
recalibration at a tensor no activation quantizer reads. Mirrors build_ptq4vit_alllayers_stats.py
(loading, calibration loader, pooled-moment aggregation) and hooks block OUTPUTS instead."""
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import argparse, types
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ptq_checkpoint", default="")
    ap.add_argument("--adalog_checkpoint", default="", help="AdaLog state_dict (W4A4/W3A3); loads via build_qac_stats.load_adalog_net")
    ap.add_argument("--w", type=int, default=6); ap.add_argument("--a", type=int, default=6)
    ap.add_argument("--output", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=200)
    ap.add_argument("--blocks", default="0,1,2,3,4,5,6,7,8,9,10,11")
    args = ap.parse_args()
    if args.adalog_checkpoint:
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        import build_qac_stats as B
        net = B.load_adalog_net(args.adalog_checkpoint, w_bit=args.w, a_bit=args.a)
    else:
        from quant_library.quant_utils.models import attention_forward as _attn_fwd, window_attention_forward as _win_attn_fwd
        from timm.models.vision_transformer import Attention as _Attn
        from timm.models.swin_transformer import WindowAttention as _WinAttn
        _Attn.attention_forward = _attn_fwd; _WinAttn.window_attention_forward = _win_attn_fwd
        net = torch.load(args.ptq_checkpoint, map_location="cpu", weights_only=False)
        for m in net.modules():
            if hasattr(m, "mode"): m.mode = "quant_forward"
    net = net.cuda()
    for m in net.modules():
        for attr in ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z"):
            t = getattr(m, attr, None)
            if isinstance(t, torch.Tensor) and not t.is_cuda: setattr(m, attr, t.cuda())
    net.eval(); net.requires_grad_(False)
    source_args = types.SimpleNamespace(data=args.data, corruption="original", level=5, batch_size=args.batch_size,
                                        workers=args.workers, imagenet_5k_eval=False, continual=False, if_shuffle=False, seed=args.seed)
    from dataset.selectedRotateImageFolder import prepare_test_data
    ds, loader = prepare_test_data(source_args); ds.switch_mode(True, False)
    want = {f"blocks.{b}" for b in args.blocks.split(",")}
    mods = [(n, m) for n, m in net.named_modules() if n in want]
    print(f"Hooking {len(mods)} block outputs: {[n for n,_ in mods]}")
    accum = {}
    def make_hook(name):
        def hook(module, inputs, output):
            with torch.no_grad():
                x = output.detach().float().reshape(-1, output.shape[-1])
                s1, s2, c = x.sum(0).cpu(), x.pow(2).sum(0).cpu(), x.shape[0]
            if name not in accum: accum[name] = [s1, s2, c]
            else: accum[name][0] += s1; accum[name][1] += s2; accum[name][2] += c
        return hook
    hs = [m.register_forward_hook(make_hook(n)) for n, m in mods]
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches > 0 and i >= args.max_batches: break
            net(batch[0].cuda())
            if (i + 1) % 50 == 0: print(f"  batch {i+1}/{args.max_batches}", flush=True)
    for h in hs: h.remove()
    mean_dict, std_dict = {}, {}
    for n, (s1, s2, c) in accum.items():
        mu = s1 / c; sig = ((s2 / c) - mu.pow(2)).clamp_min(0.0).sqrt()
        mean_dict[n] = mu; std_dict[n] = sig
        print(f"  {n:<12} C={mu.shape[0]} mean_mean={float(mu.mean()):.4f} mean_std={float(sig.mean()):.4f}")
    torch.save({"mean": mean_dict, "std": std_dict}, args.output); print(f"Saved {len(mean_dict)} block outputs to {args.output}")

if __name__ == "__main__":
    main()
