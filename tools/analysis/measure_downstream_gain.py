#!/usr/bin/env python
"""
Empirical downstream perturbation-gain diagnostic for W6A6 PTQ4ViT ViT-B/16.

Measures, per transformer block i (0..11), the empirical gain
    gain_i = mean_b  ||z_pert_b - z_clean_b||_2 / ||delta_b||_2
where delta is a per-sample random Gaussian perturbation added to the *input* of
block i with ||delta_b|| = eps_rel * ||h_{i,b}|| (h_{i,b} = clean block-i input).

This is the empirical analogue of the downstream amplification factor K_{>l}
(product of per-block operator norms after interface l). We verify whether it
DECREASES with depth (late blocks -> smaller downstream gain).

Primary result: FP ('raw') mode. Secondary robustness check: 'quant_forward' mode.

Does NOT modify any existing source/launcher file. Standalone.
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import os, sys, json
import numpy as np
import torch

# --- patch attention forwards exactly as instructed for checkpoint load ---
from quant_library.quant_utils.models import (
    attention_forward as _attn_fwd,
    window_attention_forward as _win_attn_fwd,
)
from timm.models.vision_transformer import Attention as _Attn
from timm.models.swin_transformer import WindowAttention as _WinAttn
_Attn.attention_forward = _attn_fwd
_WinAttn.window_attention_forward = _win_attn_fwd

CKPT = _os.path.join(_REPO_ROOT, "checkpoints/ptq4vit_w6a6_vitb.pt")
IMAGENET_ROOT = os.environ.get("DATA", "")
# Figure output. Defaults into outputs/; set FIG_OUT to write it into a paper tree instead.
FIG_OUT = os.environ.get("FIG_OUT", _os.path.join(_REPO_ROOT, "outputs", "fig_downstream_gain.pdf"))
RESULTS_OUT = _os.path.join(_REPO_ROOT, "outputs", "downstream_gain_results.json")

DEEP_RED = "#750014"
N_BLOCKS = 12
K_DIRS = 8           # random directions averaged per block
EPS_REL = 1e-2       # local-linear regime
BATCH_TARGET = 64    # concatenate batches up to ~64 images
SEED = 0


def load_model():
    net = torch.load(CKPT, map_location="cpu", weights_only=False)
    for m in net.modules():
        for a in ("w_interval", "a_interval", "A_interval", "B_interval", "qzero_z"):
            t = getattr(m, a, None)
            if isinstance(t, torch.Tensor) and not t.is_cuda:
                setattr(m, a, t.cuda())
    net = net.cuda().eval()
    net.requires_grad_(False)
    return net


def set_mode(net, mode):
    """Set every module that has a `.mode` attr to `mode`. Returns count set."""
    n = 0
    for m in net.modules():
        if hasattr(m, "mode"):
            m.mode = mode
            n += 1
    return n


def get_batch(net):
    """Return (images[B,3,224,224] cuda float32, target[B]). Try ImageNet val, else ImageNet-C."""
    src = None
    try:
        from quant_library.quant_utils import datasets
        g = datasets.ViTImageNetLoaderGenerator(
            IMAGENET_ROOT, "imagenet", 32, 32, 16, kwargs={"model": net}
        )
        loader = g.test_loader()
        imgs, tgts = [], []
        for x, y in loader:
            imgs.append(x); tgts.append(y)
            if sum(b.shape[0] for b in imgs) >= BATCH_TARGET:
                break
        images = torch.cat(imgs, 0)[:BATCH_TARGET]
        target = torch.cat(tgts, 0)[:BATCH_TARGET]
        src = f"clean ImageNet val ({IMAGENET_ROOT})"
        return images.cuda().float(), target.cuda(), src
    except Exception as e:
        print(f"[warn] clean ImageNet loader failed: {repr(e)[:300]}")

    # fallback: ImageNet-C gaussian_noise sev5 via timm transforms
    import torchvision.transforms as T
    import torchvision.datasets as tvd
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    cdir = os.environ.get("DATA_C", "") + "/gaussian_noise/5"
    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    ds = tvd.ImageFolder(cdir, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_TARGET, shuffle=True, num_workers=8)
    images, target = next(iter(loader))
    src = f"ImageNet-C gaussian_noise sev5 ({cdir})"
    return images.cuda().float(), target.cuda(), src


def capture_block_inputs(net, images):
    """Run one clean pass, capturing each block's input h_i (input[0])."""
    h = {}
    handles = []

    def mk(i):
        def hook(mod, inp):
            h[i] = inp[0].detach().clone()
            return None
        return hook

    for i in range(N_BLOCKS):
        handles.append(net.blocks[i].register_forward_pre_hook(mk(i)))
    with torch.no_grad():
        _ = net(images)
    for hd in handles:
        hd.remove()
    return h  # dict i -> [B,197,768]


def measure_gain_for_block(net, images, h_i, eps_rel, k_dirs, gen):
    """gain averaged over k_dirs random directions for one block. delta added to block input."""
    # h_i: [B,N,D]; per-sample norm
    h_norm = h_i.reshape(h_i.shape[0], -1).norm(dim=1)  # [B]
    gains = []
    for k in range(k_dirs):
        d = torch.randn(h_i.shape, generator=gen, device="cpu").to(h_i.device, h_i.dtype)
        d_norm = d.reshape(d.shape[0], -1).norm(dim=1)  # [B]
        scale = (eps_rel * h_norm / (d_norm + 1e-12)).reshape(-1, 1, 1)
        delta = d * scale  # ||delta_b|| = eps_rel * ||h_b||
        delta_norm = delta.reshape(delta.shape[0], -1).norm(dim=1)  # [B]

        store = {}

        def hook(mod, inp):
            x = inp[0]
            return (x + delta,) + tuple(inp[1:])

        # need block index -> use closure; find which block via identity outside
        # (we pass the specific block module below)
        return_val = None
        # handled by caller; placeholder
        gains_k = None
        store["delta"] = delta
        store["delta_norm"] = delta_norm
        store["hook"] = hook
        gains.append(store)
    return gains


def run_mode(net, images, z_clean, h, eps_rel, k_dirs, base_seed):
    """Returns list of 12 gains for current mode."""
    res = [0.0] * N_BLOCKS
    for i in range(N_BLOCKS):
        h_i = h[i]
        h_norm = h_i.reshape(h_i.shape[0], -1).norm(dim=1)
        per_dir = []
        for k in range(k_dirs):
            gen = torch.Generator(device="cpu").manual_seed(base_seed + 1000 * i + k)
            d = torch.randn(h_i.shape, generator=gen).to(h_i.device, h_i.dtype)
            d_norm = d.reshape(d.shape[0], -1).norm(dim=1)
            scale = (eps_rel * h_norm / (d_norm + 1e-12)).reshape(-1, 1, 1)
            delta = d * scale
            delta_norm = delta.reshape(delta.shape[0], -1).norm(dim=1)

            def hook(mod, inp, _delta=delta):
                return (inp[0] + _delta,) + tuple(inp[1:])

            hd = net.blocks[i].register_forward_pre_hook(hook)
            with torch.no_grad():
                z_pert = net(images)
            hd.remove()

            dz = (z_pert - z_clean).reshape(z_pert.shape[0], -1).norm(dim=1)  # [B]
            g_b = dz / (delta_norm + 1e-12)  # [B]
            per_dir.append(g_b.mean().item())
        res[i] = float(np.mean(per_dir))
    return res


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    net = load_model()
    assert len(net.blocks) == N_BLOCKS, f"expected 12 blocks, got {len(net.blocks)}"

    images, target, src = get_batch(net)
    B = images.shape[0]
    assert images.shape[1:] == (3, 224, 224), f"bad img shape {images.shape}"
    print(f"[data] source = {src} | batch = {tuple(images.shape)} | dtype={images.dtype} dev={images.device}")

    out = {"data_source": src, "batch_size": B, "eps_rel": EPS_REL, "k_dirs": K_DIRS, "seed": SEED}

    # ---------- FP / raw mode ----------
    n_raw = set_mode(net, "raw")
    with torch.no_grad():
        z_clean_fp = net(images)
    assert torch.isfinite(z_clean_fp).all(), "FP logits not finite"
    top1_fp = (z_clean_fp.argmax(1) == target).float().mean().item() * 100
    print(f"[FP/raw] set mode on {n_raw} modules | logits {tuple(z_clean_fp.shape)} | sanity top-1 = {top1_fp:.2f}%")
    out["mode_fp_string"] = "raw"
    out["n_modules_with_mode"] = n_raw
    out["sanity_top1_fp"] = top1_fp

    h_fp = capture_block_inputs(net, images)
    gains_fp = run_mode(net, images, z_clean_fp, h_fp, EPS_REL, K_DIRS, base_seed=10000)
    out["gain_fp"] = gains_fp

    # eps-stability check (local-linearity) on one early (0) and one late (10) block
    eps_check = {}
    for blk in (0, 10):
        eps_check[str(blk)] = {}
        for eps in (1e-3, 1e-2, 1e-1):
            g = run_mode(net, images, z_clean_fp, h_fp, eps, K_DIRS, base_seed=20000)
            eps_check[str(blk)][str(eps)] = g[blk]
    out["eps_stability_fp"] = eps_check

    # ---------- quant_forward mode (secondary) ----------
    set_mode(net, "quant_forward")
    with torch.no_grad():
        z_clean_q = net(images)
    top1_q = (z_clean_q.argmax(1) == target).float().mean().item() * 100
    print(f"[quant] sanity top-1 = {top1_q:.2f}%")
    out["sanity_top1_quant"] = top1_q
    h_q = capture_block_inputs(net, images)
    gains_q = run_mode(net, images, z_clean_q, h_q, EPS_REL, K_DIRS, base_seed=10000)
    out["gain_quant"] = gains_q

    # ---------- monotonicity verdict ----------
    def monotone_decreasing(v):
        diffs = np.diff(v)
        strict = bool(np.all(diffs < 0))
        n_increase = int(np.sum(diffs > 0))
        # Spearman-ish: correlation of value with index (negative => decreasing trend)
        idx = np.arange(len(v))
        r = float(np.corrcoef(idx, v)[0, 1])
        return strict, n_increase, r

    strict_fp, ninc_fp, corr_fp = monotone_decreasing(gains_fp)
    early_fp = float(np.mean(gains_fp[0:4]))
    late_fp = float(np.mean(gains_fp[8:12]))
    out["verdict_fp"] = {
        "strict_monotone_decreasing": strict_fp,
        "num_increasing_steps": ninc_fp,
        "corr_index_vs_gain": corr_fp,
        "mean_early_0_3": early_fp,
        "mean_late_8_11": late_fp,
        "late_lt_early": bool(late_fp < early_fp),
    }
    strict_q, ninc_q, corr_q = monotone_decreasing(gains_q)
    out["verdict_quant"] = {
        "strict_monotone_decreasing": strict_q,
        "num_increasing_steps": ninc_q,
        "corr_index_vs_gain": corr_q,
        "mean_early_0_3": float(np.mean(gains_q[0:4])),
        "mean_late_8_11": float(np.mean(gains_q[8:12])),
    }

    # ---------- table ----------
    print("\n=== Downstream perturbation gain  ||Δz|| / ||δ||  (eps_rel=%.0e, K=%d dirs) ===" % (EPS_REL, K_DIRS))
    print(f"{'blk':>4} | {'gain (FP/raw)':>14} | {'gain (quant)':>14}")
    print("-" * 40)
    for i in range(N_BLOCKS):
        print(f"{i:>4} | {gains_fp[i]:>14.4f} | {gains_q[i]:>14.4f}")
    print("-" * 40)
    print(f"FP : strict-decreasing={strict_fp}  #increasing-steps={ninc_fp}/11  corr(idx,gain)={corr_fp:+.3f}")
    print(f"FP : mean early(0-3)={early_fp:.4f}  mean late(8-11)={late_fp:.4f}  late<early={late_fp<early_fp}")
    print(f"Qt : strict-decreasing={strict_q}  #increasing-steps={ninc_q}/11  corr(idx,gain)={corr_q:+.3f}")
    print("\n=== eps-stability (FP), gain at the named block ===")
    for blk in ("0", "10"):
        row = " ".join(f"eps={e}:{eps_check[blk][e]:.4f}" for e in eps_check[blk])
        print(f"  block {blk}: {row}")

    # ---------- save json ----------
    with open(RESULTS_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] results -> {RESULTS_OUT}")

    # ---------- plot ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    xs = list(range(N_BLOCKS))
    span = max(gains_fp) / max(min(gains_fp), 1e-9)
    ax.plot(xs, gains_fp, "-o", color=DEEP_RED, lw=2, ms=5)
    if span > 50:
        ax.set_yscale("log")
    ax.set_xlabel("block index ℓ")
    ax.set_ylabel(r"downstream gain  $\|\Delta z\|/\|\delta\|$")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT)
    print(f"[saved] figure  -> {FIG_OUT}  (log-y={span>50})")


if __name__ == "__main__":
    main()
