"""
Backprop-based TTA baselines (TENT, EATA, SAR) for the quantized ViT-B/16.

This module is an ADDITIVE integration: it imports the OFFICIAL SAR-repo
implementations of tent / eata / sar UNCHANGED (they already support
nn.LayerNorm / ViT, see external_repos/SAR/{tent,eata,sar}.py and sam.py) and
BatchNorm-only upstream).

Every deviation from the official code is documented in
runs/backprop_tta_MODIFICATIONS.md. The most important ones:
  * We feed these wrappers the W6A6 PTQ4ViT quantized ViT instead of the FP timm
    ViT. The activation fake-quant uses a straight-through estimator
    (quant_library/quant_layers/linear.py), so gradients flow to the FP
    LayerNorm affine params exactly as in the FP model. No algorithm code is
    changed.
    BatchNorm2d, so we hook nn.LayerNorm outputs (3D, [B, tokens, C]) and match
    per-channel mean+var (reduced over batch AND token dims) by L1 to clean
    source statistics. Loss form, optimizer, lr are kept official.

The official wrappers are torch.nn.Modules whose forward() is decorated with
@torch.enable_grad(), so they adapt correctly even though the pipeline eval loop
(main.validate_adapt) runs under torch.no_grad(). They already expose .reset(),
which the per-corruption episodic loop in main.py calls.

Upstream: the TENT, EATA and SAR implementations these wrap are vendored in sar_official/,
from https://github.com/mr-eggplant/SAR (see the README (Baselines and licences)). This file adds the
straight-through estimator the frozen quantized graph needs and the clean-source loader
EATA's Fisher pass uses.
"""

import os
import sys
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Import the OFFICIAL SAR-repo modules UNCHANGED -------------------------
# Vendored from the official SAR repository; see the README (Baselines and licences).
_SAR_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sar_official")
if _SAR_REPO not in sys.path:
    sys.path.insert(0, _SAR_REPO)

import tent          # noqa: E402  (official SAR-repo tent.py, LayerNorm-aware)
import eata          # noqa: E402  (official SAR-repo eata.py, LayerNorm-aware)
import sar           # noqa: E402  (official SAR-repo sar.py,  LayerNorm-aware)
from sam import SAM  # noqa: E402  (official SAM optimizer, rho=0.05 default)


# ---------------------------------------------------------------------------
# Straight-through estimator (STE) for the PTQ4ViT activation quantizers.
#
# WHY THIS IS REQUIRED (key finding, documented in backprop_tta_MODIFICATIONS.md):
# The W6A6 PTQ4ViT checkpoint has NO adaptive quantizer attrs (qaffine/qzero/
# qstate). For that configuration the activation quantizers in
# quant_library/quant_layers/{linear,matmul}.py fall through to a HARD-ROUND
# path (e.g. linear.py L315-317, matmul.py quant_input_A/_B) that uses
# `.round_()` with NO straight-through estimator. round() has zero gradient
# everywhere, so backprop-based methods (TENT/EATA/SAR) that need gradients to
# flow from the logit-level loss back through the quantized blocks to the FP
# LayerNorm affine params receive EXACTLY zero gradient (verified empirically).
#
# The standard, faithful fake-quant remedy is the straight-through estimator:
# forward returns the quantized value Q(x); backward passes the gradient through
# as identity. We install it ADDITIVELY and REVERSIBLY by wrapping each module's
# bound activation-quant methods at runtime:
#       ste(x) = x + (Q(x) - x).detach()
# This changes NO quant-library source and only the backprop_tta builders enable
# it. It does not affect the forward numerics (Q(x) is unchanged); it only makes
# the gradient identity through the rounding, which is exactly what fake-quant
# training/adaptation assumes. Inference-only methods (NoAdapt/T3A/ZOA/FOZO/Ours)
# never call install_ste, so their behavior is untouched.
# ---------------------------------------------------------------------------
_STE_METHODS = ("quant_input", "quant_input_A", "quant_input_B")


def _make_ste(orig):
    def _ste(x, *a, **k):
        q = orig(x, *a, **k)
        # Identity backward, quantized forward. Guard against shape changes.
        if q.shape == x.shape:
            return x + (q - x).detach()
        return q
    return _ste


def install_ste(model):
    """Install the straight-through estimator on every activation quantizer.

    Idempotent and reversible (see remove_ste). Stores the original bound method
    on the instance under `_ste_orig_<name>`.
    """
    n = 0
    for m in model.modules():
        for name in _STE_METHODS:
            fn = getattr(m, name, None)
            if fn is None or not callable(fn):
                continue
            if hasattr(m, f"_ste_orig_{name}"):
                continue  # already installed
            setattr(m, f"_ste_orig_{name}", fn)
            setattr(m, name, _make_ste(fn))
            n += 1
    return n


def remove_ste(model):
    for m in model.modules():
        for name in _STE_METHODS:
            saved = getattr(m, f"_ste_orig_{name}", None)
            if saved is not None:
                setattr(m, name, saved)
                delattr(m, f"_ste_orig_{name}")


# ---------------------------------------------------------------------------
# Learning-rate formula (faithful to external_repos/SAR/main.py)
# ---------------------------------------------------------------------------
def official_vit_lr(batch_size):
    """ViT-base lr from SAR/main.py line 193: args.lr = (0.001 / 64) * bs.

    NOTE on the conditional doubling: SAR/main.py lines 205-207 double the lr when
    `exp_type == 'bs1' and method == 'sar'` (single-sample SAR). At the published
    batch_size=64 protocol that branch does not apply, so the lr is not doubled.
    Pass --sar_bs1_double_lr to reproduce it when running single-sample SAR.
    See backprop_tta_MODIFICATIONS.md.
    """
    return (0.001 / 64.0) * float(batch_size)


# ---------------------------------------------------------------------------
# Clean source loader (for EATA Fisher)
# ---------------------------------------------------------------------------
def build_clean_source_loader(args, obtain_train_loader, size):
    """Return (dataset, loader) over clean ImageNet 'original', restricted to
    `size` samples. Reuses the pipeline's own obtain_train_loader (which sets
    corruption='original' and switch_mode(True, False)), matching the clean
    source the rest of the pipeline uses (FOA.obtain_origin_stat, etc.).

    This mirrors official EATA's fisher loader construction in SAR/main.py
    (prepare_test_data with corruption='original' + set_dataset_size), but goes
    through the pipeline's loader so we use the SAME clean ImageNet root and
    transforms as the rest of this pipeline.
    """
    dataset, _ = obtain_train_loader(args)
    if size is not None and size > 0:
        dataset.set_dataset_size(size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "workers", 4),
        pin_memory=True,
    )
    return dataset, loader


# ---------------------------------------------------------------------------
# TENT (official SAR-repo tent.py)
# ---------------------------------------------------------------------------
def build_tent(args, net):
    """Configure + wrap the quantized ViT with the official Tent module.

    Faithful to SAR/main.py lines 211-216:
        net = tent.configure_model(net)
        params, _ = tent.collect_params(net)
        optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
        tent.Tent(net, optimizer)   # steps=1, episodic=False (class defaults)
    """
    lr = official_vit_lr(args.batch_size)
    install_ste(net)  # enable gradient flow through PTQ4ViT activation quant
    net = tent.configure_model(net)
    params, param_names = tent.collect_params(net)
    optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    model = tent.Tent(net, optimizer)
    return model, param_names, lr


# ---------------------------------------------------------------------------
# EATA (official SAR-repo eata.py) + official Fisher pre-computation
# ---------------------------------------------------------------------------
def build_eata(args, net, obtain_train_loader, logger=None):
    """Configure + wrap with official EATA, including the official Fisher
    pre-computation on clean source data.

    Faithful to SAR/main.py lines 238-274:
        - fisher loader = clean 'original', set_dataset_size(fisher_size)
        - net = eata.configure_model(net); params = eata.collect_params(net)
        - ewc_optimizer = SGD(params, 0.001)
        - per-batch: outputs = net(x); pseudo-labels = outputs.max(1);
          loss = CE(outputs, pseudo); loss.backward(); accumulate grad^2 as Fisher
        - optimizer = SGD(params, args.lr, momentum=0.9)
        - eata.EATA(net, optimizer, fishers, fisher_alpha, e_margin, d_margin)
    """
    lr = official_vit_lr(args.batch_size)

    install_ste(net)  # enable gradient flow through PTQ4ViT activation quant
    # --- Fisher pre-computation on clean source (official procedure) --------
    _, fisher_loader = build_clean_source_loader(args, obtain_train_loader,
                                                 args.eata_fisher_size)
    net = eata.configure_model(net)
    params, param_names = eata.collect_params(net)
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    n_batches = len(fisher_loader)
    # The Fisher pass needs grads; the pipeline runs under no_grad by default.
    with torch.enable_grad():
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            images = images.cuda(non_blocking=True)
            outputs = net(images)
            _, pseudo = outputs.max(1)
            loss = train_loss_fn(outputs, pseudo)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == n_batches:
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
    if logger is not None:
        logger.info(f"[EATA] computed Fisher over {n_batches} clean batches "
                    f"({len(fishers)} param tensors)")
    del ewc_optimizer

    optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    model = eata.EATA(net, optimizer, fishers,
                      fisher_alpha=args.eata_fisher_alpha,
                      e_margin=args.eata_e_margin,
                      d_margin=args.eata_d_margin)
    return model, param_names, lr


# ---------------------------------------------------------------------------
# SAR (official SAR-repo sar.py) + SAM optimizer
# ---------------------------------------------------------------------------
def build_sar(args, net):
    """Configure + wrap with official SAR.

    Faithful to SAR/main.py lines 285-292:
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)   # skips blocks 9/10/11 + final norm
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)  # rho=0.05
        sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
    """
    lr = official_vit_lr(args.batch_size)
    if getattr(args, 'sar_bs1_double_lr', False):
        # SAR/main.py L205-207: single-sample SAR runs at twice the scaled rate.
        lr = 2 * lr
    install_ste(net)  # enable gradient flow through PTQ4ViT activation quant
    net = sar.configure_model(net)
    params, param_names = sar.collect_params(net)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
    model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
    return model, param_names, lr


# ---------------------------------------------------------------------------
def _batch_ln_stats(outputs):
    """Per-channel mean/var of a single batch's hooked output [B, T, C] -> C.

    Computed WITH grad (no detach) so the L1-match loss backprops into params.
    """
    x = outputs.reshape(-1, outputs.shape[-1])
    return torch.mean(x, dim=0), torch.var(x, dim=0)


class _BatchHook:
    """Captures a single LayerNorm output tensor (with grad) for one batch."""
    def __init__(self):
        self.output = None

    def __call__(self, module, module_in, module_out):
        self.output = module_out

