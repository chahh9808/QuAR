"""AQR: Adaptive Quantile Recalibration (Mehrbod et al., WACV 2026) - baseline.

Independent re-implementation for baseline comparison against QuAR/the local branch.  No
official code is available; this follows the method and implementation details
described in the paper (Sec. 3).

Method
------
Setup (offline, see build_aqr_stats.py):
    Run source data through the SAME (quantized W6A6) model.  For each selected
    layer l and each channel c, store the source percentiles p^S_i,
    i in {0, 1, ..., 100} of the pre-activation values a^l_c.

Inference (per test batch, episodic / stateless):
    For each selected layer, compute the target percentiles p^T_i of the
    current batch (per channel), then map each activation value x that lies in
    the interval [p^T_i, p^T_{i+1}] with a piecewise-linear transform:

        AQR(x) = p^S_i + (x - p^T_i) / (p^T_{i+1} - p^T_i) * (p^S_{i+1} - p^S_i)

    This is a channel-wise nonparametric quantile (histogram) matching that
    maps the test-time distribution back to the source distribution observed
    during setup.

Tail calibration ("Average Sample Tails", paper Sec. 3.3 / 4.4) is applied at
stats-build time to the SOURCE p_0 / p_100 estimates (see build_aqr_stats.py).

Placement / hook parity
-----------------------
The paper defines pre-activations as normalization-layer outputs and places AQR
"after layer normalization" in the top half of a ViT.  For a controlled
comparison against QuAR (which corrects the inputs of the quantized Linear
layers), this module supports two placements, selected by --aqr_hook:

    linear_prehook (default):  apply AQR at the SAME pre-quantizer sites QuAR
                               uses (inputs of the quantized Linears; scope
                               block8_9_10_11_all).  Isolates the difference to
                               transform-only (quantile matching vs mean/std).
    ln_output:                 apply AQR directly on the LayerNorm outputs
                               (norm1/norm2), i.e. the paper-literal placement.

No backprop.  No quantizer state changes.  No extra forward passes.  AQR is
stateless per batch (episodic); reset() is a no-op beyond the base reset.
"""

import os
import torch
import torch.nn as nn

# Reuse QuAR's quant-activation detection / block indexing and the minimal
# no-adaptation base so the model construction path is identical to the local branch.
from tta_library.quar import _NoAdaptBase, _is_quant_act, _block_idx


# ─────────────────────────────────────────────────────────── percentile helper

def quantile_dim0(x: torch.Tensor, q: torch.Tensor,
                  chan_chunk: int = 1024) -> torch.Tensor:
    """Per-column linear-interpolated quantiles along dim 0.

    x: [M, C], q: [L] in [0, 1]  ->  [L, C].

    Implemented via sort (not torch.quantile) to avoid the ~2**24-element numel
    cap that torch.quantile enforces on the reduced dimension: a W6A6 batch of
    128 gives M ~ 25k tokens and C up to 3072, i.e. M*C ~ 77M > 16M, which
    torch.quantile rejects.  Channels are processed in chunks to bound memory.
    """
    M, C = x.shape
    L = q.shape[0]
    pos = q * (M - 1)
    lo = pos.floor().long()
    hi = pos.ceil().long()
    w = (pos - lo.to(pos.dtype)).unsqueeze(1)  # [L, 1]
    out = x.new_empty((L, C))
    for s in range(0, C, chan_chunk):
        e = min(s + chan_chunk, C)
        xs, _ = torch.sort(x[:, s:e], dim=0)   # [M, c]
        lo_v = xs[lo]                          # [L, c]
        hi_v = xs[hi]                          # [L, c]
        out[:, s:e] = lo_v * (1.0 - w) + hi_v * w
    return out


# ─────────────────────────────────────────────────────────── scope resolution

def _blocks_for_scope(scope: str):
    """Return the set of block indices for a scope, or None to mean 'all blocks'."""
    if scope in ("block8_9_10_11_all", "block8_9_10_11_valid"):
        return {8, 9, 10, 11}
    if scope in ("qba_valid", "last_2_blocks", "block10_11_all"):
        return {10, 11}
    if scope == "last_block_only":
        return {11}
    if scope in ("block6_to_11_all", "block6_to_11_valid"):
        return {6, 7, 8, 9, 10, 11}
    if scope in ("block7_to_11_all", "block7_to_11_valid"):
        return {7, 8, 9, 10, 11}
    if scope == "all":
        return None
    # default: match the canonical QuAR scope
    return {8, 9, 10, 11}


def _scope_excludes_fc2(scope: str) -> bool:
    return scope.endswith("_valid") or scope in ("qba_valid", "last_2_blocks",
                                                  "last_block_only")


# ─────────────────────────────────────────────────────────── AQR mixin

class AQRMixin:
    """Channel-wise piecewise-linear quantile recalibration of pre-activations.

    Place before the base class in the MRO (as with QuARLocalMixin).  Expects
    ``self.args`` and ``self.model`` (a proxy exposing ``.model`` = raw ViT and
    ``.logger``) to be set by the base ``__init__``.
    """

    def __init__(self, *args, **kwargs):
        self._aqr_hooks: list = []
        self._aqr_modules: list = []
        self._aqr_source_pct: dict = {}     # name -> Tensor[L, C] source percentiles
        self._aqr_levels: torch.Tensor = None
        self._aqr_active: bool = False
        self._aqr_enable: bool = False

        super().__init__(*args, **kwargs)

        self._aqr_enable = bool(getattr(self.args, "aqr_enable", False))
        self._aqr_scope = getattr(self.args, "aqr_scope", "block8_9_10_11_all")
        self._aqr_hook_point = getattr(self.args, "aqr_hook", "linear_prehook")
        self._aqr_alpha = float(getattr(self.args, "aqr_alpha", 1.0))
        self._aqr_eps = float(getattr(self.args, "aqr_eps", 1e-6))
        self._aqr_clip_tails = bool(getattr(self.args, "aqr_clip_tails", True))
        self._aqr_stats_file = getattr(self.args, "aqr_stats_file", "")
        # Tail estimation for the TARGET (per-batch) percentiles p^T_0 / p^T_100:
        #   raw        : batch min / max (pairs with --tail_calib standard source).
        #   avg_sample : Average Sample Tails (paper Sec 3.3) on the batch -- MUST
        #                pair with an avg_sample source stats file so source and
        #                target tails are on the SAME (m-point) scale. Applying
        #                avg_sample to only one side compresses the tails badly.
        self._aqr_target_tail = getattr(self.args, "aqr_target_tail", "raw")
        self._aqr_tail_m = int(getattr(self.args, "aqr_tail_m", 100))
        self._aqr_tail_reps = int(getattr(self.args, "aqr_tail_reps", 1000))

        if not self._aqr_enable:
            return

        self._load_aqr_source_stats()
        self._aqr_modules = self._aqr_select_modules()
        if not self._aqr_modules:
            self._aqr_enable = False
            self.model.logger.warning("AQR: no target modules found - disabling.")
            return

        for name, module in self._aqr_modules:
            if self._aqr_hook_point == "ln_output":
                h = module.register_forward_hook(self._make_aqr_post_hook(name))
            else:
                h = module.register_forward_pre_hook(self._make_aqr_pre_hook(name))
            self._aqr_hooks.append(h)

        n_matched = sum(1 for n, _ in self._aqr_modules if n in self._aqr_source_pct)
        self.model.logger.info(
            f"AQR enabled: scope={self._aqr_scope}, hook={self._aqr_hook_point}, "
            f"alpha={self._aqr_alpha}, eps={self._aqr_eps}, "
            f"clip_tails={self._aqr_clip_tails}, "
            f"L={0 if self._aqr_levels is None else self._aqr_levels.numel()}, "
            f"modules={len(self._aqr_modules)}, source_matched={n_matched}"
        )
        for name, _ in self._aqr_modules:
            tag = "OK" if name in self._aqr_source_pct else "NO_STATS"
            self.model.logger.info(f"  AQR module [{tag}]: {name}")

    # ─────────────────────────────────────────── source stats I/O

    def _load_aqr_source_stats(self):
        path = self._aqr_stats_file
        if not path or not os.path.isfile(path):
            self.model.logger.warning(
                f"AQR: stats file not found: '{path}'. Correction will be skipped."
            )
            return
        data = torch.load(path, map_location="cpu", weights_only=False)
        pct = data.get("percentiles", {})
        self._aqr_levels = data.get("levels", None)
        if self._aqr_levels is not None:
            self._aqr_levels = self._aqr_levels.float()
        for name, t in pct.items():
            self._aqr_source_pct[name] = t.float()
        cfg = data.get("config", {}) or {}
        built_hook = cfg.get("hook", None)
        self.model.logger.info(
            f"AQR source percentiles loaded from '{path}': "
            f"{len(self._aqr_source_pct)} layers, "
            f"built_hook={built_hook}, built_scope={cfg.get('scope', None)}."
        )
        if built_hook is not None and built_hook != self._aqr_hook_point:
            self.model.logger.warning(
                f"AQR: stats built with hook='{built_hook}' but running with "
                f"hook='{self._aqr_hook_point}'. Layer names may not match."
            )

    # ─────────────────────────────────────────── module selection

    def _aqr_select_modules(self):
        raw = self.model.model
        blocks = _blocks_for_scope(self._aqr_scope)
        modules = []

        if self._aqr_hook_point == "ln_output":
            # Paper-literal: hook LayerNorm outputs (norm1 / norm2) in scope blocks.
            for name, module in raw.named_modules():
                if not isinstance(module, nn.LayerNorm):
                    continue
                blk = _block_idx(name)
                if blk is None:              # skips the final net.norm (no block idx)
                    continue
                if blocks is not None and blk not in blocks:
                    continue
                modules.append((name, module))
            return modules

        # linear_prehook (default): same pre-quantizer sites as QuAR/the local branch.
        excl_fc2 = _scope_excludes_fc2(self._aqr_scope)
        for name, module in raw.named_modules():
            if not _is_quant_act(module):
                continue
            blk = _block_idx(name)
            if blk is None:
                continue
            if blocks is not None and blk not in blocks:
                continue
            if excl_fc2 and "fc2" in name:
                continue
            modules.append((name, module))
        return modules

    # ─────────────────────────────────────────── the transform

    def _aqr_avg_sample_tails(self, x_rs: torch.Tensor):
        """Average Sample Tails (paper Sec 3.3) on the current batch: sample
        `tail_m` points `tail_reps` times, average the per-draw min / max ->
        stable per-channel (p0, p100). x_rs: [M, C] -> (p0[C], p100[C])."""
        M, C = x_rs.shape
        m = min(self._aqr_tail_m, M)
        reps = self._aqr_tail_reps
        min_list, max_list = [], []
        done, chunk = 0, 100
        while done < reps:
            r = min(chunk, reps - done)
            idx = torch.randint(0, M, (r, m), device=x_rs.device)  # [r, m]
            samp = x_rs[idx]                                        # [r, m, C]
            min_list.append(samp.min(dim=1).values)                # [r, C]
            max_list.append(samp.max(dim=1).values)
            done += r
        p0 = torch.cat(min_list, 0).mean(0)
        p100 = torch.cat(max_list, 0).mean(0)
        return p0, p100

    def _aqr_apply(self, x: torch.Tensor, name: str):
        """Apply channel-wise piecewise-linear quantile matching. Returns None
        (no change) when no source percentiles exist for this layer."""
        ps = self._aqr_source_pct.get(name)
        if ps is None or self._aqr_levels is None:
            return None

        orig_shape = x.shape
        C = orig_shape[-1]
        x_rs = x.reshape(-1, C).float()          # [M, C]
        dev, dt = x_rs.device, x_rs.dtype
        ps = ps.to(dev, dt)                       # [L, C] source percentiles
        levels = self._aqr_levels.to(dev, dt)     # [L] in [0, 1]
        L = ps.shape[0]

        # Target percentiles from the current batch, per channel (detached).
        pt = quantile_dim0(x_rs, levels)          # [L, C]
        if self._aqr_target_tail == "avg_sample":
            # Match the source's Average-Sample-Tails scale (same m). Clamp to
            # preserve monotonicity of the percentile array.
            p0, p100 = self._aqr_avg_sample_tails(x_rs)
            pt[0] = torch.minimum(p0, pt[1])
            pt[-1] = torch.maximum(p100, pt[-2])

        # Per-channel row layout for batched searchsorted / gather.
        xT = x_rs.transpose(0, 1).contiguous()    # [C, M]
        ptT = pt.transpose(0, 1).contiguous()     # [C, L]
        psT = ps.transpose(0, 1).contiguous()     # [C, L]

        if self._aqr_clip_tails:
            # Values outside [p^T_0, p^T_100] cannot arise (percentiles come from
            # the same batch), but guard against numerical drift.
            xT = torch.maximum(torch.minimum(xT, ptT[:, -1:]), ptT[:, :1])

        # Interval index i s.t. p^T_i <= x < p^T_{i+1}, clamped to [0, L-2].
        idx = torch.searchsorted(ptT, xT, right=True) - 1
        idx = idx.clamp_(0, L - 2)

        pt_i = torch.gather(ptT, 1, idx)
        pt_ip1 = torch.gather(ptT, 1, idx + 1)
        ps_i = torch.gather(psT, 1, idx)
        ps_ip1 = torch.gather(psT, 1, idx + 1)

        d_t = pt_ip1 - pt_i
        d_s = ps_ip1 - ps_i
        # Delta^T_i == 0 (tied target percentiles, common with discrete/quantized
        # activations): the interval is degenerate -> map to the source level p^S_i.
        frac = torch.where(d_t > self._aqr_eps,
                           (xT - pt_i) / d_t.clamp_min(self._aqr_eps),
                           torch.zeros_like(xT))
        out = ps_i + frac * d_s                   # [C, M]

        out = out.transpose(0, 1).reshape(orig_shape).to(x.dtype)
        if self._aqr_alpha < 1.0:
            out = (1.0 - self._aqr_alpha) * x + self._aqr_alpha * out
        return out

    # ─────────────────────────────────────────── hooks

    def _make_aqr_pre_hook(self, name):
        def hook(module, inputs):
            if not self._aqr_active or not inputs:
                return None
            x = inputs[0]
            if not torch.is_tensor(x):
                return None
            with torch.no_grad():
                x_out = self._aqr_apply(x.detach(), name)
            if x_out is None:
                return None
            if x.requires_grad:
                x_out = x_out.detach().requires_grad_(True)
            return (x_out,) + inputs[1:]
        return hook

    def _make_aqr_post_hook(self, name):
        def hook(module, inputs, output):
            if not self._aqr_active or not torch.is_tensor(output):
                return None
            with torch.no_grad():
                x_out = self._aqr_apply(output.detach(), name)
            if x_out is None:
                return None
            if output.requires_grad:
                x_out = x_out.detach().requires_grad_(True)
            return x_out
        return hook

    # ─────────────────────────────────────────── forward / reset

    def forward(self, x):
        if not self._aqr_enable:
            return super().forward(x)
        self._aqr_active = True
        try:
            return super().forward(x)
        finally:
            self._aqr_active = False

    def reset(self):
        super().reset()
        # AQR is episodic and stateless per batch: nothing to reset.


class NoAdaptAQRWrapper(AQRMixin, _NoAdaptBase):
    """Applies AQR quantile recalibration to a frozen model (no TTA).

    Use with ``--algorithm no_adapt --aqr_enable``.
    Construct via ``build_no_adapt_aqr_wrapper(args, net, logger)``.
    """
    pass


def build_no_adapt_aqr_wrapper(args, net, _logger):
    """Create a NoAdaptAQRWrapper around a raw quantized net.

    Mirrors build_no_adapt_qac_wrapper: builds a lightweight proxy so AQRMixin
    can reach ``self.model.model.named_modules()`` and ``self.model.logger``.
    """
    import types as _types
    raw_model = getattr(net, "model", net)
    proxy = _types.SimpleNamespace(model=raw_model, logger=_logger)
    return NoAdaptAQRWrapper(args, proxy)
