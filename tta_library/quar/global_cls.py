"""FOA-style CLS-token activation shifting for ViT-B quantized models.

Intercepts the CLS token at the final-LayerNorm boundary and shifts its
per-channel distribution toward source statistics, without any backprop,
prompt optimization, or weight updates.

Hook positions:
  pre_ln  — CLS token BEFORE net.norm  (net.norm forward pre-hook,  inputs[0][:, 0])
  post_ln — CLS token AFTER  net.norm  (net.head  forward pre-hook,  inputs[0])

Shift modes:
  mean_only  — h + (mu_s - mu_t)            [per-batch mu_t by default]
  mean_std   — (h - mu_t) / (sig_t + eps) * sig_s + mu_s

target_ema_alpha (default 0.0 = per-batch):
  > 0  →  mu_t tracked via EMA across batches within one corruption domain
           mu_t ← (1-alpha)*mu_t_prev + alpha*batch_mean  (e.g. alpha=0.1)
  = 0  →  mu_t = h.mean(dim=0) fresh each batch (original behaviour)

alpha blending (default 1.0):
  h_out = (1 - alpha) * h + alpha * h_shifted
"""

import torch
import torch.nn as nn


class CLSShiftWrapper(nn.Module):
    """Wraps a quantized ViT and shifts the CLS token per source statistics.

    Args:
        net        : raw quantized ViT (nn.Module)
        source_stats: dict from build_cls_stats.py — has 'pre_ln' and 'post_ln' keys
        position   : 'pre_ln' or 'post_ln'
        mode       : 'mean_only' or 'mean_std'
        alpha      : blending weight (1.0 = full correction)
        eps        : std denominator clamp
        clip_ratio : max sig_ratio magnitude (prevents explosion)
        logger     : optional logger
    """

    def __init__(
        self,
        net,
        source_stats,
        position="post_ln",
        mode="mean_std",
        alpha=1.0,
        eps=1e-6,
        clip_ratio=3.0,
        target_ema_alpha=0.0,
        cumulative=False,
        stream_lag=False,
        estimator="current",
        logger=None,
    ):
        super().__init__()
        self.net = net
        self.position = position
        self.mode = mode
        self.alpha = alpha
        self.eps = eps
        self.clip_ratio = clip_ratio
        self.target_ema_alpha = target_ema_alpha
        self.stream_lag = stream_lag  # correct with the pre-update state (exclude the current batch)
        self.cumulative = cumulative  # source-initialized running mean (alpha=1/(bc+1)) instead of fixed EMA
        self.estimator = estimator    # 'current' (unchanged), 'ema' (alpha=0.1), or 'welford' (pooled var)
        self.logger = logger
        self._mu_t_ema = None      # EMA state; None until first batch
        self._sig_t_ema = None
        self._cls_bc = 1           # batch count (source = one pseudo-batch) for cumulative
        # welford estimator state (per-channel); None until first batch
        self._w_count = None
        self._w_mean = None
        self._w_M2 = None

        pos_key = position  # 'pre_ln' or 'post_ln'
        if pos_key not in source_stats:
            raise KeyError(f"source_stats missing key '{pos_key}'; available: {list(source_stats.keys())}")

        self.register_buffer("mu_s",  source_stats[pos_key]["mean"].float())
        self.register_buffer("sig_s", source_stats[pos_key]["std"].float())

        self._hooks = []
        self._register_hooks()

        n_samples = source_stats.get("n_samples", "?")
        storage_bytes = (self.mu_s.numel() + self.sig_s.numel()) * 4
        ema_str = f", target_ema_alpha={target_ema_alpha}" if target_ema_alpha > 0 else ""
        if logger:
            logger.info(
                f"CLSShift: position={position}, mode={mode}, alpha={alpha}{ema_str}, "
                f"D={self.mu_s.shape[0]}, n_src={n_samples}, "
                f"storage={storage_bytes}B"
            )
        else:
            print(f"CLSShift: position={position}, mode={mode}, alpha={alpha}{ema_str}, "
                  f"D={self.mu_s.shape[0]}, n_src={n_samples}, storage={storage_bytes}B")

    # ─── hook registration ────────────────────────────────────────────────────

    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        if self.position == "pre_ln":
            # Hook BEFORE net.norm: inputs[0] is (B, N+1, D); modify CLS at [:, 0]
            h = self.net.norm.register_forward_pre_hook(self._pre_ln_hook)
        else:
            # Hook BEFORE net.head: inputs[0] is (B, D) — already pooled CLS
            h = self.net.head.register_forward_pre_hook(self._post_ln_hook)
        self._hooks.append(h)

    def _pre_ln_hook(self, module, inputs):
        x = inputs[0]
        if x.dim() == 2:
            # ZOA calls norm(x[:, 0]) internally for feature extraction;
            # don't shift here — would corrupt ZOA's adaptation signal.
            return inputs
        # Normal path: full sequence (B, N+1, D)
        cls = x[:, 0]   # (B, D)
        cls_shifted = self._shift(cls)
        x_mod = x.clone()
        x_mod[:, 0] = cls_shifted
        return (x_mod,) + inputs[1:]

    def _post_ln_hook(self, module, inputs):
        cls = inputs[0]   # (B, D)
        cls_shifted = self._shift(cls)
        return (cls_shifted,) + inputs[1:]

    # ─── shift computation ────────────────────────────────────────────────────

    def _shift(self, h):
        """Apply per-channel mean/std shift to CLS feature h (B, D)."""
        mu_s  = self.mu_s.to(device=h.device,  dtype=h.dtype)
        sig_s = self.sig_s.to(device=h.device, dtype=h.dtype)

        # Estimator dispatch (additive). 'current' falls through to the original path below.
        if self.estimator == "welford":
            return self._shift_welford(h, mu_s, sig_s)
        if self.estimator == "ema":
            return self._shift_ema(h, mu_s, sig_s)

        batch_mu_t = h.mean(dim=0)   # (D,)

        # Skip EMA updates when the owning adapter freezes recalibration stats (e.g. during
        # ZOA SPSA perturbation forwards); still use the current EMA state for the shift.
        _frozen = getattr(getattr(self, "_freeze_owner", None), "_recal_freeze", False)

        if self.cumulative:
            # Source-initialized running mean: alpha_n = 1/(bc+1) (source = one pseudo-batch),
            # a forgetting-free cumulative estimate (like NEO) instead of a fixed-rate EMA.
            a = 1.0 / (self._cls_bc + 1.0)
            if self._mu_t_ema is None:
                self._mu_t_ema = mu_s.detach().clone()
            _mu_prev = self._mu_t_ema.to(device=h.device, dtype=h.dtype)
            if not _frozen:
                self._mu_t_ema = (1.0 - a) * _mu_prev + a * batch_mu_t.detach()
            mu_t = _mu_prev if self.stream_lag else self._mu_t_ema
        elif self.target_ema_alpha > 0:
            # EMA target center (FOA act. style)
            if self._mu_t_ema is None:
                self._mu_t_ema = batch_mu_t.detach().clone()
            elif not _frozen:
                a = self.target_ema_alpha
                self._mu_t_ema = (
                    (1.0 - a) * self._mu_t_ema.to(device=h.device, dtype=h.dtype)
                    + a * batch_mu_t.detach()
                )
            mu_t = self._mu_t_ema
        else:
            mu_t = batch_mu_t

        if self.mode == "mean_std":
            batch_sig_t = h.std(dim=0, unbiased=False)   # unbiased=False: 0 (not NaN) at bs=1
            if self.cumulative:
                a = 1.0 / (self._cls_bc + 1.0)
                if self._sig_t_ema is None:
                    self._sig_t_ema = sig_s.detach().clone()
                _sig_prev = self._sig_t_ema.to(device=h.device, dtype=h.dtype)
                if not _frozen:
                    self._sig_t_ema = (1.0 - a) * _sig_prev + a * batch_sig_t.detach()
                sig_t = _sig_prev if self.stream_lag else self._sig_t_ema
            elif self.target_ema_alpha > 0:
                if self._sig_t_ema is None:
                    self._sig_t_ema = batch_sig_t.detach().clone()
                elif not _frozen:
                    a = self.target_ema_alpha
                    self._sig_t_ema = (
                        (1.0 - a) * self._sig_t_ema.to(device=h.device, dtype=h.dtype)
                        + a * batch_sig_t.detach()
                    )
                sig_t = self._sig_t_ema
            else:
                sig_t = batch_sig_t
            if self.cumulative and not _frozen:
                self._cls_bc += 1
            sig_ratio = (sig_s / (sig_t + self.eps)).clamp(
                1.0 / self.clip_ratio, self.clip_ratio
            )
            h_shifted = (h - mu_t) * sig_ratio + mu_s
        else:  # mean_only
            h_shifted = h + (mu_s - mu_t)

        return (1.0 - self.alpha) * h + self.alpha * h_shifted

    def _shift_welford(self, h, mu_s, sig_s):
        """Exact streaming POOLED variance over the B CLS tokens, accumulated across
        batches within a domain (Welford parallel/batch merge). Source-initialized as
        one pseudo-batch: count0 = this batch's B, mean0 = mu_s, M2_0 = sig_s^2 * B.
        Unlike the running-average-of-per-batch-stds baseline, this does NOT collapse
        to 0 at bs=1 (it captures the between-batch variance of the CLS token)."""
        _frozen = getattr(getattr(self, "_freeze_owner", None), "_recal_freeze", False)
        n_new = h.shape[0]
        b_mean = h.mean(dim=0).detach()
        b_M2 = h.detach().var(dim=0, unbiased=False) * float(n_new)  # 0 at bs=1
        if self._w_count is None:
            self._w_count = float(n_new)
            self._w_mean = mu_s.detach().clone()
            self._w_M2 = (sig_s.detach() ** 2 * float(n_new)).clone()
        mean_prev = self._w_mean.to(device=h.device, dtype=h.dtype)
        M2_prev = self._w_M2.to(device=h.device, dtype=h.dtype)
        count_prev = self._w_count
        if not _frozen:
            delta = b_mean - mean_prev
            tot = count_prev + n_new
            self._w_mean = mean_prev + delta * (n_new / tot)
            self._w_M2 = M2_prev + b_M2 + (delta ** 2) * (count_prev * n_new / tot)
            self._w_count = tot
            mu_t = self._w_mean
            var_t = self._w_M2 / tot
        else:
            mu_t = mean_prev
            var_t = M2_prev / count_prev
        if self.mode == "mean_std":
            sig_t = var_t.clamp(min=0.0).sqrt()
            sig_ratio = (sig_s / (sig_t + self.eps)).clamp(
                1.0 / self.clip_ratio, self.clip_ratio
            )
            h_shifted = (h - mu_t) * sig_ratio + mu_s
        else:  # mean_only
            h_shifted = h + (mu_s - mu_t)
        return (1.0 - self.alpha) * h + self.alpha * h_shifted

    def _shift_ema(self, h, mu_s, sig_s):
        """Fixed-rate EMA (alpha=0.1) of per-batch target moments (mu_t, sig_t)."""
        a = 0.1
        _frozen = getattr(getattr(self, "_freeze_owner", None), "_recal_freeze", False)
        batch_mu_t = h.mean(dim=0)
        if self._mu_t_ema is None:
            self._mu_t_ema = batch_mu_t.detach().clone()
        elif not _frozen:
            self._mu_t_ema = (
                (1.0 - a) * self._mu_t_ema.to(device=h.device, dtype=h.dtype)
                + a * batch_mu_t.detach()
            )
        mu_t = self._mu_t_ema
        if self.mode == "mean_std":
            batch_sig_t = h.std(dim=0, unbiased=False)
            if self._sig_t_ema is None:
                self._sig_t_ema = batch_sig_t.detach().clone()
            elif not _frozen:
                self._sig_t_ema = (
                    (1.0 - a) * self._sig_t_ema.to(device=h.device, dtype=h.dtype)
                    + a * batch_sig_t.detach()
                )
            sig_t = self._sig_t_ema
            sig_ratio = (sig_s / (sig_t + self.eps)).clamp(
                1.0 / self.clip_ratio, self.clip_ratio
            )
            h_shifted = (h - mu_t) * sig_ratio + mu_s
        else:  # mean_only
            h_shifted = h + (mu_s - mu_t)
        return (1.0 - self.alpha) * h + self.alpha * h_shifted

    # ─── forward / reset ─────────────────────────────────────────────────────

    def forward(self, x):
        return self.net(x)

    def reset(self):
        self._mu_t_ema = None
        self._sig_t_ema = None
        self._cls_bc = 1
        self._w_count = None
        self._w_mean = None
        self._w_M2 = None

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─── builder ─────────────────────────────────────────────────────────────────

def build_cls_shift_wrapper(args, net, logger, local_wrapper=None):
    """Create a CLSShiftWrapper.

    If local_wrapper is provided, wraps it (forward calls local_wrapper(x) which
    already has local-branch hooks active on the underlying net).  Otherwise wraps net
    directly.

    Returns a CLSShiftWrapper whose .forward() returns logits.
    """
    stats_path = args.quar_global_stats_file
    if not stats_path or not __import__("os").path.isfile(stats_path):
        raise FileNotFoundError(f"CLS stats file not found: {stats_path}")

    source_stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    # The underlying quantized ViT (for hook registration) is always net
    raw_net = getattr(net, "model", None)
    raw_net = getattr(raw_net, "model", raw_net) if raw_net is not None else net

    wrapper = CLSShiftWrapper(
        net=raw_net,
        source_stats=source_stats,
        position=getattr(args, "quar_global_position", "post_ln"),
        mode=getattr(args, "quar_global_mode", "mean_std"),
        alpha=getattr(args, "quar_global_alpha", 1.0),
        eps=getattr(args, "quar_global_eps", 1e-6),
        clip_ratio=getattr(args, "quar_global_clip_ratio", 3.0),
        target_ema_alpha=getattr(args, "quar_global_target_ema_alpha", 0.0),
        stream_lag=bool(getattr(args, "quar_global_stream_lag", False)),
        cumulative=getattr(args, "recal_cumulative", False),
        estimator=getattr(args, "quar_global_estimator", "current"),
        logger=logger,
    )

    if local_wrapper is not None:
        # Combined: local-branch hooks are on raw_net; CLS shift hook is also on raw_net.
        # We need a thin outer wrapper that calls local_wrapper(x) so the local branch fires first,
        # then CLS shift fires via the hook on raw_net during local_wrapper's forward.
        class _Combined(nn.Module):
            def __init__(self, local_w, cls_w):
                super().__init__()
                self._qac = local_w
                self._cls = cls_w  # holds hook handles

            def forward(self, x):
                return self._qac(x)

            def reset(self):
                self._cls.reset()  # clear EMA state if any
                if hasattr(self._qac, "reset"):
                    self._qac.reset()
        return _Combined(local_wrapper, wrapper)

    return wrapper
