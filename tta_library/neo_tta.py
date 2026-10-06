"""Official NEO: no-optimization TTA through latent re-centering.

Ports the episodic NEO variant from https://github.com/awesomealex1/NEO.

Mechanism:
  - Hooks before net.head (the final linear classifier).
  - Maintains a cumulative (sample-count-weighted) mean of the CLS features
    seen so far within the current corruption domain.
  - Subtracts that running mean before each forward pass through the head.
  - Uses ONLY target statistics — no source statistics required.

State management:
  - reset() zeros the running mean and sample count.
  - Call reset() at the start of each new corruption domain (episodic).

Reference: arXiv 2510.05635
"""

import torch
import torch.nn as nn


class NEOWrapper(nn.Module):
    """Wraps a quantized ViT and applies NEO re-centering at the head input.

    Args:
        net    : raw quantized ViT (nn.Module), must have .head (nn.Linear)
        logger : optional logger
    """

    def __init__(self, net, logger=None, alpha=None):
        super().__init__()
        self.net = net
        self.logger = logger
        self.alpha = alpha  # None -> base NEO (cumulative 1/i); float -> NEO-Continual (EMA)
        self.sample_count = 0
        self._global_mean = None  # lazy-init on first forward
        self._hooks = []
        self._register_hook()

        d = net.head.in_features
        mode = (f"NEO-Continual EMA alpha={alpha}" if alpha is not None
                else "cumulative (1/i)")
        msg = f"NEO: {mode} mean subtraction at head input, D={d}"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # ─── hook ────────────────────────────────────────────────────────────────

    def _register_hook(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        def _pre_hook(module, inputs):
            x = inputs[0]
            device, dtype = x.device, x.dtype

            if self._global_mean is None:
                self._global_mean = torch.zeros(x.shape[-1], device=device, dtype=dtype)
            else:
                self._global_mean = self._global_mean.to(device=device, dtype=dtype)

            bm = x.mean(0)
            if self.alpha is None:
                # base NEO: cumulative (1/i), weighting all samples equally
                b = x.size(0)
                self.sample_count += b
                seen_w = (self.sample_count - b) / self.sample_count
                new_w  = b / self.sample_count
                self._global_mean = seen_w * self._global_mean + new_w * bm
            else:
                # NEO-Continual (official neo_layers.NEO_Continual): EMA, default alpha=0.1.
                # First batch initialised to its exact mean (unbiased, matches base NEO's first batch).
                if self.sample_count == 0:
                    self._global_mean = bm
                    self.sample_count += x.size(0)
                else:
                    self._global_mean = (1.0 - self.alpha) * self._global_mean + self.alpha * bm

            return (x - self._global_mean,) + inputs[1:]

        h = self.net.head.register_forward_pre_hook(_pre_hook)
        self._hooks.append(h)

    # ─── forward / reset ─────────────────────────────────────────────────────

    def forward(self, x):
        return self.net(x)

    def reset(self):
        self.sample_count = 0
        self._global_mean = None

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─── builder ─────────────────────────────────────────────────────────────────

def build_neo_wrapper(args, net, logger, local_wrapper=None):
    """Create a NEOWrapper, optionally combined with a local-branch wrapper.

    If local_wrapper is provided, local-branch hooks and NEO hook both fire during
    local_wrapper(x)'s forward pass (both are attached to raw_net).
    The returned adapt_model's reset() clears both NEO and the local branch state.
    """
    raw_net = getattr(net, "model", None)
    raw_net = getattr(raw_net, "model", raw_net) if raw_net is not None else net

    neo = NEOWrapper(net=raw_net, logger=logger, alpha=getattr(args, "neo_ema_alpha", None))

    if local_wrapper is not None:
        class _Combined(nn.Module):
            def __init__(self, local_w, neo_w):
                super().__init__()
                self._qac = local_w
                self._neo = neo_w

            def forward(self, x):
                return self._qac(x)

            def reset(self):
                self._neo.reset()
                if hasattr(self._qac, "reset"):
                    self._qac.reset()

        return _Combined(local_wrapper, neo)

    return neo
