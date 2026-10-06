"""QuAR: Quantizer-Aligned Recalibration, the local branch.

Recalibrates activations at the input to a frozen quantizer, mapping the running per-channel
statistics of the test stream back toward the source calibration. The global CLS branch lives in
cls_shift.py; together they are the method the paper reports.

NoAdaptQACWrapper is what the paper runs, over a plain frozen network; it does not depend on any
baseline adapter.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_quant_act(module):
    # PTQ4ViT-style
    if torch.is_tensor(getattr(module, "a_interval", None)):
        if getattr(module, "a_bit", 32) < 32 and hasattr(module, "a_qmax"):
            return True
    # AdaLog-style
    qz = getattr(module, "a_quantizer", None)
    if qz is not None and getattr(qz, "inited", False):
        if getattr(qz, "n_bits", 32) < 32:
            return True
    return False

def _block_idx(name):
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "blocks":
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


class QuARLocalMixin:
    """Pre-quantizer activation distribution correction.

    For each selected layer, intercepts the activation x before it enters the
    quantizer and corrects its distribution toward the source:

        mean_std mode:
            x_corr = (x - mu_t) / (sigma_t + eps) * sigma_s + mu_s
        std_only mode:
            x_corr = x * clamp(sigma_s / (sigma_t + eps), 1/clip_ratio, clip_ratio)
        blended:
            x_out = (1 - alpha) * x + alpha * x_corr

    Source stats (per-channel mu_s, sigma_s) are loaded from a pre-built file.
    Target stats (mu_t, sigma_t) are estimated from the current batch (detached).
    Optional EMA smoothing of target stats across batches within a corruption.

    No backprop. No quantizer state changes. No extra forward passes.
    forward_loss_calls unchanged. Place before ZOAOfficial in MRO.
    """

    def __init__(self, *args, **kwargs):
        self._quar_hooks: list = []
        self._quar_modules: list = []
        self._quar_source_stats: dict = {}
        self._quar_ema_mu: dict = {}
        self._quar_ema_sig: dict = {}
        self._quar_stream_m: dict = {}    # streaming EMA first moment (source-initialized)
        self._quar_stream_q: dict = {}    # streaming EMA second moment E[y^2]
        self._quar_stream_n: dict = {}    # streaming EMA sample count (images seen)
        self._quar_stream_bc: dict = {}   # streaming batch count (for cumulative alpha=1/(bc+1))
        self._quar_welford_mean: dict = {}   # welford estimator: per-channel running mean
        self._quar_welford_M2: dict = {}     # welford estimator: per-channel running sum of squared deviations
        self._quar_welford_count: dict = {}  # welford estimator: per-channel running token count
        self._quar_diag_buf: dict = {}    # name -> list of per-batch dicts
        self._quar_active: bool = False
        self._quar_enable: bool = False
        self._quar_interval_backup: dict = {}   # name -> saved a_interval (folded_quantizer)
        self._quar_zero_point_backup: dict = {} # name -> saved qzero_z (scale_offset_coupled)
        self._quar_adalog_backup: dict = {}     # name -> saved {scale, shift} (scale_shift_coupled)
        self._quar_qaffine_backup: dict = {}    # name -> True (scale_offset_decoupled)
        self._quar_post_backup: dict = {}       # name -> True (output_only, PTQ4ViT)
        self._quar_adalog_post_backup: dict = {} # name -> (a_s, b_s) for AdaLog output_only/decoupled

        super().__init__(*args, **kwargs)

        self._quar_enable = bool(getattr(self.args, "quar_local_enable", False))
        self._quar_scope = getattr(self.args, "quar_local_scope", "block8_9_10_11_all")
        self._quar_chunk = int(getattr(self.args, "quar_local_chunk", 0))
        self._quar_alpha = float(getattr(self.args, "quar_local_lambda", 0.5))
        _ap = getattr(self.args, "quar_local_alpha_profile", "") or ""
        self._quar_alpha_profile = {}
        if _ap:
            for _tok in _ap.split(","):
                _b, _v = _tok.split(":")
                self._quar_alpha_profile[int(_b)] = float(_v)
        self._quar_stats_mode = getattr(self.args, "quar_local_stats", "mean_std")
        self._quar_eps = float(getattr(self.args, "quar_local_eps", 1e-6))
        self._quar_clip_ratio = float(getattr(self.args, "quar_local_clip_ratio", 3.0))
        self._quar_use_ema = bool(getattr(self.args, "quar_local_use_ema", False))
        self._quar_ema_momentum = float(getattr(self.args, "quar_local_ema_momentum", 0.1))
        self._quar_stream_ema  = bool(getattr(self.args, "quar_local_stream_ema", False))
        self._quar_stream_alpha = float(getattr(self.args, "quar_local_stream_alpha", 0.05))
        self._recal_cumulative = bool(getattr(self.args, "recal_cumulative", False))
        self._quar_stream_nwarm = int(getattr(self.args, "quar_local_stream_nwarm", 64))
        self._quar_stream_lag   = bool(getattr(self.args, "quar_local_stream_lag", False))
        self._quar_estimator = getattr(self.args, "quar_local_estimator", "current")
        self._quar_stats_file = getattr(self.args, "quar_local_stats_file", "")
        self._quar_fp_mode = bool(getattr(self.args, "quar_local_fp_mode", False))
        self._quar_exclude_layers: set = set(
            getattr(self.args, "quar_local_exclude_layers", None) or []
        )
        self._quar_mode: str = getattr(self.args, "quar_local_mode", "explicit")
        self._quar_fold_type: str = getattr(self.args, "quar_local_fold_type", "scale_only")
        self._quar_detail_logged: bool = False

        if not self._quar_enable:
            return

        self._quar_modules = self._quar_select_modules()
        if not self._quar_modules:
            self._quar_enable = False
            self.model.logger.warning("the local branch: no target modules found — disabling.")
            return

        self._load_qac_source_stats()

        for name, module in self._quar_modules:
            if self._quar_scope.startswith("resid_"):
                h = module.register_forward_hook(self._make_qac_output_hook(name))
            else:
                h = module.register_forward_pre_hook(self._make_qac_hook(name))
            self._quar_hooks.append(h)
            # folded_quantizer mode: register post-hook to restore patched quantizer state
            if self._quar_mode == "folded_quantizer":
                _has_interval = torch.is_tensor(getattr(module, "a_interval", None))
                _aq = getattr(module, "a_quantizer", None)
                _has_adalog = (_aq is not None
                               and torch.is_tensor(getattr(_aq, "scale", None)))
                if _has_interval or _has_adalog:
                    h2 = module.register_forward_hook(self._make_qac_restore_hook(name))
                    self._quar_hooks.append(h2)
            self._quar_diag_buf[name] = []

        n_matched = sum(1 for n in self._quar_source_stats if n in
                        {nm for nm, _ in self._quar_modules})
        _ema_str = (f"stream_ema(alpha={self._quar_stream_alpha},nwarm={self._quar_stream_nwarm},"
                    f"{'lag' if self._quar_stream_lag else 'incl'})"
                    if self._quar_stream_ema else
                    (f"ema(m={self._quar_ema_momentum})" if self._quar_use_ema else "no_ema"))
        self.model.logger.info(
            f"the local branch enabled: scope={self._quar_scope}, mode={self._quar_mode}, "
            f"alpha={self._quar_alpha}, stats={self._quar_stats_mode}, "
            f"eps={self._quar_eps}, clip_ratio={self._quar_clip_ratio}, "
            f"{_ema_str}, modules={len(self._quar_modules)}, "
            f"source_matched={n_matched}"
        )
        for name, _ in self._quar_modules:
            matched = "OK" if name in self._quar_source_stats else "NO_STATS"
            self.model.logger.info(f"  the local branch module [{matched}]: {name}")

    # ────────────────────────────────────────────── module selection

    # Target layer names for FP control (same topology as qba_valid, selected by name)
    _QAC_FP_TARGETS = {
        "qba_valid":      {f"blocks.{b}.{s}"
                           for b in (10, 11)
                           for s in ("attn.qkv", "attn.proj", "mlp.fc1")},
        "last_2_blocks":  {f"blocks.{b}.{s}"
                           for b in (10, 11)
                           for s in ("attn.qkv", "attn.proj", "mlp.fc1")},
        "last_block_only":{f"blocks.11.{s}"
                           for s in ("attn.qkv", "attn.proj", "mlp.fc1")},
        "attn_only":      {f"blocks.{b}.{s}"
                           for b in (10, 11)
                           for s in ("attn.qkv", "attn.proj")},
        "mlp_only":       {f"blocks.{b}.mlp.fc1" for b in (10, 11)},
    }

    def _quar_select_modules(self):
        fp_mode = getattr(self, "_quar_fp_mode", False)
        scope   = self._quar_scope

        if fp_mode:
            # FP model: no quantizer attributes; select by layer name
            targets = self._QAC_FP_TARGETS.get(scope, set())
            modules = []
            for name, module in self.model.model.named_modules():
                if name in targets and name not in self._quar_exclude_layers:
                    modules.append((name, module))
            return modules

        modules = []
        # Placement control: the same recalibration applied to the OUTPUT of whole transformer
        # blocks, i.e. the residual stream, which no activation quantizer reads. Scope
        # 'resid_block8_9_10_11' selects blocks.8..11; __init__ hooks their outputs instead of
        # the inputs of their quantized linears.
        if scope.startswith("resid_block"):
            want = {f"blocks.{b}" for b in scope[len("resid_block"):].split("_") if b.isdigit()}
            return [(name, module) for name, module in self.model.model.named_modules()
                    if name in want and name not in self._quar_exclude_layers]
        for name, module in self.model.model.named_modules():
            if not _is_quant_act(module):
                continue
            blk = _block_idx(name)
            if blk is None:
                continue
            if scope in ("last_2_blocks", "qba_valid"):
                if blk not in (10, 11):
                    continue
                if "fc2" in name:
                    continue
            elif scope == "last_block_only":
                if blk != 11:
                    continue
                if "fc2" in name:
                    continue
            elif scope == "attn_only":
                if blk not in (10, 11):
                    continue
                if "fc1" in name or "fc2" in name:
                    continue
            elif scope == "mlp_only":
                if blk not in (10, 11):
                    continue
                if "attn" in name:
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block11_valid":
                if blk != 11:
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block11_all":
                if blk != 11:
                    continue
            elif scope == "block10_valid":
                if blk != 10:
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block10_all":
                if blk != 10:
                    continue
            elif scope == "block10_11_all":
                if blk not in (10, 11):
                    continue
            elif scope == "block9_10_11_valid":
                if blk not in (9, 10, 11):
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block9_10_11_all":
                if blk not in (9, 10, 11):
                    continue
            elif scope == "block8_9_10_11_valid":
                if blk not in (8, 9, 10, 11):
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block8_9_10_11_all":
                if blk not in (8, 9, 10, 11):
                    continue
            elif scope == "block8_9_10_11_attn":
                # blocks 8-11, attention modules only (attn.qkv, attn.proj); exclude MLP
                if blk not in (8, 9, 10, 11):
                    continue
                if "fc1" in name or "fc2" in name:
                    continue
            elif scope == "block8_9_10_11_mlp1":
                # blocks 8-11, mlp.fc1 only; exclude attention and post-GELU fc2
                if blk not in (8, 9, 10, 11):
                    continue
                if "attn" in name:
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block7_to_11_valid":
                if blk not in range(7, 12):
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block6_to_11_valid":
                if blk not in range(6, 12):
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block5_to_11_valid":
                if blk not in range(5, 12):
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block4_to_11_valid":
                if blk not in range(4, 12):
                    continue
                if "fc2" in name:
                    continue
            elif scope == "block7_to_11_all":
                if blk not in range(7, 12):
                    continue
            elif scope == "block6_to_11_all":
                if blk not in range(6, 12):
                    continue
            elif scope == "block5_to_11_all":
                if blk not in range(5, 12):
                    continue
            elif scope == "block0_to_3_all":
                if blk not in range(0, 4):
                    continue
            elif scope == "block4_to_7_all":
                if blk not in range(4, 8):
                    continue
            elif scope == "all_blocks_valid_no_fc2":
                if "fc2" in name:
                    continue
            elif scope == "all":
                pass
            elif scope in ("source_reliable", "random_same_k"):
                pass  # collect all; filter below
            else:
                continue
            modules.append((name, module))
        if self._quar_exclude_layers:
            modules = [(n, m) for n, m in modules if n not in self._quar_exclude_layers]

        if scope == "source_reliable":
            # Keep layers whose source stats have finite values and std.mean() > 1e-3.
            stats_file = self._quar_stats_file
            reliable_names: set = set()
            if stats_file and os.path.isfile(stats_file):
                data = torch.load(stats_file, map_location="cpu", weights_only=False)
                std_dict = data.get("std", {})
                mean_dict = data.get("mean", {})
                for lname, std_val in std_dict.items():
                    mean_val = mean_dict.get(lname, std_val)
                    if (torch.isfinite(std_val).all() and
                            torch.isfinite(mean_val).all() and
                            float(std_val.mean()) > 1e-3):
                        reliable_names.add(lname)
            self.model.logger.info(
                f"local-branch source_reliable: {len(reliable_names)} reliable layers from stats file."
            )
            modules = [(n, m) for n, m in modules if n in reliable_names]

        elif scope == "random_same_k":
            # Build qba_valid reference set to get k.
            qba_names = [n for n, _ in modules
                         if _block_idx(n) in (10, 11) and "fc2" not in n]
            k = len(qba_names)
            seed = int(getattr(self.args, "quar_local_random_scope_seed", 0))
            rng = _random.Random(seed)
            all_names = [n for n, _ in modules]
            chosen = set(rng.sample(all_names, min(k, len(all_names))))
            self.model.logger.info(
                f"the local branch random_same_k: k={k}, seed={seed}, chosen={len(chosen)} layers."
            )
            modules = [(n, m) for n, m in modules if n in chosen]

        return modules

    # ────────────────────────────────────────────── source stats I/O

    def _load_qac_source_stats(self):
        path = self._quar_stats_file
        if not path or not os.path.isfile(path):
            self.model.logger.warning(
                f"the local branch: stats file not found: '{path}'. Correction will be skipped."
            )
            return
        data = torch.load(path, map_location="cpu", weights_only=False)
        mu_dict  = data.get("mean", {})
        sig_dict = data.get("std", {})
        for name in mu_dict:
            self._quar_source_stats[name] = {
                "mean": mu_dict[name].float(),
                "std":  sig_dict.get(name, torch.ones_like(mu_dict[name])).float(),
            }
        self.model.logger.info(
            f"local-branch source stats loaded from '{path}': {len(self._quar_source_stats)} layers."
        )

    # ────────────────────────────────────────────── correction hook

    def _quar_compute_a_eff_for_interval(
        self, x: torch.Tensor, src: dict, module
    ) -> torch.Tensor:
        """Per-group scale factor for PTQ4ViT a_interval folding.

        Returns tensor of shape (n_a, 1) matching module.a_interval.
        a_eff = (1-alpha) + alpha * clamp(sigma_s / sigma_t)
        """
        dev = x.device
        sig_s = src["std"].to(dev)
        x_rs = x.reshape(-1, x.shape[-1])  # (B*N, C)
        sig_t = x_rs.std(dim=0)             # (C,)
        sig_ratio = (sig_s / (sig_t + self._quar_eps)).clamp(
            1.0 / self._quar_clip_ratio, self._quar_clip_ratio
        )
        a_eff = (1.0 - self._quar_alpha) + self._quar_alpha * sig_ratio  # (C,)
        # Average within each quantizer group to match (n_a, 1) interval shape
        n_a = module.a_interval.shape[0]
        C = x.shape[-1]
        group_size = max(C // n_a, 1)
        n_full = n_a * group_size
        a_eff_grouped = a_eff[:n_full].reshape(n_a, group_size).mean(dim=1, keepdim=True)
        return a_eff_grouped

    def _quar_compute_ab_grouped(
        self, x: torch.Tensor, src: dict, module
    ) -> tuple:
        """Per-group (a, b) for PTQ4ViT scale_offset folding.

        Returns (a_grouped, b_grouped) each of shape (n_a, 1).
        a = (1-alpha) + alpha * clamp(sig_s / sig_t)   (blended scale factor)
        b = alpha * (mu_s - sig_ratio * mu_t)           (mean shift term)
        """
        dev = x.device
        sig_s = src["std"].to(dev)
        mu_s  = src["mean"].to(dev)
        x_rs  = x.reshape(-1, x.shape[-1])
        mu_t  = x_rs.mean(dim=0)
        sig_t = x_rs.std(dim=0)
        sig_ratio = (sig_s / (sig_t + self._quar_eps)).clamp(
            1.0 / self._quar_clip_ratio, self._quar_clip_ratio
        )
        a_ch = (1.0 - self._quar_alpha) + self._quar_alpha * sig_ratio  # (C,)
        b_ch = self._quar_alpha * (mu_s - sig_ratio * mu_t)             # (C,)
        n_a = module.a_interval.shape[0]
        C = x.shape[-1]
        group_size = max(C // n_a, 1)
        n_full = n_a * group_size
        a_grouped = a_ch[:n_full].reshape(n_a, group_size).mean(dim=1, keepdim=True)
        b_grouped = b_ch[:n_full].reshape(n_a, group_size).mean(dim=1, keepdim=True)
        return a_grouped, b_grouped

    def _quar_compute_ab_perchan(
        self, x: torch.Tensor, src: dict, module
    ) -> tuple:
        """Per-channel (a, b) for PTQ4ViT scale_offset_decoupled folding.

        Returns (a_perchan, b_perchan) each of shape (n_a, crb_acts), matching
        the per-channel correction granularity of _quar_correct (explicit mode).
        """
        dev = x.device
        sig_s = src["std"].to(dev)
        mu_s  = src["mean"].to(dev)
        x_rs  = x.reshape(-1, x.shape[-1])
        mu_t  = x_rs.mean(dim=0)
        sig_t = x_rs.std(dim=0)
        sig_ratio = (sig_s / (sig_t + self._quar_eps)).clamp(
            1.0 / self._quar_clip_ratio, self._quar_clip_ratio
        )
        a_ch = (1.0 - self._quar_alpha) + self._quar_alpha * sig_ratio  # (C,)
        b_ch = self._quar_alpha * (mu_s - sig_ratio * mu_t)             # (C,)
        n_a = module.a_interval.shape[0]
        C = x.shape[-1]
        group_size = max(C // n_a, 1)
        n_full = n_a * group_size
        a_perchan = a_ch[:n_full].reshape(n_a, group_size)
        b_perchan = b_ch[:n_full].reshape(n_a, group_size)
        return a_perchan, b_perchan

    def _quar_compute_ab_scalar(
        self, x: torch.Tensor, src: dict
    ) -> tuple:
        """Global scalar (a, b) for AdaLog scale_shift folding.

        Returns (a, b) as scalar tensors on x.device.
        Uses channel-mean of source stats and global target stats.
        """
        dev = x.device
        sig_s = src["std"].to(dev).mean()
        mu_s  = src["mean"].to(dev).mean()
        x_rs  = x.reshape(-1)
        mu_t  = x_rs.mean()
        sig_t = x_rs.std()
        sig_ratio = (sig_s / (sig_t + self._quar_eps)).clamp(
            1.0 / self._quar_clip_ratio, self._quar_clip_ratio
        )
        a = (1.0 - self._quar_alpha) + self._quar_alpha * sig_ratio
        b = self._quar_alpha * (mu_s - sig_ratio * mu_t)
        return a, b

    def _make_qac_hook(self, name):
        """Wrap the hook so a per-block lambda profile can override _quar_alpha.

        Additive: with --quar_local_alpha_profile unset the impl hook is returned as-is.
        """
        inner = self._make_qac_hook_impl(name)
        prof = getattr(self, "_quar_alpha_profile", None)
        if not prof:
            return inner
        import re as _re_ap
        _m = _re_ap.search(r"blocks\.(\d+)\.", name)
        a_blk = prof.get(int(_m.group(1))) if _m else None
        if a_blk is None:
            return inner

        def wrapped(module, inputs):
            saved = self._quar_alpha
            self._quar_alpha = a_blk
            try:
                return inner(module, inputs)
            finally:
                self._quar_alpha = saved
        return wrapped

    def _make_qac_hook_impl(self, name):
        def hook(module, inputs):
            if not self._quar_active or not inputs:
                return None
            x = inputs[0]
            if not torch.is_tensor(x):
                return None
            src = self._quar_source_stats.get(name)
            if src is None:
                if self._quar_stats_mode == "target_only":
                    src = {}  # target_only uses only online target stats; source file not needed
                else:
                    return None

            if self._quar_mode == "folded_quantizer":
                fold = self._quar_fold_type
                interval = getattr(module, "a_interval", None)
                aq = getattr(module, "a_quantizer", None)

                if torch.is_tensor(interval):
                    # PTQ4ViT uniform quantizer path
                    with torch.no_grad():
                        dev_iv = interval.device
                        dtype_iv = interval.dtype
                        if fold in ("scale_offset", "scale_offset_coupled"):
                            # Coupled: fold (a,b) into interval+qzero_z.
                            # q = round(x / interval' + z)  — codes match explicit.
                            # Dequant: (q-z)*interval' ≈ x  (correction cancelled).
                            a_g, b_g = self._quar_compute_ab_grouped(x.detach(), src, module)
                            self._quar_interval_backup[name] = interval.clone()
                            interval.div_(a_g.to(device=dev_iv, dtype=dtype_iv))
                            z = (b_g / self._quar_interval_backup[name]).to(device=dev_iv, dtype=dtype_iv)
                            self._quar_zero_point_backup[name] = getattr(module, "qzero_z", None)
                            module.qzero_z = z
                        elif fold == "scale_offset_decoupled":
                            # Decoupled: fold (a,b) via qaffine_gamma/beta (per-channel).
                            # quant_input path: x_cal = a_ch*x + b_ch, then q = round(x_cal/interval),
                            # dequant = q * interval  (original interval, z=0).
                            # Per-channel (n_a, gs) shape matches _quar_correct granularity exactly.
                            a_ch, b_ch = self._quar_compute_ab_perchan(x.detach(), src, module)
                            module.qaffine_gamma = a_ch.to(device=dev_iv, dtype=dtype_iv)
                            module.qaffine_beta  = b_ch.to(device=dev_iv, dtype=dtype_iv)
                            self._quar_qaffine_backup[name] = True
                        elif fold == "output_only":
                            # Output-only: quantize with original parameters; apply γ·dq + β after dequant.
                            # Tests whether output distribution correction alone explains local-branch gain,
                            # without changing code selection (no input modification).
                            a_ch, b_ch = self._quar_compute_ab_perchan(x.detach(), src, module)
                            module._quar_post_gamma = a_ch.to(device=dev_iv, dtype=dtype_iv)
                            module._quar_post_beta  = b_ch.to(device=dev_iv, dtype=dtype_iv)
                            self._quar_post_backup[name] = True
                        else:
                            # scale_only: fold a into interval only (legacy behavior)
                            a_eff = self._quar_compute_a_eff_for_interval(x.detach(), src, module)
                            self._quar_interval_backup[name] = interval.clone()
                            interval.div_(a_eff.to(device=dev_iv, dtype=dtype_iv))
                    return None

                elif (aq is not None
                      and torch.is_tensor(getattr(aq, "scale", None))):
                    # AdaLog quantizer path (ShiftAdaLog: scale+shift; UniformQuantizer: scale+zero_point)
                    _aq_has_shift = torch.is_tensor(getattr(aq, "shift", None))
                    if _aq_has_shift and fold in ("scale_shift", "scale_shift_coupled"):
                        # Coupled: fold (a,b) into AdaLog scale+shift.
                        # scale' = scale/a, shift' = (b + shift) / a
                        # Integer codes match the explicit local branch; dequant output differs.
                        with torch.no_grad():
                            a_s, b_s = self._quar_compute_ab_scalar(x.detach(), src)
                            self._quar_adalog_backup[name] = {
                                "scale": aq.scale.data.clone(),
                                "shift": aq.shift.data.clone(),
                            }
                            new_shift = (b_s + aq.shift.data) / a_s
                            aq.scale.data.div_(a_s)
                            aq.shift.data.copy_(new_shift)
                        return None
                    elif fold in ("output_only", "scale_shift_decoupled"):
                        # Scalar post-correction on module output (applied in restore hook).
                        # output_only: no input modification; scalar (a,b) corrects output dist.
                        # scale_shift_decoupled: explicit pre-correction + output post-correction.
                        a_s, b_s = self._quar_compute_ab_scalar(x.detach(), src)
                        self._quar_adalog_post_backup[name] = (a_s, b_s)
                        if fold == "output_only":
                            return None  # no input modification
                        # scale_shift_decoupled: fall through to explicit for input correction
                    # unknown folds: fall through to explicit

            # explicit mode (or folded fallback for unsupported module/fold combinations
            with torch.no_grad():
                x_out = self._quar_correct(x.detach(), name, src)
            # preserve requires_grad if present (ZOA has LayerNorm grads in graph)
            if x.requires_grad:
                x_out = x_out.detach().requires_grad_(True)
            return (x_out,) + inputs[1:]
        return hook

    def _make_qac_output_hook(self, name):
        """Placement control: the explicit correction on a module's OUTPUT (residual stream)."""
        def hook(module, inputs, output):
            if not self._quar_active or not torch.is_tensor(output):
                return None
            src = self._quar_source_stats.get(name)
            if src is None:
                return None
            with torch.no_grad():
                out = self._quar_correct(output.detach(), name, src)
            if output.requires_grad:
                out = out.detach().requires_grad_(True)
            return out
        return hook

    def _make_qac_restore_hook(self, name):
        """Post-hook: restore patched quantizer state after folded_quantizer forward pass."""
        def hook(module, inputs, output):
            # Restore a_interval (PTQ4ViT scale_only / scale_offset_coupled)
            saved_iv = self._quar_interval_backup.pop(name, None)
            if saved_iv is not None:
                interval = getattr(module, "a_interval", None)
                if torch.is_tensor(interval):
                    with torch.no_grad():
                        interval.copy_(saved_iv)

            # Restore qzero_z (PTQ4ViT scale_offset_coupled)
            if name in self._quar_zero_point_backup:
                saved_z = self._quar_zero_point_backup.pop(name)
                if saved_z is None:
                    module.__dict__.pop("qzero_z", None)
                else:
                    module.qzero_z = saved_z

            # Remove qaffine_gamma/beta (PTQ4ViT scale_offset_decoupled)
            if self._quar_qaffine_backup.pop(name, None):
                module.__dict__.pop("qaffine_gamma", None)
                module.__dict__.pop("qaffine_beta", None)

            # Remove _quar_post_gamma/beta (PTQ4ViT output_only)
            if self._quar_post_backup.pop(name, None):
                module.__dict__.pop("_quar_post_gamma", None)
                module.__dict__.pop("_quar_post_beta", None)

            # Restore AdaLog scale+shift (scale_shift_coupled)
            saved_aq = self._quar_adalog_backup.pop(name, None)
            if saved_aq is not None:
                aq = getattr(module, "a_quantizer", None)
                if aq is not None:
                    with torch.no_grad():
                        aq.scale.data.copy_(saved_aq["scale"])
                        aq.shift.data.copy_(saved_aq["shift"])

            # AdaLog output_only / scale_shift_decoupled: scalar post-correction on module output
            adalog_post = self._quar_adalog_post_backup.pop(name, None)
            if adalog_post is not None and torch.is_tensor(output):
                a_s, b_s = adalog_post
                a_s = a_s.to(device=output.device, dtype=output.dtype)
                b_s = b_s.to(device=output.device, dtype=output.dtype)
                return output * a_s + b_s
        return hook

    def _quar_correct(self, x: torch.Tensor, name: str, src: dict) -> torch.Tensor:
        dev = x.device
        mode = self._quar_stats_mode

        needs_src = mode not in ("target_only",)
        if needs_src:
            mu_s  = src["mean"].to(dev)
            sig_s = src["std"].to(dev)
        else:
            mu_s = sig_s = None

        # Per-batch target stats over B*N spatial tokens → per-channel
        x_rs = x.reshape(-1, x.shape[-1])  # [B*N, C]
        mu_t  = x_rs.mean(dim=0)
        sig_t = x_rs.std(dim=0)

        rho = 1.0  # warm-up blend factor; 1.0 means full correction

        if self._quar_estimator == "welford" and needs_src:
            # Exact streaming POOLED variance (Welford parallel/batch merge) over B*N tokens,
            # accumulated across batches within a domain. Source-initialized as one pseudo-batch:
            #   count0 = this batch's token count, mean0 = mu_s, M2_0 = sig_s^2 * count0.
            # Inclusive: merge the current batch, then correct with the updated pooled estimate.
            n_new = x_rs.shape[0]  # B*N tokens in this batch
            b_mean = mu_t.detach()                                       # per-channel batch mean
            b_M2 = x_rs.detach().var(dim=0, unbiased=False) * float(n_new)  # within-batch sum of sq dev
            if name not in self._quar_welford_count:
                self._quar_welford_count[name] = float(n_new)
                self._quar_welford_mean[name] = mu_s.float().clone()
                self._quar_welford_M2[name] = (sig_s.float() ** 2 * float(n_new)).clone()
            mean_prev  = self._quar_welford_mean[name].to(dev, dtype=x.dtype)
            M2_prev    = self._quar_welford_M2[name].to(dev, dtype=x.dtype)
            count_prev = self._quar_welford_count[name]
            _frozen = getattr(self, "_recal_freeze", False)
            if not _frozen:
                delta = b_mean - mean_prev
                tot = count_prev + n_new
                new_mean = mean_prev + delta * (n_new / tot)
                new_M2 = M2_prev + b_M2 + (delta ** 2) * (count_prev * n_new / tot)
                self._quar_welford_mean[name]  = new_mean
                self._quar_welford_M2[name]    = new_M2
                self._quar_welford_count[name] = tot
                mu_t  = new_mean
                sig_t = (new_M2 / tot).clamp(min=0.0).sqrt()
            else:
                mu_t  = mean_prev
                sig_t = (M2_prev / count_prev).clamp(min=0.0).sqrt()
            rho = 1.0

        elif self._quar_stream_ema and needs_src:
            # Source-initialized streaming EMA with second-moment variance tracking.
            # Initializes from source stats so correction is ~identity at start, avoids
            # cold-start collapse at small batch sizes.
            alpha  = self._quar_stream_alpha
            n_warm = self._quar_stream_nwarm
            n_new  = x.shape[0]  # images in this batch

            if name not in self._quar_stream_m:
                self._quar_stream_m[name] = mu_s.float().clone()
                self._quar_stream_q[name] = (sig_s.float() ** 2 + mu_s.float() ** 2).clone()
                self._quar_stream_n[name] = 0
                self._quar_stream_bc[name] = 1  # source counts as one pseudo-batch (cumulative)

            m_prev = self._quar_stream_m[name].to(dev, dtype=x.dtype)
            q_prev = self._quar_stream_q[name].to(dev, dtype=x.dtype)
            n_prev = self._quar_stream_n[name]
            q_batch = (x_rs.detach() ** 2).mean(dim=0)  # per-channel E[y^2]
            # Cumulative (source-initialized running mean): alpha_n = 1/(batch_count+1) so the
            # estimate is a count-weighted average of all batches seen (forgetting-free, like NEO's
            # cumulative mean), instead of a fixed-rate EMA. Resets per domain via reset() (episodic).
            if self._recal_cumulative:
                alpha = 1.0 / (self._quar_stream_bc[name] + 1.0)

            _frozen = getattr(self, "_recal_freeze", False)
            if self._quar_stream_lag:
                # Lagged: use prev EMA state for correction, then update (skip the update
                # when frozen, e.g. during ZOA SPSA perturbation forwards).
                m_use, q_use, n_use = m_prev, q_prev, n_prev
                if not _frozen:
                    self._quar_stream_m[name] = (1 - alpha) * m_prev.float() + alpha * mu_t.float()
                    self._quar_stream_q[name] = (1 - alpha) * q_prev.float() + alpha * q_batch.float()
                    self._quar_stream_n[name] = n_prev + n_new
                    self._quar_stream_bc[name] += 1
            else:
                # Inclusive: update EMA first, then use updated state
                if not _frozen:
                    m_new = (1 - alpha) * m_prev.float() + alpha * mu_t.float()
                    q_new = (1 - alpha) * q_prev.float() + alpha * q_batch.float()
                    self._quar_stream_m[name] = m_new
                    self._quar_stream_q[name] = q_new
                    self._quar_stream_n[name] = n_prev + n_new
                    self._quar_stream_bc[name] += 1
                    m_use = m_new.to(dev, dtype=x.dtype)
                    q_use = q_new.to(dev, dtype=x.dtype)
                    n_use = n_prev + n_new
                else:
                    m_use, q_use, n_use = m_prev, q_prev, n_prev

            # Variance from second moment: Var = E[y^2] - E[y]^2; clamp for numerical safety
            sig_t = (q_use - m_use ** 2).clamp(min=self._quar_eps).sqrt()
            mu_t  = m_use
            rho   = float(min(1.0, n_use / n_warm)) if n_warm > 0 else 1.0

        elif self._quar_use_ema:
            # Old token-level EMA (effectively per-batch at all bs due to [B*N,C] reshape)
            m = self._quar_ema_momentum
            if name not in self._quar_ema_mu:
                self._quar_ema_mu[name]  = mu_t.clone()
                self._quar_ema_sig[name] = sig_t.clone()
            else:
                self._quar_ema_mu[name]  = (1 - m) * self._quar_ema_mu[name]  + m * mu_t
                self._quar_ema_sig[name] = (1 - m) * self._quar_ema_sig[name] + m * sig_t
            mu_t  = self._quar_ema_mu[name]
            sig_t = self._quar_ema_sig[name]

        clip = self._quar_clip_ratio
        eps  = self._quar_eps

        sig_ratio = None

        # Oracle target moments: replace the online estimate with per-channel moments
        # measured on this very stream, so the shrinkage optimum can be read at
        # delta^2 ~ 0. Diagnostic only; never set in the deployed configuration.
        if getattr(self, "_quar_oracle_tgt", None):
            _o = self._quar_oracle_tgt.get(name)
            if _o is not None:
                mu_t  = _o[0].to(mu_t.device, dtype=mu_t.dtype)
                sig_t = _o[1].to(sig_t.device, dtype=sig_t.dtype)
                rho   = 1.0
        _chunking = (self._quar_chunk > 1 and not x.requires_grad
                     and x.shape[0] >= self._quar_chunk)
        if _chunking:
            # x_corr is built per slice below; skip the whole-tensor form.
            sig_ratio = None
            if mode in ("std_centered", "std_only", "mean_std"):
                sig_ratio = (sig_s / (sig_t + eps)).clamp(1.0 / clip, clip)
            x_corr = None
        elif mode == "target_only":
            x_corr = (x - mu_t) / (sig_t + eps)
        elif mode == "mean_only":
            x_corr = x + (mu_s - mu_t)
        elif mode == "std_centered":
            sig_ratio = (sig_s / (sig_t + eps)).clamp(1.0 / clip, clip)
            x_corr = (x - mu_t) * sig_ratio + mu_t
        elif mode == "std_only":
            sig_ratio = (sig_s / (sig_t + eps)).clamp(1.0 / clip, clip)
            x_corr = x * sig_ratio
        else:  # mean_std
            sig_ratio = (sig_s / (sig_t + eps)).clamp(1.0 / clip, clip)
            x_corr = (x - mu_t) * sig_ratio + mu_s

        effective_alpha = rho * self._quar_alpha

        # --quar_local_chunk=n applies the correction in n slices, in place, instead of materialising
        # x_corr and x_out whole. Peak memory is set by one interface only: fc2, whose input is
        # the 3072-wide MLP hidden state, four times the width of the other fifteen. Measured on
        # Jetson at batch 16, the twelve 768-wide interfaces cost 0.05 MB between them while fc2
        # alone costs 37 MB, because its temporary is larger than any block the caching
        # allocator already holds free. With this on, peak drops to within 0.17 MB of an
        # unadapted forward at both batch 16 and batch 64.
        #
        # This is bit-identical, not an approximation, which matters because the corrected
        # activation feeds a quantiser and a one-ULP move can flip an integer code. Every
        # operation below is elementwise or a broadcast against a per-channel vector, so no
        # element's result depends on any other and slicing cannot change a value. The two
        # reductions that DO span the tensor, mu_t and sig_t, are taken above over the whole
        # input before any slice is touched. The operations and their order are the ones the
        # unchunked expression uses -- (1-a)*x, then a*x_corr, then their sum -- and IEEE 754
        # results do not depend on which buffer they are written to. Verified on real
        # activations: 160/160 hook calls identical, 0 of 677,806,080 elements differing.
        #
        # Folding the correction into the single affine this file already computes elsewhere,
        # x*a_eff + b_ch, would also remove the buffer but is NOT bit-identical: measured over
        # 80 hook calls it changed 36.8% of elements by up to 1.5e-5.
        _nchunk = self._quar_chunk
        if _chunking:
            # In place on x: at every interface in this scope the input is a freshly produced
            # intermediate (a LayerNorm output, an activation output, an attention output) whose
            # only consumer is the module being hooked, so overwriting it aliases nothing. The
            # residual stream is never a hook input.
            _step = -(-x.shape[0] // _nchunk)
            for _i in range(0, x.shape[0], _step):
                _sl = x[_i:_i + _step]
                if mode == "target_only":
                    _c = (_sl - mu_t) / (sig_t + eps)
                elif mode == "mean_only":
                    _c = _sl + (mu_s - mu_t)
                elif mode == "std_centered":
                    _c = (_sl - mu_t) * sig_ratio + mu_t
                elif mode == "std_only":
                    _c = _sl * sig_ratio
                else:
                    _c = (_sl - mu_t) * sig_ratio + mu_s
                _c.mul_(effective_alpha)
                _sl.mul_(1.0 - effective_alpha).add_(_c)
            x_out = x
        else:
            x_out = (1.0 - effective_alpha) * x + effective_alpha * x_corr

        buf = self._quar_diag_buf.get(name)
        if buf is not None and needs_src:
            buf.append({
                "mu_s":    float(mu_s.mean()),
                "sig_s":   float(sig_s.mean()),
                "mu_t":    float(mu_t.mean()),
                "sig_t":   float(sig_t.mean()),
                "sig_rat": float(sig_ratio.mean()) if sig_ratio is not None else 1.0,
                "mu_shft": float((mu_t - mu_s).abs().mean()),
            })

        return x_out

    # ────────────────────────────────────────────── forward override (activate hooks)

    def forward(self, x):
        if not self._quar_enable:
            return super().forward(x)
        self._quar_active = True
        try:
            result = super().forward(x)
        finally:
            self._quar_active = False
            # Restore any state not cleaned up by post-hooks (e.g., if forward raised)
            _mod_map = {n: m for n, m in self._quar_modules}
            for _name, _saved in list(self._quar_interval_backup.items()):
                _m = _mod_map.get(_name)
                if _m is not None:
                    _iv = getattr(_m, "a_interval", None)
                    if torch.is_tensor(_iv):
                        with torch.no_grad():
                            _iv.copy_(_saved)
            self._quar_interval_backup.clear()
            for _name, _saved_z in list(self._quar_zero_point_backup.items()):
                _m = _mod_map.get(_name)
                if _m is not None:
                    if _saved_z is None:
                        _m.__dict__.pop("qzero_z", None)
                    else:
                        _m.qzero_z = _saved_z
            self._quar_zero_point_backup.clear()
            for _name, _saved_aq in list(self._quar_adalog_backup.items()):
                _m = _mod_map.get(_name)
                if _m is not None:
                    _aq = getattr(_m, "a_quantizer", None)
                    if _aq is not None:
                        with torch.no_grad():
                            _aq.scale.data.copy_(_saved_aq["scale"])
                            _aq.shift.data.copy_(_saved_aq["shift"])
            self._quar_adalog_backup.clear()
            for _name in list(self._quar_qaffine_backup.keys()):
                _m = _mod_map.get(_name)
                if _m is not None:
                    _m.__dict__.pop("qaffine_gamma", None)
                    _m.__dict__.pop("qaffine_beta", None)
            self._quar_qaffine_backup.clear()
            for _name in list(self._quar_post_backup.keys()):
                _m = _mod_map.get(_name)
                if _m is not None:
                    _m.__dict__.pop("_quar_post_gamma", None)
                    _m.__dict__.pop("_quar_post_beta", None)
            self._quar_post_backup.clear()
        self._quar_log_diag()
        return result

    def _quar_log_diag(self):
        if self._quar_detail_logged:
            return
        all_empty = all(len(v) == 0 for v in self._quar_diag_buf.values())
        if all_empty:
            return
        self._quar_detail_logged = True
        self.model.logger.info("the local branch first-batch activation stats:")
        for name in [nm for nm, _ in self._quar_modules]:
            buf = self._quar_diag_buf.get(name, [])
            if not buf:
                continue
            d = buf[-1]
            self.model.logger.info(
                f"  {name}: mu_s={d['mu_s']:.4f} sig_s={d['sig_s']:.4f} "
                f"mu_t={d['mu_t']:.4f} sig_t={d['sig_t']:.4f} "
                f"sig_ratio={d['sig_rat']:.3f} mu_shift={d['mu_shft']:.4f}"
            )

    # ────────────────────────────────────────────── episodic reset

    def reset(self):
        super().reset()
        if not self._quar_enable:
            return
        self._quar_ema_mu.clear()
        self._quar_ema_sig.clear()
        self._quar_stream_m.clear()
        self._quar_stream_q.clear()
        self._quar_stream_n.clear()
        self._quar_stream_bc.clear()
        self._quar_welford_mean.clear()
        self._quar_welford_M2.clear()
        self._quar_welford_count.clear()
        for buf in self._quar_diag_buf.values():
            buf.clear()
        self._quar_detail_logged = False



class _NoAdaptBase(nn.Module):
    """Minimal ZOAOfficial-compatible base for QuARLocalMixin without TTA adaptation.

    Expects ``model`` to be a namespace/object with:
      - ``.model``: the raw ViT (nn.Module) — used for module selection in QuARLocalMixin
      - ``.logger``: a logger instance
    """

    def __init__(self, args, model, fitness_lambda=30.0, log_dir=None, **kwargs):
        super().__init__()
        self.args = args
        self.model = model  # proxy: .model -> raw ViT, .logger -> logger

    def forward(self, x):
        return self.model.model(x)

    def reset(self):
        pass



class NoAdaptQACWrapper(QuARLocalMixin, _NoAdaptBase):
    """Applies the local branch pre-quantizer correction to a frozen model (no TTA).

    Use with ``--algorithm no_adapt --quar_local_enable``.
    Construct via ``build_no_adapt_qac_wrapper(args, net, logger)``.
    """
    pass



def build_no_adapt_qac_wrapper(args, net, _logger):
    """Create a NoAdaptQACWrapper around a raw quantized net.

    ``net`` is the raw ViT (possibly quantized); not wrapped in PromptViT for
    no_adapt.  Builds a lightweight proxy so QuARLocalMixin can call
    ``self.model.model.named_modules()`` and ``self.model.logger``.
    """
    import types as _types
    raw_model = getattr(net, "model", net)
    proxy = _types.SimpleNamespace(model=raw_model, logger=_logger)
    return NoAdaptQACWrapper(args, proxy)


