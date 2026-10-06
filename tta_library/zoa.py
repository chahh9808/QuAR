"""ZOA (Deng et al., 2025), the zeroth-order prompt adaptation baseline.

A port of the published method onto this project's quantized graph rather than a copy: the
closest upstream file, ZOA_official/tta_library/zoa_vit.py, is 11% similar. The paper-faithful
details it does reproduce are marked in the code, and configure_model_for_official_zoa follows
upstream's configure_model including the fix_layers pattern.

Upstream: https://github.com/mala-lab/ZOA (MIT, (c) 2025 DengZeshuai).
"""
import csv
import math
import os
import random as _random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tta_library.fozo import FOZO


@torch.jit.script
def official_zoa_softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    x = x / 1.0
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


criterion_mse = nn.MSELoss(reduction="mean")


class ZOAOfficial(nn.Module):
    """Paper-faithful official ZOA ViT adapter port."""

    def __init__(self, args, model, fitness_lambda=30.0, log_dir=None):
        super().__init__()
        self.args = args
        self.steps = args.official_zoa_steps
        self.model = model
        self.embed_dim = model.embed_dim
        self.num_features = model.num_features
        self.fitness_lambda = fitness_lambda
        self.hist_stat = None
        self.forward_loss_calls = 0
        self.batch_count = 0
        self.current_corruption = ""
        self.metrics_path = os.path.join(log_dir, "official_zoa_metrics.csv") if log_dir else None
        self._metrics_header_written = False
        self.layer_balance_mode = getattr(args, "zoa_layer_balance", "none")
        self.lb_eps = getattr(args, "zoa_layer_balance_eps", 1e-4)
        self.lb_clip_min = getattr(args, "zoa_layer_balance_clip_min", 0.1)
        self.lb_clip_max = getattr(args, "zoa_layer_balance_clip_max", 10.0)
        self._lb_weights = None  # set by _setup_layer_balance after train stats are loaded

    def _update_hist(self, batch_mean):
        if self.hist_stat is None:
            self.hist_stat = batch_mean.detach()
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean.detach()

    def _get_shift_vector(self):
        if self.hist_stat is None:
            return None
        return self.train_info[1][-self.num_features:] - self.hist_stat

    def forward(self, x):
        self.batch_count += 1
        calls_before = self.forward_loss_calls
        shift_vector = self._get_shift_vector()
        outputs, batch_mean, loss, domain_change = self.forward_and_get_loss(x, shift_vector)
        if domain_change:
            self.model.logger.info("Official ZOA reset shift vector after domain memory update.")
            self.hist_stat = None
            shift_vector = None

        alphas, params, _ = self.model.parameters_to_vector()
        self.params_ori = params.detach().clone()

        if self.steps > 0:
            for step in range(self.steps):
                ghat_alpha, ghat, loss = self.spsa_grad_estimate_bi(
                    x,
                    alphas,
                    params,
                    shift_vector=shift_vector,
                    ck_alpha=self.args.official_zoa_spsa_c_alpha,
                    ck=self.args.official_zoa_spsa_c,
                    sp_avg=self.args.official_zoa_sp_avg,
                    loss=loss if step == 0 else None,
                )
                self.model.vector_to_parameters(alphas, self.params_ori)
                self.model.record_gradient(ghat_alpha, ghat)
                self.model.optimizes()
                alphas, params, _ = self.model.parameters_to_vector()
                self.params_ori = params.detach().clone()

        self._update_hist(batch_mean[-self.model.num_features:])
        self._log_metrics(loss, domain_change, self.forward_loss_calls - calls_before)
        return outputs

    def _setup_layer_balance(self):
        """Compute per-layer balance weights from train_std (source_norm mode only)."""
        if self.layer_balance_mode != "source_norm":
            self._lb_weights = None
            return
        train_std = self.train_info[0]
        D = self.embed_dim
        n = train_std.shape[0] // D
        # Average source std per layer: higher std → larger expected D_l → lower weight.
        per_layer_std = train_std.view(n, D).mean(dim=1)
        w = 1.0 / (per_layer_std + self.lb_eps)
        w = w.clamp(self.lb_clip_min, self.lb_clip_max)
        w = w / w.mean()  # normalize so mean weight = 1
        self._lb_weights = w.detach()
        self.model.logger.info(
            "ZOA layer balance weights [source_norm]: "
            + ", ".join(f"L{i}={v:.3f}" for i, v in enumerate(w.tolist()))
        )

    def _compute_discrepancy(self, batch_std, batch_mean, train_std, train_mean):
        """Feature-alignment discrepancy loss with optional layer balancing."""
        if self.layer_balance_mode == "none":
            return self.fitness_lambda * (
                criterion_mse(batch_std, train_std) + criterion_mse(batch_mean, train_mean)
            )
        D = self.embed_dim
        n = batch_std.shape[0] // D
        lam = self.fitness_lambda
        # Per-layer D_l values (reuses already-computed batch_std/batch_mean slices).
        Dl = torch.stack([
            lam * (
                criterion_mse(batch_std[l * D:(l + 1) * D], train_std[l * D:(l + 1) * D])
                + criterion_mse(batch_mean[l * D:(l + 1) * D], train_mean[l * D:(l + 1) * D])
            )
            for l in range(n)
        ])
        if self.layer_balance_mode == "source_norm":
            w = self._lb_weights.to(batch_std.device)
        else:  # uniform_batch: normalize by current-batch D_l scale (ablation)
            scale = Dl.detach().clamp_min(self.lb_eps)
            w = (1.0 / scale).clamp(self.lb_clip_min, self.lb_clip_max)
            w = w / w.mean()
        return (w * Dl).mean()

    def forward_and_get_loss(self, images, shift_vector):
        self.forward_loss_calls += 1
        with torch.no_grad():
            features, domain_change = self.model.layers_cls_features_with_prompts(images)
            train_std, train_mean = self.train_info
            # At batch size 1 the unbiased estimator has zero degrees of freedom, so std is NaN,
            # which poisons the fitness and makes the adapter return None. Fall back to the biased
            # estimator (batch term 0) ONLY then: at any batch >= 2 this is a no-op, so the
            # batch-16 and batch-64 numbers are untouched. Adaptation quality at batch 1 is not
            # meaningful either way -- see jetson/device_env.txt -- but the compute path stays
            # intact, which is what a latency and memory measurement needs.
            batch_std, batch_mean = torch.std_mean(features, dim=0, correction=0 if features.shape[0] < 2 else 1)
            discrepancy_loss = self._compute_discrepancy(
                batch_std, batch_mean, train_std, train_mean
            )
            cls_features = features[:, -self.model.num_features:]
            output = self.model.model.head(cls_features)
            entropy_loss = official_zoa_softmax_entropy(output).mean()
            loss = discrepancy_loss + entropy_loss
            if shift_vector is not None:
                output = self.model.model.head(cls_features + shift_vector)
        return output, batch_mean, loss, domain_change

    def spsa_grad_estimate_bi(
        self,
        inputs,
        alphas,
        params,
        shift_vector=None,
        ck_alpha=0.05,
        ck=0.02,
        sp_avg=1,
        loss=None,
    ):
        ghats_alpha, ghats = [], []
        n_alphas = len(alphas)
        n_params = len(params)
        device = params.device

        # Freeze the local branch/CLS streaming target-moment updates during the SPSA perturbation
        # forwards: these run on parameter-perturbed models and would otherwise corrupt
        # the running statistics (which assume one clean forward per batch). The
        # recalibration is still applied (using the current moments); only the EMA update
        # is skipped. The clean prediction forward in forward() runs with the flag off.
        self._recal_freeze = True
        for _ in range(sp_avg):
            perturb_a = self._segmented_uniform_perturbation(n_alphas, device)
            perturb = self._segmented_uniform_perturbation(n_params, device)
            alphas_r = alphas + ck_alpha * perturb_a
            params_r = params + ck * perturb
            self.model.vector_to_parameters(alphas_r, params_r)
            _, _, loss1, _ = self.forward_and_get_loss(inputs, shift_vector)

            if loss is not None:
                loss2 = loss
            else:
                self.model.vector_to_parameters(alphas, params)
                _, _, loss2, _ = self.forward_and_get_loss(inputs, shift_vector)
                loss = loss2

            ghats_alpha.append(((loss1 - loss2) / (ck_alpha * perturb_a)).reshape(1, -1))
            ghats.append(((loss1 - loss2) / (ck * perturb)).reshape(1, -1))
            loss = (loss1 + loss2) / 2.0

        self._recal_freeze = False
        return torch.cat(ghats_alpha, dim=0).mean(dim=0), torch.cat(ghats, dim=0).mean(dim=0), loss

    @staticmethod
    def _segmented_uniform_perturbation(size, device):
        p_side = (torch.rand(size, 1, device=device) + 1.0) / 2.0
        samples = torch.cat([p_side, -p_side], dim=1)
        index = torch.bernoulli(torch.ones_like(p_side) / 2.0).to(torch.int64)
        return torch.gather(samples, 1, index).reshape(-1)

    def obtain_origin_stat(self, train_loader, stats_file=None, max_batches=-1):
        cache_allowed = max_batches <= 0
        if cache_allowed and stats_file and os.path.exists(stats_file):
            stats = torch.load(stats_file, map_location="cuda", weights_only=False)
            self.train_info = (stats["std"].cuda(), stats["mean"].cuda())
            self.model.logger.info(f"Official ZOA source stats loaded from {stats_file}")
            self._setup_layer_balance()
            return

        self.model.logger.info("Official ZOA calculating source mean/std.")
        self.model.eval()
        features = []
        with torch.no_grad():
            for batch_idx, dl in enumerate(train_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break
                images = dl[0].cuda()
                features.append(self.model.layers_cls_features(images).detach().cpu())
            features = torch.cat(features, dim=0).cuda()
            train_std, train_mean = torch.std_mean(features, dim=0)
            self.train_info = (train_std, train_mean)
        self._setup_layer_balance()
        if cache_allowed and stats_file:
            os.makedirs(os.path.dirname(stats_file) or ".", exist_ok=True)
            torch.save({"std": train_std.cpu(), "mean": train_mean.cpu()}, stats_file)
            self.model.logger.info(f"Official ZOA source stats saved to {stats_file}")
        elif stats_file:
            self.model.logger.info("Official ZOA source stats cache skipped because max_batches is set.")

    def reset(self):
        self.hist_stat = None
        self.batch_count = 0
        self.forward_loss_calls = 0
        if hasattr(self.model, "reset"):
            self.model.reset()

    def _log_metrics(self, loss, domain_change, forward_loss_calls):
        if not self.metrics_path:
            return
        alpha = self.model.alpha_distribution()
        row = {
            "corruption": self.current_corruption,
            "batch": self.batch_count,
            "loss": float(loss.detach().item()) if torch.is_tensor(loss) else float(loss),
            "domain_change": int(bool(domain_change)),
            "memory_vectors": int(max(0, self.model.weight_nums - 1)),
            "alpha_count": int(alpha.numel()),
            "alpha_max": float(alpha.max().item()) if alpha.numel() else 0.0,
            "alpha_entropy": float((-(alpha * (alpha + 1e-12).log()).sum()).item()) if alpha.numel() else 0.0,
            "forward_loss_calls": int(forward_loss_calls),
            "fused_layernorm_count": int(len(getattr(self.model, "fuse_modules", []))),
        }
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._metrics_header_written:
                writer.writeheader()
                self._metrics_header_written = True
            writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# QBAMixin — Quantization Boundary Alignment
#
# Aligns source vs target statistics of the normalized cell coordinate:
#     u = (A - c_k) / cell_hw
# where c_k = DeQuant(Quant(A)) and cell_hw is the half-width of the
# quantization cell containing A.
#
# For uniform quantizers:  cell_hw = step/2  (constant)
# For log quantizers:      cell_hw = c_k * (ratio^0.5 - ratio^-0.5) / 2
#
# Since u ∈ [-1, 1] by construction for non-saturated elements, the
# statistics (mean_abs_u, boundary_rate) are naturally scale-invariant
# across bit-widths without needing source_norm or step_norm.
#
# L_QBA = Σ_l [ |mean_abs_u_t - mean_abs_u_s|
#              + w_b * |boundary_rate_t - boundary_rate_s|
#              + w_s * |saturation_rate_t - saturation_rate_s| ]
#
# No extra forward passes.  Place before ZOAOfficial in MRO.
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# QZOMixin — Zeroth-Order Quantizer-State Adaptation
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# QuARLocalMixin — QuAR local recalibration
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# QuARLocalQuantGuidedMixin — code-distribution-guided the local branch
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# NoAdaptQACWrapper — the local branch-only baseline (no TTA adaptation)
# ─────────────────────────────────────────────────────────────────────────────

def configure_model_for_official_zoa(model, fix_paper_layers=True):
    """Replicate ZOA's configure_model (ZOA_official/tta_library/zoa_vit.py).

    Upstream calls this on the FuseViT *wrapper*, so its parameter names carry a
    'model.' prefix and its fix_layers patterns match. We call it on the bare
    quantized net, where names are 'blocks.0.norm1.weight', so the prefixed
    patterns matched nothing and every LayerNorm stayed trainable (50 tensors,
    38,400 params) instead of the paper's 32 tensors / 24,576 params. Match on
    the un-prefixed name, tolerating either naming.

    fix_paper_layers=False restores that earlier all-LayerNorm behaviour, for
    reproducing numbers measured before this was corrected.
    """
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            module.requires_grad_(True)

    if fix_paper_layers:
        # ZOA fixes the first block's LayerNorms, the last three blocks, and the
        # final norm; it adapts blocks 1-8 x {norm1, norm2}.
        fix_prefixes = ("blocks.0.norm1", "blocks.0.norm2",
                        "blocks.9.", "blocks.10.", "blocks.11.", "norm.")
        for name, param in model.named_parameters():
            bare = name[len("model."):] if name.startswith("model.") else name
            if bare.startswith(fix_prefixes):
                param.requires_grad_(False)
    return model
