"""Fused prompt-ViT wrapper for the ZOA baseline.

Adapted from https://github.com/mala-lab/ZOA, models/fuse_vit.py
(MIT, (c) 2025 DengZeshuai), about 31% unchanged. The edits carry the wrapper onto the
quantized graph and expose the optimizer handles main.py sets up.
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OfficialZoaFuseViT(nn.Module):
    """Official ZOA ViT LayerNorm domain-vector memory wrapper."""

    def __init__(self, args, model, logger):
        super().__init__()
        self.args = args
        self.model = model
        self.logger = logger
        self.embed_dim = model.embed_dim
        self.num_features = model.num_features
        self.weight_margin = 0.01
        self.domain_var = None
        self.domain_mean = None
        self.max_weight_nums = args.official_zoa_max_weight_nums
        self.weight_nums = 1
        self.signature_store_hook = None
        self.memory_admission_hook = None
        self.memory_select_remove_hook = None
        self.alpha_prior_hook = None
        self.memory_removal_hook = None

    def register_signature_store_hook(self, hook):
        self.signature_store_hook = hook

    def register_memory_admission_hook(self, hook):
        self.memory_admission_hook = hook

    def register_memory_select_remove_hook(self, hook):
        self.memory_select_remove_hook = hook

    def register_alpha_prior_hook(self, hook):
        self.alpha_prior_hook = hook

    def register_memory_removal_hook(self, hook):
        self.memory_removal_hook = hook

    def estimate_scale_factor(self):
        max_weight = max(p.abs().mean().item() for p in self.params["epsilon_weight"])
        max_bias = max(p.abs().mean().item() for p in self.params["epsilon_bias"])
        return max(max_weight, max_bias) / self.weight_margin

    @torch.no_grad()
    def save_current_weight(self):
        if (self.weight_nums - 1) >= self.args.official_zoa_max_weight_nums:
            if self.memory_select_remove_hook is not None:
                idx = self.memory_select_remove_hook()
                idx = torch.as_tensor(idx, device=self.alpha.device, dtype=torch.long) - 1
            else:
                _, k, p = self.select_similar_pair()
                idx = torch.min(k, p)
            self._remove(idx + 1)

        self.weight_nums += 1
        scale_factor = self.estimate_scale_factor()
        if torch.isnan(torch.tensor([scale_factor], device=self.alpha.device)).any():
            raise RuntimeError("Official ZOA scale factor became NaN.")

        self.logger.info(f"Official ZOA saving domain vector, scale_factor={scale_factor:.6f}")
        new_alpha_value = torch.tensor([-float("inf")], device=self.alpha.device)
        if scale_factor > 1:
            total = torch.exp(self.alpha * self.alpha_scale).sum()
            new_alpha_value = torch.log(total * (scale_factor - 1.0)).view(1)
        new_alpha_value = torch.max(torch.max(self.alpha * self.alpha_scale).view(1), new_alpha_value)
        new_alpha_value = torch.clamp(new_alpha_value / self.alpha_scale, -10, 10)
        self.alpha = nn.Parameter(torch.cat([self.alpha.detach(), new_alpha_value.detach()]))

        for module in self.model.modules():
            if isinstance(module, OfficialZoaFuseLayerNorm):
                module.save_current_weight(self.alpha)

        if self.signature_store_hook is not None:
            self.signature_store_hook(self.weight_nums - 1)
        self.update_similar_matrix(add_idx=-1)
        self.collect_params()
        self.alpha_optimizer.param_groups = []
        self.alpha_optimizer.add_param_group(
            {
                "params": self.params["alpha"],
                "lr": self.args.official_zoa_lr_alpha,
                "weight_decay": self.args.official_zoa_weight_decay_alpha,
            }
        )

    def select_similar_pair(self):
        if not hasattr(self, "similar_matrix"):
            self.similar_matrix = self.update_similar_matrix()
        max_val, max_idx = torch.max(self.similar_matrix.view(-1), dim=0)
        k = max_idx // self.similar_matrix.size(1)
        p = max_idx % self.similar_matrix.size(1)
        return max_val, k, p

    def update_similar_matrix(self, add_idx=None):
        if not hasattr(self, "similar_matrix"):
            self.similar_matrix = torch.zeros(
                (self.max_weight_nums, self.max_weight_nums),
                device=self.alpha.device,
            )

        delta_weights = [[] for _ in range(self.weight_nums - 1)]
        delta_biases = [[] for _ in range(self.weight_nums - 1)]
        for module in self.model.modules():
            if isinstance(module, OfficialZoaFuseLayerNorm):
                for i in range(self.weight_nums - 1):
                    delta_weights[i].append(module.weights[i + 1].flatten())
                    delta_biases[i].append(module.biases[i + 1].flatten())

        if add_idx is None:
            for i in range(self.weight_nums - 1):
                for j in range(i + 1, self.weight_nums - 1):
                    self.similar_matrix[i, j] = self.calculate_cosine_similarity(
                        delta_weights[i] + delta_biases[i],
                        delta_weights[j] + delta_biases[j],
                    )
        else:
            if add_idx < 0:
                add_idx += self.weight_nums - 1
            for j in range(add_idx):
                self.similar_matrix[j, add_idx] = self.calculate_cosine_similarity(
                    delta_weights[j] + delta_biases[j],
                    delta_weights[add_idx] + delta_biases[add_idx],
                )
            for i in range(add_idx + 1, self.weight_nums - 1):
                self.similar_matrix[add_idx, i] = self.calculate_cosine_similarity(
                    delta_weights[i] + delta_biases[i],
                    delta_weights[add_idx] + delta_biases[add_idx],
                )
        return self.similar_matrix

    @staticmethod
    def calculate_cosine_similarity(params_x, params_y):
        cos_sims = [F.cosine_similarity(x, y, dim=0) for x, y in zip(params_x, params_y)]
        return torch.stack(cos_sims, dim=0).mean()

    def replace_fuse_ln(self):
        self.weight_nums = 1
        self.state_dicts = [self.model.state_dict()]
        self._replace_fuse_layer()
        del self.state_dicts
        self.collect_params()

    def _replace_fuse_layer(self):
        device = next(self.model.parameters()).device
        self.alpha = nn.Parameter(torch.randn(self.weight_nums, device=device) / self.weight_nums)
        self.alpha_scale = torch.ones(1, device=device) * self.args.official_zoa_alpha_scale
        ln_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.LayerNorm)
            and module.weight is not None
            and module.weight.requires_grad
        ]
        self.fuse_modules = []
        modules_dict = dict(self.model.named_modules())
        for name, module in ln_modules:
            ln_weight_list = [sd[f"{name}.weight"].to(device) for sd in self.state_dicts]
            ln_bias_list = [sd[f"{name}.bias"].to(device) for sd in self.state_dicts]
            fuse_ln = OfficialZoaFuseLayerNorm(
                name,
                module.normalized_shape,
                module.eps,
                ln_weight_list,
                ln_bias_list,
                self.alpha,
                self.alpha_scale,
            ).to(device)
            self.fuse_modules.append(fuse_ln)
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                setattr(modules_dict[parent_name], child_name, fuse_ln)
            else:
                setattr(self.model, name, fuse_ln)
        if not self.fuse_modules:
            raise RuntimeError("Official ZOA did not find any trainable LayerNorm modules to fuse.")

    def configure_model(self):
        self.model.train()
        self.model.requires_grad_(False)
        self.alpha.requires_grad_(True)
        for module in self.model.modules():
            if isinstance(module, OfficialZoaFuseLayerNorm):
                module.epsilon_weight.requires_grad_(True)
                module.epsilon_bias.requires_grad_(True)

    def collect_params(self):
        params = {"epsilon_weight": [], "epsilon_bias": [], "alpha": [self.alpha], "alpha_scale": []}
        for module in self.model.modules():
            if isinstance(module, OfficialZoaFuseLayerNorm):
                params["epsilon_weight"].append(module.epsilon_weight)
                params["epsilon_bias"].append(module.epsilon_bias)
        self.params = params
        return params

    def parameters_to_vector(self):
        params = self.collect_params()
        alphas = nn.utils.parameters_to_vector(params["alpha"])
        delta_param = [
            param
            for layer_params in zip(params["epsilon_weight"], params["epsilon_bias"])
            for param in layer_params
        ]
        delta_param = nn.utils.parameters_to_vector(delta_param)
        return alphas, delta_param, self.embed_dim

    def vector_to_parameters(self, alphas, delta_param):
        params = self.collect_params()
        delta_params = [
            param
            for layer_params in zip(params["epsilon_weight"], params["epsilon_bias"])
            for param in layer_params
        ]
        nn.utils.vector_to_parameters(torch.cat([alphas, delta_param], dim=0), params["alpha"] + delta_params)

    def record_gradient(self, alphas_grad, delta_param_grad):
        params = self.collect_params()
        delta_params = [
            param
            for layer_params in zip(params["epsilon_weight"], params["epsilon_bias"])
            for param in layer_params
        ]
        grad_vector = torch.cat([alphas_grad, delta_param_grad], dim=0)
        pointer = 0
        for param in params["alpha"] + delta_params:
            num_param = param.numel()
            param.grad = grad_vector[pointer:pointer + num_param].view_as(param).detach()
            pointer += num_param

    @torch.no_grad()
    def forward(self, x):
        x = self.forward_features(x)
        return self.model.forward_head(x)

    def forward_features(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return x

    @torch.no_grad()
    def forward_embedding(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        return self.model.norm_pre(x)[:, 1:]

    def _collect_layers_features(self, x):
        cls_features = []
        for i in range(len(self.model.blocks)):
            x = self.model.blocks[i](x)
            if i < len(self.model.blocks) - 1:
                cls_features.append(self.model.blocks[i + 1].norm1(x[:, 0]))
            else:
                cls_features.append(self.model.norm(x[:, 0]))
        return torch.cat(cls_features, dim=1)

    def layers_cls_features(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        return self._collect_layers_features(x)

    def layers_cls_features_with_prompts(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        defer_memory_save = self.signature_store_hook is not None or self.memory_admission_hook is not None

        with torch.no_grad():
            embed_features = x[:, 1:].detach()
            domain_change = self._check_should_save(embed_features)
            if domain_change and not defer_memory_save:
                admit = True
                if self.memory_admission_hook is not None:
                    admit = bool(self.memory_admission_hook())
                if admit:
                    self.logger.info("Official ZOA domain change detected; saving domain vector.")
                    self.save_current_weight()
                else:
                    self.logger.info("Official ZOA domain change detected; QK-ZOA skipped redundant memory.")
                self.reset_domain_info()
            self._update_domain_info(embed_features)

        features = self._collect_layers_features(x)
        if domain_change and defer_memory_save:
            with torch.no_grad():
                admit = True
                if self.memory_admission_hook is not None:
                    admit = bool(self.memory_admission_hook())
                if admit:
                    self.logger.info("Official ZOA domain change detected; saving domain vector.")
                    self.save_current_weight()
                else:
                    self.logger.info("Official ZOA domain change detected; QK-ZOA skipped redundant memory.")
                self.reset_domain_info()

        return features, domain_change

    def alpha_distribution(self):
        with torch.no_grad():
            return F.softmax(self.alpha_logits(), dim=0).detach().cpu()

    def alpha_logits(self):
        logits = self.alpha * self.alpha_scale
        if self.alpha_prior_hook is not None:
            prior = self.alpha_prior_hook()
            if prior is not None:
                logits = logits + prior.to(device=logits.device, dtype=logits.dtype)
        return logits

    def plot_alpha(self, num=100):
        with torch.no_grad():
            alpha = self.alpha_distribution()
            tail = alpha if alpha.numel() <= num else alpha[-num:]
            dw = self.params["epsilon_weight"][0].abs().mean().item()
            db = self.params["epsilon_bias"][0].abs().mean().item()
        return f"Official ZOA alpha={tail.tolist()}, dw={dw:.6f}, db={db:.6f}"

    def set_optimizers(self, alpha_optimizer, eps_optimizer):
        self.alpha_optimizer = alpha_optimizer
        self.eps_optimizer = eps_optimizer

    def reset_domain_info(self):
        self.domain_var = None
        self.domain_mean = None

    @torch.no_grad()
    def _check_should_save(self, embed_features):
        if embed_features.shape[0] < 64:
            return False
        if self.domain_var is None or self.domain_mean is None:
            return False
        emb_var, emb_mean = embed_features.var(dim=(0, 1)), embed_features.mean(dim=(0, 1))
        return self._calculate_domain_shift(self.domain_var, self.domain_mean, emb_var, emb_mean) > self.args.official_zoa_domain_t

    @staticmethod
    @torch.no_grad()
    def _calculate_domain_shift(domain_var, domain_mean, cur_var, cur_mean, eps=1e-8):
        d1 = (domain_var + (domain_mean - cur_mean) ** 2) / (2.0 * (cur_var + eps)) - 0.5
        d2 = (cur_var + (domain_mean - cur_mean) ** 2) / (2.0 * (domain_var + eps)) - 0.5
        return torch.mean(d1 + d2)

    @torch.no_grad()
    def _update_domain_info(self, embed_features):
        if embed_features.shape[0] < 64:
            return
        emb_var, emb_mean = embed_features.var(dim=(0, 1)), embed_features.mean(dim=(0, 1))
        if self.domain_var is None:
            self.domain_var, self.domain_mean = emb_var, emb_mean
        else:
            self.domain_var = 0.8 * self.domain_var + 0.2 * emb_var
            self.domain_mean = 0.8 * self.domain_mean + 0.2 * emb_mean

    @torch.no_grad()
    def reset(self):
        self.weight_nums = 1
        device = self.alpha.device
        self.alpha = nn.Parameter(torch.randn(self.weight_nums, device=device) / self.weight_nums)
        self.alpha_scale = torch.ones(1, device=device) * self.args.official_zoa_alpha_scale
        for module in self.fuse_modules:
            module.reset(self.weight_nums, self.alpha, self.alpha_scale)
        params = self.collect_params()
        self.alpha_optimizer = torch.optim.AdamW(
            [
                {
                    "params": params["alpha"],
                    "lr": self.args.official_zoa_lr_alpha,
                    "weight_decay": self.args.official_zoa_weight_decay_alpha,
                }
            ]
        )
        self.eps_optimizer = torch.optim.SGD(
            params["epsilon_weight"] + params["epsilon_bias"],
            getattr(self.args, "official_zoa_lr_eps", 0.005),
            momentum=self.args.official_zoa_spsa_momentum,
            weight_decay=self.args.official_zoa_weight_decay,
        )
        self.reset_domain_info()

    @torch.no_grad()
    def _remove(self, idx):
        if self.memory_removal_hook is not None:
            self.memory_removal_hook(idx)
        self.weight_nums -= 1
        self.alpha = nn.Parameter(torch.cat([self.alpha.detach()[:idx], self.alpha.detach()[idx + 1:]]))
        self.logger.info(f"Official ZOA removing domain vector idx={idx - 1}, remaining={self.weight_nums - 1}")
        for module in self.fuse_modules:
            module.remove_idx(self.alpha, idx)

    def optimizes(self):
        if self.weight_nums > 1:
            self.alpha_optimizer.step()
        self.eps_optimizer.step()
        self.alpha_optimizer.zero_grad()
        self.eps_optimizer.zero_grad()
        self.alpha_optimizer.state = defaultdict(dict)


class OfficialZoaFuseLayerNorm(nn.Module):
    def __init__(self, name, normalized_shape, eps, weights, biases, alpha, alpha_scale):
        super().__init__()
        self.name = name
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight_nums = len(weights)
        self.origin_weight = weights[0].detach().clone()
        self.origin_bias = biases[0].detach().clone()
        self.weights = torch.stack(weights, dim=0).detach() - self.origin_weight
        self.biases = torch.stack(biases, dim=0).detach() - self.origin_bias
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.epsilon_weight = nn.Parameter(torch.zeros_like(self.weights[0]))
        self.epsilon_bias = nn.Parameter(torch.zeros_like(self.biases[0]))

    def get_fused_weights_and_biases(self):
        logits = self.alpha * self.alpha_scale
        prior = getattr(self, "alpha_prior", None)
        if torch.is_tensor(prior) and prior.numel() == logits.numel():
            logits = logits + prior.to(device=logits.device, dtype=logits.dtype)
        alphas_normalized = F.softmax(logits, dim=0).view(-1, 1)
        fuse_weight = self.origin_weight + (self.weights * alphas_normalized).sum(0) + self.epsilon_weight
        fuse_bias = self.origin_bias + (self.biases * alphas_normalized).sum(0) + self.epsilon_bias
        return fuse_weight, fuse_bias

    @torch.no_grad()
    def get_current_delta_weights_and_biases(self):
        logits = self.alpha * self.alpha_scale
        prior = getattr(self, "alpha_prior", None)
        if torch.is_tensor(prior) and prior.numel() == logits.numel():
            logits = logits + prior.to(device=logits.device, dtype=logits.dtype)
        alphas_normalized = F.softmax(logits, dim=0).view(-1, 1)
        delta_weight = (self.weights * alphas_normalized).sum(0) + self.epsilon_weight
        delta_bias = (self.biases * alphas_normalized).sum(0) + self.epsilon_bias
        return delta_weight, delta_bias

    def forward(self, x):
        fuse_weight, fuse_bias = self.get_fused_weights_and_biases()
        return F.layer_norm(x, self.normalized_shape, fuse_weight, fuse_bias, self.eps)

    @torch.no_grad()
    def save_current_weight(self, cur_alpha):
        self.weight_nums += 1
        cur_weight, cur_bias = self.get_fused_weights_and_biases()
        self.weights = torch.cat([self.weights, (cur_weight - self.origin_weight).unsqueeze(0)], dim=0)
        self.biases = torch.cat([self.biases, (cur_bias - self.origin_bias).unsqueeze(0)], dim=0)
        self.alpha = cur_alpha
        new_weight, new_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_weight
        self.epsilon_bias += cur_bias - new_bias

    @torch.no_grad()
    def remove_idx(self, cur_alpha, remove_idx):
        self.weight_nums -= 1
        cur_weight, cur_bias = self.get_fused_weights_and_biases()
        self.weights = torch.cat([self.weights[:remove_idx], self.weights[remove_idx + 1:]], dim=0)
        self.biases = torch.cat([self.biases[:remove_idx], self.biases[remove_idx + 1:]], dim=0)
        self.alpha = cur_alpha
        new_weight, new_bias = self.get_fused_weights_and_biases()
        self.epsilon_weight += cur_weight - new_weight
        self.epsilon_bias += cur_bias - new_bias

    @torch.no_grad()
    def reset(self, weight_nums, alpha, alpha_scale):
        self.weight_nums = weight_nums
        self.weights = self.weights[0:1]
        self.biases = self.biases[0:1]
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.epsilon_weight = nn.Parameter(torch.zeros_like(self.weights[0]))
        self.epsilon_bias = nn.Parameter(torch.zeros_like(self.biases[0]))
