"""QuAR on quantized ResNet-50 backbones: shared helpers and a PTQ4ViT-stack runner.

`run_resnet_mqbench_tta.py`, which produces the CNN appendix (MQBench-quantized GroupNorm and
BatchNorm ResNet-50), imports `install_nchw_patch`, `corruption_loader`, `quar_args` and
`evaluate` from here. The paper's local-branch implementation is reused, not reimplemented, so
the estimator, the clamp and the lambda blending are the ones behind the ViT numbers. Two things
are patched at runtime, in this file only, so that the ViT code path is untouched:

  1. QuARLocalMixin._quar_select_modules. The ViT selector filters on `blocks.{i}.` and returns
     an empty list for ResNet module names. Replaced by a stage-based selector.
  2. QuARLocalMixin._quar_correct. The ViT path does `x.reshape(-1, x.shape[-1])`, i.e. it
     assumes channel-last. A conv input is [B, C, H, W], so shape[-1] is the spatial width and
     the statistics would be taken over the wrong axis. The patch permutes 4-D inputs to
     channel-last, calls the original, and permutes back. 2-D and 3-D inputs are untouched.

Run directly, this file evaluates QuAR on a ResNet-50 quantized with the PTQ4ViT stack
(--mode stats collects source statistics on clean ImageNet, --mode eval runs ImageNet-C).
"""
# Repository root (this tool lives one level below it).
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

sys.path.insert(0, _REPO_ROOT)
from quant_library.quant_utils import datasets as qdatasets  # noqa: E402
from tta_library.quar import QuARLocalMixin, build_no_adapt_qac_wrapper, _is_quant_act, _block_idx  # noqa: E402

# ───────────────────────────── scope

# ResNet-50 has 53 convs: stem(1) + layer1(10) + layer2(13) + layer3(19) + layer4(10).
SCOPES = {
    "layer4":      lambda n: n.startswith("layer4"),
    "layer3_4":    lambda n: n.startswith("layer3") or n.startswith("layer4"),
    "layer4_last": lambda n: n.startswith("layer4.1") or n.startswith("layer4.2"),
    "all_conv":    lambda n: True,
    # Full QuAR: local branch on the late conv interfaces PLUS the global branch
    # at the classifier input, the ResNet analogue of the CLS branch at the head.
    "layer4_fc":   lambda n: n.startswith("layer4") or n == "fc",
}

# The paper runs the global branch at full strength while the local branch is
# partial (launcher: --quar_global_alpha 1.0 against --quar_local_lambda 0.5). cls_shift.py
# uses the identical operator, "(h - mu_t)/(sig_t + eps) * sig_s + mu_s" blended
# by "(1 - alpha) * h + alpha * h_shifted" with clip_ratio 3.0, so realising the
# global branch through the same local-branch hook at the `fc` input is faithful.
GLOBAL_BRANCH_ALPHA = 1.0
GLOBAL_BRANCH_MODULE = "fc"


def install_resnet_scope(scope_name):
    """Patch 1: select conv interfaces by stage instead of by `blocks.{i}.`."""
    keep = SCOPES[scope_name]

    def _select(self):
        mods = []
        for name, module in self.model.model.named_modules():
            if not _is_quant_act(module):
                continue
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if keep(name):
                mods.append((name, module))
        return mods

    QuARLocalMixin._quar_select_modules = _select
    return _select


def install_global_branch_alpha():
    """Patch 3: give the classifier interface its own lambda.

    Upstream _make_qac_hook can already override lambda per hook, but it keys off
    a `blocks.(\d+)` regex, so it never fires on ResNet names. Same mechanism,
    keyed by module name instead.
    """
    impl = QuARLocalMixin._make_qac_hook_impl

    def _make(self, name):
        inner = impl(self, name)
        if name != GLOBAL_BRANCH_MODULE:
            return inner

        def wrapped(module, inputs):
            saved = self._quar_alpha
            self._quar_alpha = GLOBAL_BRANCH_ALPHA
            try:
                return inner(module, inputs)
            finally:
                self._quar_alpha = saved
        return wrapped

    QuARLocalMixin._make_qac_hook = _make


def install_nchw_patch():
    """Patch 2: make _quar_correct channel-axis aware for 4-D conv inputs."""
    orig = QuARLocalMixin._quar_correct

    def _correct(self, x, name, src):
        if x.dim() == 4:                       # [B, C, H, W]
            xc = x.permute(0, 2, 3, 1)         # -> [B, H, W, C], channel last
            out = orig(self, xc, name, src)
            return out.permute(0, 3, 1, 2).contiguous()
        return orig(self, x, name, src)

    QuARLocalMixin._quar_correct = _correct


# ───────────────────────────── model / data

def load_quant_net(ckpt):
    net = torch.load(ckpt, map_location='cpu', weights_only=False).cuda().eval()
    for m in net.modules():
        if hasattr(m, 'mode'):
            m.mode = 'quant_forward'
        for a in ('w_interval', 'a_interval', 'A_interval', 'B_interval'):
            t = getattr(m, a, None)
            if isinstance(t, torch.Tensor) and not t.is_cuda:
                setattr(m, a, t.cuda())
    net.requires_grad_(False)
    return net


def transform_for(arch):
    from timm.data import resolve_data_config, create_transform
    return create_transform(**resolve_data_config({}, model=timm.create_model(arch)))


def corruption_loader(root, corruption, severity, arch, batch_size, id_file=None, workers=8, seed=None):
    """ImageNet-C ImageFolder, optionally restricted to the RobustBench 5k subset.

    ImageFolder sorts class directories, which is the standard wnid order, so the
    integer labels line up with the model's output indices.
    """
    path = os.path.join(root, corruption, str(severity))
    ds = ImageFolder(path, transform_for(arch))
    if id_file:
        wanted = {l.strip() for l in open(id_file) if l.strip()}
        keep = [i for i, (p, _) in enumerate(ds.samples)
                if os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)) in wanted]
        ds = Subset(ds, keep)
    if seed is None:
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    gen = torch.Generator(); gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen,
                      num_workers=workers, pin_memory=True)


# ───────────────────────────── stats

@torch.no_grad()
def collect_source_stats(net, module_names, data_root, arch, batch_size, n_batches, out_path):
    """Per-channel mean/std of the tensor entering each selected quantizer.

    Reduced over (batch, height, width), the conv analogue of the ViT reduction
    over (batch, tokens). Source data is clean ImageNet val, as for the ViT runs.
    """
    mods = dict(net.named_modules())
    acc = {n: {"sum": None, "sqsum": None, "count": 0} for n in module_names}
    handles = []

    def mk(name):
        def pre(module, inputs):
            x = inputs[0].detach().float()
            xc = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) if x.dim() == 4 \
                else x.reshape(-1, x.shape[-1])
            a = acc[name]
            s, sq, c = xc.sum(0), (xc * xc).sum(0), xc.shape[0]
            a["sum"] = s if a["sum"] is None else a["sum"] + s
            a["sqsum"] = sq if a["sqsum"] is None else a["sqsum"] + sq
            a["count"] += c
        return pre

    for n in module_names:
        handles.append(mods[n].register_forward_pre_hook(mk(n)))

    g = qdatasets.ViTImageNetLoaderGenerator(data_root, 'imagenet', batch_size, batch_size, 8,
                                             kwargs={"model": timm.create_model(arch)})
    loader = g.test_loader()
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        net(x.cuda(non_blocking=True))
        if (i + 1) % 50 == 0:
            print(f"  [stats] {i+1}/{n_batches} batches", flush=True)
    for h in handles:
        h.remove()

    mean_d, std_d = {}, {}
    for n, a in acc.items():
        c = max(a["count"], 1)
        mu = a["sum"] / c
        var = (a["sqsum"] / c - mu * mu).clamp(min=0.0)
        mean_d[n], std_d[n] = mu.cpu(), var.sqrt().cpu()
    torch.save({"mean": mean_d, "std": std_d}, out_path)
    print(f"[stats] saved {len(mean_d)} interfaces to {out_path}")


# ───────────────────────────── args namespace

def quar_args(stats_file, scope, alpha):
    """Mirror of the deployed launcher flags (run_bs_sweep.sh OURS block)."""
    import types
    return types.SimpleNamespace(
        quar_local_enable=True, quar_local_mode='explicit', quar_local_fold_type='scale_only',
        quar_local_scope=scope, quar_local_stats='mean_std', quar_local_eps=1e-6, quar_local_clip_ratio=3.0,
        quar_local_lambda=alpha, quar_local_stats_file=stats_file,
        quar_local_stream_ema=True, quar_local_stream_alpha=0.05, quar_local_stream_nwarm=0, quar_local_stream_lag=True,
        recal_cumulative=True, quar_local_estimator='current',
        quar_local_use_ema=False, quar_local_ema_momentum=0.1, quar_local_fp_mode=False,
        quar_local_random_scope_seed=0, quar_local_exclude_layers=None,
        quar_local_quant_guidance='none', quar_local_code_metric='l1', quar_local_code_bins=0,
        quar_local_alpha_max=1.0, quar_local_alpha_min=0.0, quar_local_guidance_temperature=1.0,
        quar_local_alpha_profile='', quar_mismatch_diag=False,
        quar_mismatch_diag_scope=scope,
    )


# ───────────────────────────── eval

@torch.no_grad()
def evaluate(net, loader, holder=None):
    # QuARLocalMixin.forward is what flips _quar_active, so the correction only fires
    # when the call goes through the wrapper rather than the raw net.
    fwd = holder if holder is not None else net
    correct = total = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        correct += (fwd(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['stats', 'eval'], required=True)
    ap.add_argument('--ckpt', required=True, help='PTQ4ViT-stack quantized ResNet checkpoint')
    ap.add_argument('--arch', default='resnet50_gn')
    ap.add_argument('--data', default=os.environ.get("DATA", ""))
    ap.add_argument('--data_corruption', default=os.environ.get("DATA_C", ""))
    ap.add_argument('--id_file', default='robustbench/data/imagenet_test_image_ids.txt')
    ap.add_argument('--scope', default='layer4', choices=list(SCOPES))
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--stats_batches', type=int, default=200)
    ap.add_argument('--stats_file', default='')
    ap.add_argument('--corruptions', default='gaussian_noise,contrast,snow')
    ap.add_argument('--severity', type=int, default=5)
    ap.add_argument('--quar', action='store_true', help='shorthand for --algorithm quar')
    ap.add_argument('--algorithm', default='',
                    choices=['', 'no_adapt', 'quar', 't3a', 'neo', 'bn_adapt', 'quar_bnadapt'],
                    help='bn_adapt is AdaBN alone. T3A and ZOA call a configure_model that '
                         'switches BatchNorm to batch statistics, so on a BN backbone they get '
                         'AdaBN for free; bn_adapt and quar_bnadapt isolate that contribution.')
    ap.add_argument('--t3a_filter_k', type=int, default=20)
    ap.add_argument('--seed', type=int, default=None,
                    help='shuffles the test stream, as main.py does with if_shuffle=True; '
                         'QuAR is order-dependent through its online estimator')
    args = ap.parse_args()

    stats_file = args.stats_file or f"stats/resnet50gn_ptq4vit_w6a6_srcstats_{args.scope}.pt"
    install_nchw_patch()
    install_global_branch_alpha()
    select = install_resnet_scope(args.scope)
    net = load_quant_net(args.ckpt)

    import types as _t
    names = [n for n, _ in select(_t.SimpleNamespace(model=_t.SimpleNamespace(model=net)))]
    n_glob = sum(1 for n in names if n == GLOBAL_BRANCH_MODULE)
    print(f"[scope] '{args.scope}' selects {len(names)} interfaces "
          f"({len(names)-n_glob} local conv, {n_glob} global at '{GLOBAL_BRANCH_MODULE}' "
          f"with lambda={GLOBAL_BRANCH_ALPHA})")
    print(f"        first {names[:2]} ... last {names[-2:]}")

    if args.mode == 'stats':
        collect_source_stats(net, names, args.data, args.arch, args.batch_size,
                             args.stats_batches, stats_file)
        return

    algo = args.algorithm or ('quar' if args.quar else 'no_adapt')

    def _to_batch_stat_bn(m):
        # Exactly what t3a.configure_model / zoa_resnet.configure_model do.
        n = 0
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.track_running_stats = False
                mod.running_mean = None
                mod.running_var = None
                n += 1
        print(f"[bn_adapt] switched {n} BatchNorm2d layers to batch statistics")
        return m

    if algo in ('bn_adapt', 'quar_bnadapt'):
        _to_batch_stat_bn(net)
    if algo == 't3a':
        from tta_library.t3a import T3A, configure_model as t3a_cfg
        import types as _tt
        net = t3a_cfg(net)
        net = T3A(_tt.SimpleNamespace(arch='resnet50'), net, 1000, args.t3a_filter_k).cuda()
        net.imagenet_mask = None  # main.py's eval loop sets this per corruption
        print(f"[t3a] filter_K={args.t3a_filter_k} (its configure_model also enables batch-stat BN)")
    elif algo == 'neo':
        from tta_library.neo_tta import NEOWrapper
        # NEO hooks `net.head`; a ResNet classifier is `net.fc`. Aliasing points the
        # hook at the same module object, so the correction fires on the real forward.
        net.head = net.fc
        net = NEOWrapper(net).cuda()
        print("[neo] cumulative mean subtraction at the classifier input")

    holder = None
    if algo in ('quar', 'quar_bnadapt'):
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        holder = build_no_adapt_qac_wrapper(
            quar_args(stats_file, args.scope, args.alpha), net, logging.getLogger('quar'))
        print(f"[quar] scope={args.scope} lambda={args.alpha} stats={stats_file}")

    results = {}
    for corr in args.corruptions.split(','):
        if holder is not None and hasattr(holder, 'reset'):
            holder.reset()
        loader = corruption_loader(args.data_corruption, corr, args.severity,
                                   args.arch, args.batch_size, args.id_file, seed=args.seed)
        acc = evaluate(net, loader, holder)
        results[corr] = acc
        print(f"  {corr:18s} top-1 = {acc:.2f}", flush=True)
    mean = sum(results.values()) / len(results)
    tag = f"{algo}(scope={args.scope},lam={args.alpha})" if algo.startswith('quar') else algo
    print(f"\n===== {tag} | W6A6 resnet50_gn | mean over {len(results)} corruptions = {mean:.2f} =====")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
