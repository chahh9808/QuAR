"""TTA on the MQBench-quantized ResNet-50, the artefact ZOA actually publishes on.

Why a separate runner. The PTQ4ViT path puts quantization inside the conv modules,
so QuAR pre-hooks the convs. MQBench FX-traces the model and inserts standalone
`*_post_act_fake_quantizer` modules, so QuAR pre-hooks those directly. That is a
closer match to how the paper describes the method, "the tensor entering a frozen
quantizer", since here the quantizer really is its own module.

Scope. 8 activation quantizers inside layer4 form the local branch; the classifier
side quantizer `reshape_post_act_fake_quantizer`, which sits on the pooled feature
just before `fc`, is the global branch and runs at lambda 1.0, mirroring the
launcher's --quar_global_alpha 1.0.

The paper's the local branch code is reused unchanged. Two runtime patches are imported from
run_resnet_gn_quar_pilot (channel-axis handling for 4-D inputs, per-module lambda);
the module selector is redefined here because MQBench quantizers are not
recognised by _is_quant_act, which looks for PTQ4ViT/AdaLog
attributes. quar_local_mode is 'explicit', which never inspects quantizer internals, so
the correction itself is quantizer-agnostic.

Verified on this artefact: 50 activation fake-quantizers, BatchNorm NOT folded
(53 modules remain, so AdaBN-style baselines still apply), `.fc` present.
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

sys.path.insert(0, _REPO_ROOT)

from tta_library.quar import QuARLocalMixin, build_no_adapt_qac_wrapper  # noqa: E402
from tools.stats.build_resnet_mqbench import rebuild_mqbench_model  # noqa: E402
from tools.analysis.run_resnet_gn_quar_pilot import (  # noqa: E402
    install_nchw_patch, corruption_loader, quar_args, evaluate,
)

LOCAL_PREFIX = "layer4"
# The classifier-side quantizer sits on the pooled feature just before `fc`. Its
# traced name depends on how the backbone pools: torchvision reshapes, timm uses
# global_pool + flatten.
GLOBAL_Q_CANDIDATES = ("reshape_post_act_fake_quantizer",
                       "global_pool_flatten_post_act_fake_quantizer")
GLOBAL_Q = GLOBAL_Q_CANDIDATES[0]
GLOBAL_ALPHA = 1.0


def resolve_global_q(net):
    global GLOBAL_Q
    names = {n for n, _ in net.named_modules()}
    for c in GLOBAL_Q_CANDIDATES:
        if c in names:
            GLOBAL_Q = c
            return c
    raise RuntimeError(f"no classifier-side quantizer found among {GLOBAL_Q_CANDIDATES}")


def is_mqbench_act_quant(m):
    return (hasattr(m, "scale") and hasattr(m, "zero_point")
            and hasattr(m, "activation_post_process"))


def install_mqbench_scope(include_global=True):
    def _select(self):
        mods = []
        for name, module in self.model.model.named_modules():
            if not name.endswith("post_act_fake_quantizer"):
                continue
            if not is_mqbench_act_quant(module):
                continue
            if name.startswith(LOCAL_PREFIX) or (include_global and name == GLOBAL_Q):
                mods.append((name, module))
        return mods
    QuARLocalMixin._quar_select_modules = _select
    return _select


def install_global_alpha():
    impl = QuARLocalMixin._make_qac_hook_impl

    def _make(self, name):
        inner = impl(self, name)
        if name != GLOBAL_Q:
            return inner

        def wrapped(module, inputs):
            saved = self._quar_alpha
            self._quar_alpha = GLOBAL_ALPHA
            try:
                return inner(module, inputs)
            finally:
                self._quar_alpha = saved
        return wrapped
    QuARLocalMixin._make_qac_hook = _make


@torch.no_grad()
def collect_stats(net, names, data_root, batch_size, n_batches, out_path):
    from quant_library.quant_utils import datasets as qdatasets
    mods = dict(net.named_modules())
    acc = {n: {"s": None, "sq": None, "c": 0} for n in names}
    handles = []

    def mk(name):
        def pre(module, inputs):
            x = inputs[0].detach().float()
            xc = (x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) if x.dim() == 4
                  else x.reshape(-1, x.shape[-1]))
            a = acc[name]
            s, sq = xc.sum(0), (xc * xc).sum(0)
            a["s"] = s if a["s"] is None else a["s"] + s
            a["sq"] = sq if a["sq"] is None else a["sq"] + sq
            a["c"] += xc.shape[0]
        return pre

    for n in names:
        handles.append(mods[n].register_forward_pre_hook(mk(n)))
    g = qdatasets.ImageNetLoaderGenerator(data_root, 'imagenet', batch_size, batch_size, 8)
    for i, (x, _) in enumerate(g.test_loader()):
        if i >= n_batches:
            break
        net(x.cuda(non_blocking=True))
        if (i + 1) % 50 == 0:
            print(f"  [stats] {i+1}/{n_batches}", flush=True)
    for h in handles:
        h.remove()
    mean_d, std_d = {}, {}
    for n, a in acc.items():
        c = max(a["c"], 1)
        mu = a["s"] / c
        var = (a["sq"] / c - mu * mu).clamp(min=0.0)
        mean_d[n], std_d[n] = mu.cpu(), var.sqrt().cpu()
    torch.save({"mean": mean_d, "std": std_d}, out_path)
    print(f"[stats] saved {len(mean_d)} interfaces to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['stats', 'eval'], required=True)
    ap.add_argument('--ckpt', default='checkpoints/resnet50_mqbench_w6a6_sd.pt')
    ap.add_argument('--bit', type=int, default=6)
    ap.add_argument('--arch', default='zoa_resnet50')
    ap.add_argument('--data', default=os.environ.get("DATA", ""))
    ap.add_argument('--data_corruption', default=os.environ.get("DATA_C", ""))
    ap.add_argument('--id_file', default='robustbench/data/imagenet_test_image_ids.txt')
    ap.add_argument('--stats_file', default='stats/resnet50_mqbench_w6a6_srcstats.pt')
    ap.add_argument('--stats_batches', type=int, default=200)
    ap.add_argument('--algorithm', default='no_adapt',
                    choices=['no_adapt', 'quar', 't3a', 'neo', 'bn_adapt', 'quar_bnadapt'])
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--corruptions', default='gaussian_noise')
    ap.add_argument('--severity', type=int, default=5)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--t3a_filter_k', type=int, default=20)
    ap.add_argument('--no_global', action='store_true',
                    help='local branch only, for the component ablation')
    ap.add_argument('--bn_batch_stats', action='store_true',
                    help='switch BatchNorm to batch statistics (AdaBN). Also honoured in '
                         'stats mode, so source moments can be collected in the SAME BN '
                         'configuration they will be applied in.')
    args = ap.parse_args()

    install_nchw_patch()
    net = rebuild_mqbench_model(args.ckpt, bit=args.bit, arch=args.arch)
    net.requires_grad_(False)
    resolve_global_q(net)
    install_global_alpha()
    select = install_mqbench_scope(include_global=not args.no_global)

    import types as _t
    names = [n for n, _ in select(_t.SimpleNamespace(model=_t.SimpleNamespace(model=net)))]
    print(f"[scope] {len(names)} quantizers "
          f"({sum(1 for n in names if n != GLOBAL_Q)} local, "
          f"{sum(1 for n in names if n == GLOBAL_Q)} global at lambda={GLOBAL_ALPHA})")

    def _batch_stat_bn(m):
        n = 0
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.track_running_stats = False
                mod.running_mean = None
                mod.running_var = None
                n += 1
        print(f"[bn_adapt] {n} BatchNorm2d switched to batch statistics")

    if args.bn_batch_stats or args.algorithm in ('bn_adapt', 'quar_bnadapt'):
        _batch_stat_bn(net)

    if args.mode == 'stats':
        collect_stats(net, names, args.data, args.batch_size, args.stats_batches,
                      args.stats_file)
        return

    if args.algorithm == 'never_taken_bn_adapt':
        n = 0
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                n += 1
        print(f"[bn_adapt] {n} BatchNorm2d switched to batch statistics")
    elif args.algorithm == 't3a':
        from tta_library.t3a import T3A, configure_model as t3a_cfg
        net = t3a_cfg(net)
        net = T3A(_t.SimpleNamespace(arch='resnet50'), net, 1000, args.t3a_filter_k).cuda()
        net.imagenet_mask = None
        print(f"[t3a] filter_K={args.t3a_filter_k} (also enables batch-stat BN)")
    elif args.algorithm == 'neo':
        from tta_library.neo_tta import NEOWrapper
        net.head = net.fc
        net = NEOWrapper(net).cuda()
        print("[neo] cumulative mean subtraction at the classifier input")

    holder = None
    if args.algorithm in ('quar', 'quar_bnadapt'):
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        holder = build_no_adapt_qac_wrapper(
            quar_args(args.stats_file, 'mqbench_layer4_fc', args.alpha),
            net, logging.getLogger('quar'))
        print(f"[quar] lambda_local={args.alpha} lambda_global={GLOBAL_ALPHA}")

    results = {}
    for corr in args.corruptions.split(','):
        if holder is not None and hasattr(holder, 'reset'):
            holder.reset()
        tfm_arch = 'resnet50_gn' if args.arch == 'resnet50_gn' else 'resnet50.tv_in1k'
        loader = corruption_loader(args.data_corruption, corr, args.severity,
                                   tfm_arch, args.batch_size,
                                   args.id_file, seed=args.seed)
        acc = evaluate(net, loader, holder)
        results[corr] = acc
        print(f"  {corr:18s} top-1 = {acc:.2f}", flush=True)
    mean = sum(results.values()) / len(results)
    tag = f"{args.algorithm}(lam={args.alpha})" if args.algorithm == 'quar' else args.algorithm
    print(f"\n===== {tag} | MQBench W{args.bit}A{args.bit} ResNet-50 | "
          f"mean over {len(results)} corruptions = {mean:.2f} =====")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
