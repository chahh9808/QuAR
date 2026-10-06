import os
import sys
import time
import math
import argparse
import random
from importlib import reload, import_module
import copy

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data, calib_transform_override
from dataset.ImageNetMask import imagenet_r_mask, imagenet_a_mask

import torch    
import timm
import numpy as np
from tta_library.fozo import FOZO
from tta_library.foa import FOA  # official CMA-ES FOA (Niu et al., ICML 2024)
from tta_library.zoa import ZOAOfficial, configure_model_for_official_zoa
from tta_library.quar import NoAdaptQACWrapper, build_no_adapt_qac_wrapper
from tta_library.quar.global_cls import build_cls_shift_wrapper
from tta_library.neo_tta import build_neo_wrapper
from calibration_library.metrics import ECELoss

from quant_library.quant_utils.models import get_net, create_backbone
from quant_library.quant_utils import net_wrap
import quant_library.quant_utils.datasets as datasets
from quant_library.quant_utils.quant_calib import HessianQuantCalibrator

from models.vpt import PromptViT
from models.fuse_vit_official_zoa import OfficialZoaFuseViT


def validate_adapt(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')  
    top1 = AverageMeter('Acc@1', ':6.2f')  
    top5 = AverageMeter('Acc@5', ':6.2f')  
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')  
    
    outputs_list, targets_list = [], []
    _flop_samples = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.gpu)


    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    forward_times = []
    _latprof = getattr(args, 'latency_profile', False)
    lat_fwd_ms = []   # per-batch forward times (batches>=2) for --latency_profile

    with torch.no_grad(): 
        end = time.time()  
        for i, dl in enumerate(val_loader): 
            images, target = dl[0], dl[1] 
            images = images.cuda()  
            target = target.cuda()  
            signature_logger = getattr(model, "quant_signature_logger", None)
            if signature_logger is not None:
                signature_logger.begin_batch(getattr(args, "corruption", ""), i + 1)
            if i < 2:
                start_events[i].record()
                output = model(images)
                end_events[i].record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_events[i].elapsed_time(end_events[i])
                forward_times.append(elapsed_time_ms)
                logger.info(f"Forward pass {i+1} time: {elapsed_time_ms:.3f} ms")
            else:
                if getattr(args, 'measure_flops', False) and 2 <= i <= 4:
                    from torch.profiler import profile as _tprof, ProfilerActivity as _PA
                    with _tprof(activities=[_PA.CPU, _PA.CUDA], with_flops=True) as _pf:
                        output = model(images)
                    _ftot = sum((getattr(e, 'flops', 0) or 0) for e in _pf.key_averages())
                    _flop_samples.append(_ftot / max(1, images.size(0)))
                elif _latprof:
                    _se = torch.cuda.Event(enable_timing=True); _ee = torch.cuda.Event(enable_timing=True)
                    _se.record(); output = model(images); _ee.record(); torch.cuda.synchronize()
                    lat_fwd_ms.append(_se.elapsed_time(_ee))
                else:
                    output = model(images)
            if signature_logger is not None:
                signature_logger.end_batch(output)

            # -----------------------------------------------------------------
            # For ImageNet-R: model.imagenet_mask is set by the per-corruption loop.
            # If the model returned full 1000-dim logits (NoAdapt, NEO, Ours, FOZO,
            # ZOA, and — after the T3A patch — also T3A), apply the mask so
            # output[:,k] aligns with the ImageFolder 200-class labels. Wrappers
            # that already mask internally (FOA) return smaller-dim output, so we
            # only mask when output is still 1000-dim.
            _raw_out_dim = output.shape[-1]
            _mask_applied = False
            _mask = getattr(model, 'imagenet_mask', None)
            if _mask is not None:
                _mlen = _mask.shape[0] if hasattr(_mask, 'shape') else len(_mask)
                if output.shape[-1] == _mlen:
                    if not isinstance(_mask, torch.Tensor):
                        _mask = torch.as_tensor(_mask, dtype=torch.bool, device=output.device)
                    output = output[:, _mask]
                    _mask_applied = True
            # AUDIT: log dim handling on first batch
            if i == 0 and _mask is not None:
                _tmin, _tmax = int(target.min().item()), int(target.max().item())
                logger.info(
                    f"[R-audit] algo={getattr(args, 'algorithm', None)} "
                    f"raw_out_dim={_raw_out_dim} masked={_mask_applied} "
                    f"final_out_dim={output.shape[-1]} "
                    f"target_range=[{_tmin},{_tmax}]"
                )
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            del output

            batch_time.update(time.time() - end)  
            end = time.time()  
            if i % 5 == 0:  
                logger.info(progress.display(i))  
            if args.max_eval_batches > 0 and (i + 1) >= args.max_eval_batches:
                break
            
        outputs_list = torch.cat(outputs_list, dim=0).numpy()  
        targets_list = torch.cat(targets_list, dim=0).numpy()  
        
        logits = args.algorithm != 'lame'  
        ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) 

    if len(forward_times) >= 2:
        logger.info(f"Summary: 1st forward: {forward_times[0]:.3f} ms, 2nd forward: {forward_times[1]:.3f} ms")
    steady_forward_ms = forward_times[1] if len(forward_times) >= 2 else (forward_times[0] if forward_times else float("nan"))

    if _latprof and lat_fwd_ms:
        import statistics as _stp
        _bs = getattr(args, 'batch_size', 1) or 1
        _med = _stp.median(lat_fwd_ms); _mean = sum(lat_fwd_ms) / len(lat_fwd_ms)
        logger.info(f"LATPROF n_batches={len(lat_fwd_ms)} median_fwd_ms={_med:.3f} mean_fwd_ms={_mean:.3f} "
                    f"median_per_image_ms={_med/_bs:.4f} mean_per_image_ms={_mean/_bs:.4f}")

    try:
        _peak_mb = torch.cuda.max_memory_allocated(args.gpu) / (1024.0 ** 2)
        logger.info(f"PEAK_GPU_MEM_MB={_peak_mb:.2f}")
    except Exception as _e:
        logger.info(f"PEAK_GPU_MEM_MB=nan ({_e})")
    if getattr(args, 'measure_flops', False) and _flop_samples:
        import statistics as _st
        logger.info(f"MEASURED_GFLOPS_PER_IMG={_st.median(_flop_samples)/1e9:.4f} (n={len(_flop_samples)})")

    return top1.avg, top5.avg, ece_avg, steady_forward_ms

def obtain_train_loader(args):
    """Get the training set loader"""
    temp_args = copy.deepcopy(args)
    
    # Ensure obtain_train_loader always loads the complete ImageNet original validation set
    # Temporarily set continual to False and corruption to 'original'
    temp_args.continual = False
    temp_args.imagenet_5k_eval = False
    temp_args.corruption = 'original' 
    temp_args.source_dir_split = getattr(args, 'source_split', 'val')
    
    # Call prepare_test_data to get the complete ImageNet validation set
    train_dataset, train_loader = prepare_test_data(temp_args) 
    train_dataset.switch_mode(True, False) 
    return train_dataset, train_loader


def build_official_zoa_adapter(args, net, log_dir, enable_qkzoa=False, enable_diag=False):
    """Build the official ZOA memory path, optionally with QK-ZOA hooks or diagnostics."""
    # Footgun guard: ZOA does NOT read --lr (it uses --official_zoa_lr_alpha/--official_zoa_lr_eps);
    # its stability hinges on a bitwidth-matched source-stat file. The generic default
    # 'official_zoa_train_stats.pt' silently mis-anchors the domain-shift term and diverges
    # over long streams. ZOA must also run EPISODICALLY (reset per corruption); it diverges on
    # full-data streams (~79 batches/corruption) even with the correct config.
    _zsf = str(getattr(args, "official_zoa_stats_file", "official_zoa_train_stats.pt"))
    if _zsf.endswith("official_zoa_train_stats.pt") and getattr(args, "quant", False):
        logger.warning(
            "[ZOA] --official_zoa_stats_file is the generic default ('%s'); for quantized runs "
            "pass a bitwidth-matched file (e.g. official_zoa_train_stats_w%sa%s_seed%s.pt). Note: "
            "ZOA ignores --lr and diverges on full-data streams; run episodically.",
            _zsf, getattr(args, "quant_w_bit", "?"), getattr(args, "quant_a_bit", "?"),
            getattr(args, "seed", "?"),
        )
    configured_net = configure_model_for_official_zoa(
        net, fix_paper_layers=not getattr(args, "official_zoa_all_ln", False))
    fuse_model = OfficialZoaFuseViT(args, configured_net, logger)
    fuse_model.replace_fuse_ln()
    fuse_model.configure_model()
    params = fuse_model.collect_params()
    alpha_optimizer = torch.optim.AdamW(
        [
            {
                "params": params["alpha"],
                "lr": args.official_zoa_lr_alpha,
                "weight_decay": args.official_zoa_weight_decay_alpha,
            }
        ]
    )
    eps_optimizer = torch.optim.SGD(
        params["epsilon_weight"] + params["epsilon_bias"],
        getattr(args, "official_zoa_lr_eps", 0.005),
        momentum=args.official_zoa_spsa_momentum,
        weight_decay=args.official_zoa_weight_decay,
    )
    fuse_model.set_optimizers(alpha_optimizer, eps_optimizer)
    adapter_cls = ZOAOfficial
    kwargs = {}
    adapt_model = adapter_cls(
        args,
        fuse_model,
        fitness_lambda=args.official_zoa_lambda_bp,
        log_dir=log_dir,
        **kwargs,
    )
    _, train_loader = obtain_train_loader(args)
    adapt_model.obtain_origin_stat(
        train_loader,
        stats_file=args.official_zoa_stats_file,
        max_batches=args.official_zoa_max_source_stat_batches,
    )
    return adapt_model

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./quant_library/configs"))  
    if config_name+".py" in files:  
        quant_cfg = import_module(f"quant_library.configs.{config_name}")  
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")  
    reload(quant_cfg)  
    return quant_cfg 

def get_args():
    """Get arguments"""
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')  # Create argument parser

    # path of data, output dir
    # Data path, output directory
    parser.add_argument('--data', default='/root/autodl-tmp/ILSVRC2012_img_val', help='path to dataset')
    parser.add_argument('--data_v2', default='/root/autodl-tmp/ImageNetV2', help='path to dataset')
    parser.add_argument('--data_sketch', default='/root/autodl-tmp/ImageNet-Sketch', help='path to dataset')
    parser.add_argument('--data_corruption', default='/root/autodl-tmp/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--data_rendition', default='/root/autodl-tmp/imagenet-r', help='path to corruption dataset')
    parser.add_argument('--data_a', default=os.environ.get('DATA_A', ''), help='path to ImageNet-A dataset')
    # Non-ImageNet datasets (additive). 'imagenet' keeps every existing path byte-identical.
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar10', 'cifar100', 'domainnet126'],
                        help='source dataset family; selects the eval/calibration transform (CIFAR: Resize 224, mean/std 0.5). Data layout stays ImageNet-C-like: --data/{train,val}, --data_corruption/<corruption>/<level>.')
    parser.add_argument('--num_classes', default=1000, type=int, help='classifier size of the backbone (1000 = timm default head)')
    parser.add_argument('--arch_checkpoint', default=None, type=str, help='timm-format state_dict of a fine-tuned backbone (skips timm pretrained download); see tools/models/convert_hf_vit_to_timm.py')
    parser.add_argument('--source_split', default='val', choices=['val', 'train'], help='split used by obtain_train_loader for FOA/FOZO/ZOA source statistics (default val = ImageNet protocol)')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2000, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')
    parser.add_argument('--stream_order', default='default', choices=['default', 'class_sorted', 'imbalanced', 'sar'],
        help='Test stream ordering: default (DataLoader shuffle), class_sorted (extreme non-IID, GT-class-sorted), imbalanced (Dirichlet online label shift, 5k), or sar (SAR online imbalanced label shift, full set). Labels used only for ordering.')
    parser.add_argument('--noniid_dirichlet_delta', default=0.1, type=float, help='Dirichlet concentration for --stream_order imbalanced (smaller = more imbalanced).')
    parser.add_argument('--noniid_seed', default=0, type=int, help='Seed for the imbalanced/SAR stream ordering.')
    parser.add_argument('--sar_imbalance_ratio', default=500000, type=float, help='SAR online label-shift imbalance ratio (1=uniform, 500000=~pure class order). Used with --stream_order sar.')
    parser.add_argument('--sar_T', default=100000, type=int, help='SAR online label-shift stream length (SAR default 100000, with replacement).')
    parser.add_argument('--max_eval_batches', default=-1, type=int, help='limit eval batches for debugging; -1 runs full set')
    parser.add_argument('--sar_bs1_double_lr', default=False, action='store_true', help="reproduce SAR/main.py L205-207, which doubles the lr when exp_type=='bs1' and method=='sar'. Off by default so the bs=64 protocol is unchanged; set it for single-sample SAR.")
    parser.add_argument('--codedist_diag', default=False, action='store_true', help='diagnostic: log per-corruption post-LN B3 quantizer code-distribution L1 to source (episodic)')
    parser.add_argument('--codedist_src', default='', type=str, help='source code_hist stats (.pt with code_hist key) for --codedist_diag')
    parser.add_argument('--official_zoa_all_ln', default=False, action='store_true', help="ZOA: adapt every LayerNorm instead of the paper's set (blocks 1-8 x {norm1,norm2}); reproduces runs made before configure_model_for_official_zoa's fix_layers matching was corrected")
    parser.add_argument('--codedist_scope', default='b3', type=str, help="--codedist_diag interfaces: 'b3' (default, blocks 8-11 x {qkv,proj,fc1}), 'b3_all' (adds mlp.fc2), 'all17' (those 16 plus the head) or 'head' (the classifier's own activation quantizer, where the global CLS branch and NEO both act)")
    parser.add_argument('--sqnr_scope', default='b3', type=str, help="--sqnr_diag interfaces: 'b3' (default) or 'b3_all' (adds mlp.fc2)")
    parser.add_argument('--quar_local_oracle_target_file', default='', type=str,
                        help='diagnostic: .pt of {layer: (mean, std)} per-channel TARGET moments '
                             'measured on the same stream, used in place of the online estimate so '
                             'the shrinkage optimum can be read at delta^2 ~ 0. Never deployed.')
    parser.add_argument('--codehist_dump', default='', type=str,
                        help="diagnostic: dump the PER-CHANNEL code histogram at one interface "
                             "(e.g. 'blocks.10.attn.proj') as a [C, 2**n_bits] tensor, for the "
                             "figure that shows what the shift does to the code distribution. "
                             "Read-only; independent of --codedist_diag.")
    parser.add_argument('--sqnr_diag', default=False, action='store_true', help='diagnostic: log per-corruption per-channel SQNR (Var(x)/mean(e^2)) at the same B8-11 quantizer interfaces as --codedist_diag; no source stats needed')
    parser.add_argument('--measure_flops', default=False, action='store_true', help='profile FLOPs of the full per-batch adaptation (torch.profiler with_flops) on batches 2-4; logs MEASURED_GFLOPS_PER_IMG')
    parser.add_argument('--mem_profile', default=False, action='store_true', help='log peak CUDA memory (allocated/reserved MiB) per corruption; peak counter is reset once after model+adapter build')
    parser.add_argument('--latency_profile', default=False, action='store_true', help='time every batch forward (CUDA events) for batches>=2 and log LATPROF median/mean steady forward ms (per-image too); use with --max_eval_batches ~50 on an idle GPU')

    # algorithm selection
    parser.add_argument('--algorithm', default='no_adapt', type=str, help='supported: quar (the deployed paper configuration), no_adapt, foa, foa_spsa, t3a, zoa, zoa_diag, qk_zoa, qkzoa, qkzoa_diag, fozo, quest, qstta, qrecal, qfozo, q2stage, qfastcal, tent, eata, sar')
    parser.add_argument('--arch', default='vit_base_patch16_224', type=str, help='backbone architecture name; T3A uses this to dispatch ResNet vs ViT branches')
    parser.add_argument('--t3a_filter_k', default=20, type=int, help='T3A per-class top-K filter for support set selection')
    # backprop-based TTA baselines (TENT/EATA/SAR), official hyperparameters (SAR/main.py)
    # NOTE: ViT-base lr is computed inside the build functions as (0.001/64)*batch_size (SAR/main.py L193);
    #       --lr is NOT used for these methods (it belongs to FOZO/ZOA SPSA paths).
    parser.add_argument('--eata_fisher_size', default=2000, type=int, help='EATA: number of clean source samples for Fisher (official default 2000)')
    parser.add_argument('--eata_fisher_alpha', default=2000.0, type=float, help='EATA: Fisher regularizer trade-off beta (official default 2000.0)')
    parser.add_argument('--eata_e_margin', default=math.log(1000) * 0.40, type=float, help='EATA: entropy margin E_0 for reliable-sample filtering (official math.log(1000)*0.40)')
    parser.add_argument('--eata_d_margin', default=0.05, type=float, help='EATA: cosine-similarity epsilon for redundant-sample filtering (official 0.05)')
    parser.add_argument('--sar_margin_e0', default=math.log(1000) * 0.40, type=float, help='SAR: reliable-entropy margin E_0 (official math.log(1000)*0.40)')
    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='all', type=str, help='corruption type(s): "all" or comma-separated list')
    parser.add_argument('--corruption_repeat', default=1, type=int, help='Repeat the selected corruption stream for long-term continual evaluation.')
    parser.add_argument('--rounds_per_corruption', default=1, type=int, help='Number of evaluation rounds per corruption (state carried across rounds within a corruption; resets at corruption boundary if not --continual). Default 1 = current single-pass behavior.')
    parser.add_argument('--corruption_order_seed', default=None, type=int, help='Shuffle selected corruptions with this seed before optional repetition.')
    # model settings
    parser.add_argument('--quant', default=False, action='store_true', help='whether to use quantized model in the experiment')
    parser.add_argument('--quant_bit', default=8, type=int, help='global quantization bit-width for PTQ4ViT when --quant is enabled')
    parser.add_argument('--quant_w_bit', default=None, type=int, help='weight bit-width override for PTQ4ViT; default uses quant_bit')
    parser.add_argument('--quant_a_bit', default=None, type=int, help='activation bit-width override for PTQ4ViT; default uses quant_bit')
    parser.add_argument('--adalog_checkpoint', default=None, type=str, help='path to AdaLog pre-calibrated checkpoint; skips PTQ4ViT calibration when --quant is set')
    parser.add_argument('--ptq_checkpoint', default=None, type=str, help='path to PTQ4ViT pre-calibrated checkpoint (whole model via torch.save); skips Hessian calibration when provided')
    parser.add_argument('--save_ptq_checkpoint', default=None, type=str, help='ADDITIVE/default-off: if set, torch.save the freshly Hessian-calibrated PTQ4ViT net to this path (for producing matched-bit-width cross-quantizer checkpoints). No effect on any existing run when unset.')
    # output settings
    parser.add_argument('--output', default='./experiment_results', help='the output directory of this experiment')
    parser.add_argument('--tag', default='_first_experiment', type=str, help='the tag of experiment')
    # fozo settings
    parser.add_argument('--num_prompts', default=3, type=int, help='number of inserted prompts for test-time adaptation.')
    parser.add_argument('--fitness_lambda', default=0.4, type=float, help='the balance factor $lambda$')   
    parser.add_argument('--zo_eps', default=0.5, type=float, help='1')
    parser.add_argument('--lr', default=0.08, type=float, help='2')
    parser.add_argument('--n_spsa', default=1, type=int, help='3')
    parser.add_argument('--popsize', default=28, type=int, help='CMA-ES population size for --algorithm foa (Niu et al., ICML 2024 default)')
    parser.add_argument('--compute_train_info', default=False, action='store_true', help='force recomputation of train_info_std / train_info_mean for --algorithm foa')
    parser.add_argument('--foa_stats_dir', default='foa_stats', type=str, help='directory for FOA train_info cache (filename encodes bitwidth + num_prompts)')
    parser.add_argument('--foa_no_shift', default=False, action='store_true', help='ablation: disable FOA back-to-source activation shifting (Eqn. 7), leaving prompt+discrepancy only')
    # --- PEA (Ma et al., ICLR 2026) baseline; defaults are the authors' ImageNet-C settings ---
    parser.add_argument('--quar_local_alpha_profile', default='', type=str, help="per-block local lambda, e.g. '8:0.27,9:0.44,10:0.57,11:0.72'; overrides --quar_local_lambda for the listed blocks")
    parser.add_argument('--max_source_stat_batches', default=-1, type=int, help='limit source-stat batches for FOZO debugging; -1 uses full set')
    parser.add_argument('--fozo_stats_file', default='train_stats_2.pt', type=str, help='cache file path for FOZO source statistics')
    # official ZOA settings
    parser.add_argument('--official_zoa_steps', default=1, type=int, help='Official ZOA SPSA update steps per batch')
    parser.add_argument('--official_zoa_lr_alpha', default=0.01, type=float, help='Official ZOA learning rate for alpha aggregation coefficients')
    parser.add_argument('--official_zoa_lr_eps', default=0.0002, type=float, help='Official ZOA SGD learning rate for epsilon (domain-shift) parameters; official default 0.005')
    parser.add_argument('--official_zoa_spsa_c', default=0.02, type=float, help='Official ZOA SPSA perturbation scale for domain parameters')
    parser.add_argument('--official_zoa_spsa_c_alpha', default=0.05, type=float, help='Official ZOA SPSA perturbation scale for alpha coefficients')
    parser.add_argument('--official_zoa_spsa_momentum', default=0.0, type=float, help='Official ZOA SGD momentum for current domain parameters')
    parser.add_argument('--official_zoa_weight_decay', default=0.4, type=float, help='Official ZOA weight decay for current domain parameters')
    parser.add_argument('--official_zoa_weight_decay_alpha', default=0.1, type=float, help='Official ZOA weight decay for alpha coefficients')
    parser.add_argument('--official_zoa_sp_avg', default=1, type=int, help='Official ZOA number of SPSA perturbation averages')
    parser.add_argument('--official_zoa_lambda_bp', default=30.0, type=float, help='Official ZOA source-stat/entropy tradeoff for ViT')
    parser.add_argument('--official_zoa_stats_file', default='official_zoa_train_stats.pt', type=str, help='Official ZOA source-stat cache file')
    parser.add_argument('--official_zoa_max_source_stat_batches', default=-1, type=int, help='Limit official ZOA source-stat batches for smoke tests; -1 uses full source set')
    parser.add_argument('--official_zoa_max_weight_nums', default=32, type=int, help='Official ZOA maximum stored domain vectors')
    parser.add_argument('--official_zoa_alpha_scale', default=10.0, type=float, help='Official ZOA alpha softmax scale')
    parser.add_argument('--official_zoa_domain_t', default=0.1, type=float, help='Official ZOA domain-change threshold')
    parser.add_argument('--zoa_layer_balance', default='none', choices=['none', 'source_norm', 'uniform_batch'], help="ZOA layer-balanced discrepancy mode. Default 'none' matches the main experiments (project convention; launchers set 'none' explicitly 65x vs source_norm 7x). source_norm weights layers by inverse mean source std; uniform_batch by inverse batch D_l (ablation only).")
    parser.add_argument('--zoa_layer_balance_eps', default=1e-4, type=float, help='Epsilon added to scale denominator to avoid division by zero in layer balance weights.')
    parser.add_argument('--zoa_layer_balance_clip_min', default=0.1, type=float, help='Minimum clamp value for layer balance weights before normalization.')
    parser.add_argument('--zoa_layer_balance_clip_max', default=10.0, type=float, help='Maximum clamp value for layer balance weights before normalization.')
    # Q-risk update damping settings
    # QBA settings
    # QBA-v2: layer balance
    # QBA-v2: ratio-calibrated lambda
    # AdaLog scale TTA settings
    # QZO settings
    # QuAR local-branch settings
    parser.add_argument('--quar_local_enable', default=False, action='store_true',
                        help='Enable the QuAR local branch.')
    parser.add_argument('--quar_local_mode', default='explicit',
                        choices=['explicit', 'folded_quantizer'],
                        help='Local-branch application mode. '
                             'explicit: modifies x before the quantizer (default). '
                             'folded_quantizer: folds correction into quantizer state; '
                             'fold type controlled by --quar_local_fold_type.')
    parser.add_argument('--quar_local_fold_type', default='scale_only',
                        choices=['scale_only',
                                 'scale_offset', 'scale_offset_coupled', 'scale_offset_decoupled',
                                 'scale_shift', 'scale_shift_coupled', 'scale_shift_decoupled',
                                 'output_only'],
                        help='Fold type for folded_quantizer mode. '
                             'scale_only: fold a into a_interval only (PTQ4ViT). '
                             'scale_offset / scale_offset_coupled: fold (a,b) into a_interval+qzero_z; '
                             'codes match explicit but dequant cancels correction (coupled). '
                             'scale_offset_decoupled: fold via qaffine_gamma/beta (per-channel); '
                             'codes AND dequant output match the explicit local branch (output-equivalent). '
                             'scale_shift / scale_shift_coupled: fold (a,b) into AdaLog scale+shift (coupled). '
                             'scale_shift_decoupled: not yet implemented for AdaLog (nonlinear quant). '
                             'output_only: quantize with original params, then apply γ·dq + β post-correction; '
                             'tests whether output distribution correction alone explains local-branch gain. '
                             'Falls back to explicit for unsupported module types.')
    parser.add_argument('--quar_local_scope', default='block8_9_10_11_all',
                        choices=['last_block_only', 'last_2_blocks', 'qba_valid', 'attn_only', 'mlp_only', 'all',
                                 'source_reliable', 'random_same_k',
                                 'block11_valid', 'block11_all',
                                 'block10_valid', 'block10_all',
                                 'block10_11_all',
                                 'block9_10_11_valid', 'block9_10_11_all',
                                 'block8_9_10_11_valid', 'block8_9_10_11_all',
                                 'block8_9_10_11_attn', 'block8_9_10_11_mlp1', 'all_blocks_valid_no_fc2',
                                 'block7_to_11_valid', 'block6_to_11_valid',
                                 'block5_to_11_valid', 'block4_to_11_valid',
                                 'block7_to_11_all', 'block6_to_11_all', 'block5_to_11_all',
                                 'block0_to_3_all', 'block4_to_7_all',
                                 'resid_block8_9_10_11'],   # placement control: block OUTPUTS (residual stream)
                        help='Layer scope for the local branch correction.')
    parser.add_argument('--quar_local_random_scope_seed', default=0, type=int,
                        help='Random seed for random_same_k scope (selects k=len(qba_valid) layers randomly).')
    parser.add_argument('--quar_local_lambda', default=0.5, type=float,
                        help='Blend weight: x_out = (1-alpha)*x + alpha*x_corr. 1.0 = full correction.')
    parser.add_argument('--quar_local_stats', default='mean_std',
                        choices=['mean_std', 'std_only', 'target_only', 'mean_only', 'std_centered'],
                        help='Correction mode: mean_std (full the local branch); std_only (scale only, uncentered); '
                             'target_only (standardize to target mean/std, no source ref); '
                             'mean_only (shift mean to source, keep variance); '
                             'std_centered (scale std to source around target mean, no mean shift).')
    parser.add_argument('--quar_local_eps', default=1e-6, type=float,
                        help='Epsilon for numerical stability in sigma estimation.')
    parser.add_argument('--quar_local_chunk', default=2, type=int,
                        help='apply the local correction over this many batch slices, in place, '
                             'instead of materialising the corrected tensor. 0 or 1 disables it. '
                             'Bit-identical to the unsliced form -- every operation is elementwise '
                             'or a per-channel broadcast and the target statistics are reduced over '
                             'the whole input first -- and it removes the peak-memory cost of the '
                             'fc2 interface, whose input is the 3072-wide MLP hidden state '
                             '(+148 MB at batch 64, +37 MB at batch 16, both down to +0.17 MB). '
                             'Ignored when the input needs gradients, or when the batch has fewer '
                             'rows than slices.')
    parser.add_argument('--quar_local_clip_ratio', default=3.0, type=float,
                        help='Clip sigma ratio to [1/clip_ratio, clip_ratio] to prevent extremes.')
    parser.add_argument('--quar_local_use_ema', default=False, action='store_true',
                        help='Use EMA smoothing of target stats across batches within a corruption.')
    parser.add_argument('--quar_local_ema_momentum', default=0.1, type=float,
                        help='EMA momentum for target stats (0=no update, 1=no smoothing).')
    parser.add_argument('--quar_local_stream_ema', default=False, action='store_true',
                        help='Source-initialized streaming EMA for the local branch with second-moment variance '
                             'tracking and warm-up ramp. Requires explicit mode.')
    parser.add_argument('--quar_local_stream_alpha', default=0.05, type=float,
                        help='EMA learning rate for streaming EMA-the local branch.')
    parser.add_argument('--quar_local_stream_nwarm', default=64, type=int,
                        help='Warm-up length in images for streaming EMA-the local branch (rho ramp 0→1).')
    parser.add_argument('--quar_local_stream_lag', default=False, action='store_true',
                        help='Use lagged EMA state for correction (update after); '
                             'default is inclusive (update before correction).')
    parser.add_argument('--quar_local_stream_incl', default=False, action='store_true',
                        help='local branch: include the current batch in the running moments (overrides --quar_local_stream_lag and the algorithm=quar preset)')
    parser.add_argument('--quar_local_stats_file', default='', type=str,
                        help='Path to pre-built local-branch source stats .pt file.')
    parser.add_argument('--quar_local_fp_mode', default=False, action='store_true',
                        help='FP control: select modules by name (no quantizer detection).')
    # Quantizer-output mismatch diagnostic (read-only; collects pre-quant + post-DQ
    # per-channel activation stats at the local branch-selected local locations).
    parser.add_argument('--quar_mismatch_diag', default=False, action='store_true',
                        help='Collect pre-quant + post-DQ activation stats at local branch-selected '
                             'locations and dump qac_mismatch_stats.pt. Read-only.')
    parser.add_argument('--quar_mismatch_diag_include_fc2', default=False, action='store_true', help='--quar_mismatch_diag: also read mlp.fc2. The mismatch metric is a per-channel moment distance and does not depend on the quantizer grid, so fc2 can be measured like the others; it is off by default to preserve earlier runs.')
    parser.add_argument('--quar_mismatch_diag_scope', default='block8_9_10_11_all', type=str,
                        help='Module scope for the mismatch diagnostic.')
    # Code-guided QuAR local-branch settings
    parser.add_argument('--quar_local_quant_guidance', default='none',
                        choices=['none', 'code_distance_layer', 'code_distance_global',
                                 'apply_if_improves'],
                        help='Quantizer code-distribution guidance for the local branch. '
                             'Requires code_hist in --quar_local_stats_file '
                             '(built with build_qac_stats.py --save_code_hist). '
                             'none: uniform alpha; code_distance_layer: per-layer alpha '
                             'proportional to code-dist shift; code_distance_global: single '
                             'alpha from mean code dist (one-batch lag); '
                             'apply_if_improves: apply only when the local branch reduces code dist.')
    parser.add_argument('--quar_local_code_metric', default='l1',
                        choices=['l1', 'kl', 'hellinger'],
                        help='Distance metric for code histogram comparison.')
    parser.add_argument('--quar_local_code_bins', default=0, type=int,
                        help='Code histogram bins (0 = auto: 2**n_bits).')
    parser.add_argument('--quar_local_alpha_max', default=1.0, type=float,
                        help='Maximum alpha for guided modes (replaces --quar_local_lambda as ceiling).')
    parser.add_argument('--quar_local_alpha_min', default=0.0, type=float,
                        help='Minimum alpha for guided modes (floor).')
    parser.add_argument('--quar_local_guidance_temperature', default=1.0, type=float,
                        help='Temperature for code_distance_layer softmax alpha normalization.')
    parser.add_argument('--quar_local_exclude_layers', nargs='*', default=None,
                        help='Layer names to exclude from the local branch scope (e.g. blocks.11.attn.proj).')
    # AQR baseline (Adaptive Quantile Recalibration, Mehrbod et al., WACV 2026).
    # Additive baseline; independent of the local branch/QuAR. Enable with
    #   --algorithm no_adapt --aqr_enable ...  (see tta_library/aqr.py).
    parser.add_argument('--aqr_enable', default=False, action='store_true',
                        help='Enable AQR channel-wise quantile recalibration (baseline).')
    parser.add_argument('--aqr_stats_file', default='', type=str,
                        help='Path to AQR source percentile .pt (from build_aqr_stats.py).')
    parser.add_argument('--aqr_scope', default='block8_9_10_11_all', type=str,
                        help='AQR layer scope. Default matches the canonical QuAR scope.')
    parser.add_argument('--aqr_hook', default='linear_prehook',
                        choices=['linear_prehook', 'ln_output'],
                        help='linear_prehook: same pre-quantizer sites as QuAR (default); '
                             'ln_output: paper-literal LayerNorm-output placement.')
    parser.add_argument('--aqr_alpha', default=1.0, type=float,
                        help='Blend x_out=(1-a)x+a*AQR(x). 1.0 = pure AQR (default).')
    parser.add_argument('--aqr_eps', default=1e-6, type=float,
                        help='Degenerate-interval / division guard.')
    parser.add_argument('--aqr_clip_tails', default=True, action='store_true',
                        help='Clamp values to [p^T_0, p^T_100] before mapping (numerical guard).')
    parser.add_argument('--aqr_target_tail', default='raw', choices=['raw', 'avg_sample'],
                        help='Target p0/p100 estimation: raw batch min/max (pair with '
                             '--tail_calib standard source) or avg_sample (Average Sample '
                             'Tails; MUST pair with an avg_sample source stats file).')
    parser.add_argument('--aqr_tail_m', default=100, type=int,
                        help='Average Sample Tails: sample size per draw (paper: 100).')
    parser.add_argument('--aqr_tail_reps', default=1000, type=int,
                        help='Average Sample Tails: number of draws to average (paper: 1000).')
    # CLS shift settings (FOA-style final CLS activation shifting)
    parser.add_argument('--quar_global_enable', default=False, action='store_true',
                        help='Enable FOA-style CLS-token activation shifting.')
    parser.add_argument('--quar_global_stats_file', default='', type=str,
                        help='Path to CLS source statistics file (from build_cls_stats.py).')
    parser.add_argument('--quar_global_position', default='post_ln',
                        choices=['pre_ln', 'post_ln'],
                        help='CLS interception point: pre_ln (before net.norm) or post_ln (classifier input).')
    parser.add_argument('--quar_global_mode', default='mean_std',
                        choices=['mean_only', 'mean_std'],
                        help='Shift mode: mean_only shifts mean; mean_std shifts mean and std.')
    parser.add_argument('--quar_global_alpha', default=1.0, type=float,
                        help='Blend weight for CLS shift (1.0=full correction, 0.5=half).')
    parser.add_argument('--quar_global_eps', default=1e-6, type=float,
                        help='Epsilon for std denominator in mean_std mode.')
    parser.add_argument('--quar_global_clip_ratio', default=3.0, type=float,
                        help='Max sig_ratio magnitude for mean_std mode.')
    parser.add_argument('--quar_global_target_ema_alpha', default=0.0, type=float,
                        help='EMA alpha for target center estimation (0=per-batch, 0.1=FOA-act style).')
    parser.add_argument('--quar_global_stream_lag', default=False, action='store_true',
                        help='global branch: correct with the pre-update running moments (exclude the current batch), matching the local branch')
    parser.add_argument('--recal_cumulative', default=False, action='store_true',
                        help='Use a source-initialized CUMULATIVE running mean (alpha_n=1/(batch_count+1), '
                             'forgetting-free, NEO-style) for BOTH the local local-branch stream and the global CLS '
                             'target moments, instead of the fixed-rate EMA. Default False = unchanged EMA behavior. '
                             'Resets per domain via reset() (episodic); intended for non-IID-robustness study.')
    # Target-moment ESTIMATOR ablation (additive; default reproduces current behavior byte-for-byte).
    parser.add_argument('--quar_local_estimator', default='current',
                        choices=['current', 'welford'],
                        help='Target-moment estimator for the LOCAL (the local branch) branch. '
                             'current: existing behavior (unchanged, driven by --quar_local_stream_ema/--recal_cumulative). '
                             'welford: exact streaming POOLED variance over B*N tokens '
                             '(per-channel running count/mean/M2, source-initialized as one pseudo-batch).')
    parser.add_argument('--quar_global_estimator', default='current',
                        choices=['current', 'ema', 'welford'],
                        help='Target-moment estimator for the GLOBAL (CLS) branch. '
                             'current: existing behavior (unchanged, driven by --recal_cumulative/--quar_global_target_ema_alpha). '
                             'ema: fixed-rate EMA with alpha=0.1 for mu_t and sig_t. '
                             'welford: exact streaming POOLED variance over B CLS tokens '
                             '(per-channel running count/mean/M2, source-initialized as one pseudo-batch).')
    # NEO: no-optimization TTA through latent re-centering
    parser.add_argument('--neo_enable', default=False, action='store_true',
                        help='Enable official NEO cumulative mean subtraction at head input.')
    parser.add_argument('--neo_ema_alpha', default=None, type=float,
                        help='NEO-Continual EMA momentum (official neo_layers.NEO_Continual default 0.1; '
                             'None=base NEO cumulative 1/i). Use the EMA variant for the continual setting.')
    # qstta settings
    parser.add_argument('--continual', default=False, action='store_true', help='If true, keep adaptation state across corruptions.')
    parser.add_argument('--imagenet_5k_eval', default=False, action='store_true', help='Use the RobustBench ImageNet 5k subset without changing reset behavior.')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    robustbench_data_dir = os.path.join(current_dir, 'robustbench', 'data')
    parser.add_argument('--imagenet_5k_indices_file',
                        default=os.path.join(robustbench_data_dir, 'imagenet_test_image_ids.txt'),
                        type=str,
                        help='Path to the file containing 5k ImageNet validation image paths from RobustBench.')
    parser.add_argument('--imagenet_class_map_file',
                        default=os.path.join(robustbench_data_dir, 'imagenet_class_to_id_map.json'),
                        type=str,
                        help='Path to the ImageNet class-to-ID mapping file from RobustBench.')

    return parser.parse_args() 


def apply_quar_preset(args, argv):
    """Expand --algorithm quar into the deployed configuration from the paper.

    QuAR is no_adapt plus the local and global recalibration switches, which is how every
    launcher spells it. Those switches stay off by default on purpose: the wrappers are built
    inside the no_adapt and zoa branches, so defaulting them on would silently turn the
    No Adapt reference row and the ZOA baseline into QuAR. Scoping them to a single algorithm
    value leaves every existing command line untouched.

    Flags passed explicitly win, so the lambda and batch-size sweeps still run through this.
    Statistics paths follow the naming that build_qac_stats.py / build_cls_stats.py emit.

    """
    if args.algorithm != 'quar':
        return args
    given = set()
    for tok in argv:
        if tok.startswith('--'):
            given.add(tok.split('=', 1)[0])
    deployed = (('quar_local_enable', True), ('quar_local_stream_ema', True), ('quar_local_stream_nwarm', 0),
                ('quar_local_stream_lag', True), ('quar_global_enable', True),
                ('quar_global_target_ema_alpha', 0.05), ('recal_cumulative', True))
    for key, value in deployed:
        if '--' + key not in given:
            setattr(args, key, value)
    w_bit = args.quant_w_bit if args.quant_w_bit is not None else args.quant_bit
    a_bit = args.quant_a_bit if args.quant_a_bit is not None else args.quant_bit
    bw = 'w{}a{}'.format(w_bit, a_bit) if args.quant else 'fp'
    here = os.path.dirname(os.path.abspath(__file__))
    if not args.quar_local_stats_file:
        args.quar_local_stats_file = os.path.join(here, 'stats', 'qac_stats_{}_all_layers_seed0.pt'.format(bw))
    if not args.quar_global_stats_file:
        args.quar_global_stats_file = os.path.join(here, 'stats', 'cls_stats_{}_seed0.pt'.format(bw))
    if getattr(args, 'quar_local_stream_incl', False):
        args.quar_local_stream_lag = False
    args.algorithm = 'no_adapt'
    return args


if __name__ == '__main__':
    args = get_args()
    args = apply_quar_preset(args, sys.argv[1:])
    if getattr(args, 'quar_local_stream_incl', False):
        args.quar_local_stream_lag = False  # explicit override wins on every path
    
    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # create logger for experiment
    args.output += '/' + args.algorithm + args.tag + '/'
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    log_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt"
    logger = get_logger(name="project", output_directory=args.output, log_name=log_name, debug=False)
    original_log_path = os.path.join(args.output, log_name)
    logger.info(args)

    # configure the domains for adaptation
    # options for ImageNet-R/V2/Sketch are ['rendition', 'v2', 'sketch']
    all_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    if args.corruption in ['all', 'imagenet-c-all']:
        corruptions = all_corruptions
    else:
        corruptions = [c.strip() for c in args.corruption.split(',') if c.strip()]
    if args.corruption_order_seed is not None:
        order_rng = random.Random(int(args.corruption_order_seed))
        corruptions = list(corruptions)
        order_rng.shuffle(corruptions)
        logger.info(f"Using shuffled corruption order seed={args.corruption_order_seed}: {corruptions}")
    repeat_count = max(1, int(args.corruption_repeat))
    stream_entries = [
        (round_idx + 1, order_idx + 1, corruption)
        for round_idx in range(repeat_count)
        for order_idx, corruption in enumerate(corruptions)
    ]
    if repeat_count > 1:
        logger.info(f"Repeating corruption stream {repeat_count} times; total domains={len(stream_entries)}.")

    if args.continual:
        logger.info("Running in continual mode: adaptation state is kept across specified corruptions.")
    if args.imagenet_5k_eval:
        logger.info("Running with the RobustBench ImageNet 5k subset.")
    if getattr(args, 'dataset', 'imagenet') != 'imagenet' and args.imagenet_5k_eval:
        raise ValueError("--imagenet_5k_eval is ImageNet-only; drop it for --dataset %s" % args.dataset)
    # Backbone class count: None keeps timm's default head (ImageNet paths unchanged).
    _bb_nc = int(args.num_classes) if int(getattr(args, 'num_classes', 1000)) != 1000 else None
    if _bb_nc is not None or args.arch_checkpoint:
        logger.info(f"Backbone: {args.arch} num_classes={_bb_nc or 'default'} checkpoint={args.arch_checkpoint} dataset={args.dataset}")
    if args.quant:
        if args.adalog_checkpoint is not None:
            # AdaLog path: load pre-calibrated/optimized checkpoint, skip Hessian calibration.
            # AdaLog wraps quantizers as nn.Module submodules (a_quantizer, w_quantizer, A_quantizer).
            _adalog_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'quant_library', 'adalog')
            _adalog_utils_init = os.path.join(_adalog_root, 'utils', '__init__.py')
            if not os.path.exists(_adalog_utils_init):
                open(_adalog_utils_init, 'w').close()
            if _adalog_root not in sys.path:
                sys.path.insert(0, _adalog_root)
            import importlib as _importlib
            # Temporarily displace cached modules that shadow AdaLog packages:
            # FOZO has its own 'utils' and 'quant_layers' which shadow AdaLog's.
            _adalog_shadow_keys = [k for k in list(sys.modules.keys())
                                   if k in ('utils', 'quant_layers', 'quantizers', 'configs')
                                   or k.startswith(('utils.', 'quant_layers.', 'quantizers.', 'configs.'))]
            _adalog_saved_mods = {k: sys.modules.pop(k) for k in _adalog_shadow_keys}
            try:
                _adalog_wrap = _importlib.import_module('utils.wrap_net')
                _adalog_wbit = int(args.quant_w_bit) if args.quant_w_bit is not None else int(args.quant_bit)
                _adalog_abit = int(args.quant_a_bit) if args.quant_a_bit is not None else int(args.quant_bit)
                # (4, 8) is the asymmetric deployment setting; its config differs from
                # configs.4bit only in the activation-side bit-widths, including
                # qhead_a_bit, which the overrides below do not touch. Without this entry
                # the lookup fell back to configs.4bit and left qhead_a_bit at 4 while the
                # checkpoint had been calibrated with 8, which loads cleanly but produces
                # chance-level accuracy.
                _adalog_cfg_name = {(4, 4): 'configs.4bit', (6, 6): 'configs.6bit', (3, 3): 'configs.3bit', (8, 8): 'configs.8bit', (4, 8): 'configs.w4a8'}.get(
                    (_adalog_wbit, _adalog_abit), 'configs.4bit')
                _adalog_cfg_mod = _importlib.import_module(_adalog_cfg_name)
            finally:
                # Restore FOZO modules; keep AdaLog's under namespaced keys
                for _k in list(sys.modules.keys()):
                    if _k in ('utils', 'quant_layers', 'quantizers', 'configs') or \
                       _k.startswith(('utils.', 'quant_layers.', 'quantizers.', 'configs.')):
                        sys.modules[f'_adalog_{_k}'] = sys.modules.pop(_k)
                sys.modules.update(_adalog_saved_mods)
            adalog_cfg = _adalog_cfg_mod.Config()
            adalog_cfg.w_bit = _adalog_wbit
            adalog_cfg.a_bit = _adalog_abit
            adalog_cfg.s_bit = _adalog_abit
            net = create_backbone(args.arch, num_classes=_bb_nc, checkpoint=args.arch_checkpoint)
            net = _adalog_wrap.wrap_modules_in_net(net, adalog_cfg, reparam=False)
            for _name, _mod in net.named_modules():
                if hasattr(_mod, 'mode'):
                    _mod.calibrated = True
                    _mod.mode = 'quant_forward'
                for _qattr in ('a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer'):
                    if hasattr(_mod, _qattr):
                        getattr(_mod, _qattr).inited = True
            _ckpt = torch.load(args.adalog_checkpoint, map_location='cpu')
            _load_result = net.load_state_dict(_ckpt, strict=False)
            logger.info(f"AdaLog checkpoint loaded from {args.adalog_checkpoint}")
            logger.info(f"  missing keys: {len(_load_result.missing_keys)}, unexpected keys: {len(_load_result.unexpected_keys)}")
            if _load_result.missing_keys:
                logger.info(f"  first 5 missing: {_load_result.missing_keys[:5]}")
            net = net.cuda()
            g = datasets.ViTImageNetLoaderGenerator(args.data, 'imagenet', 32, 32, 16, kwargs={"model": net, "transform": calib_transform_override(args)})
            test_loader = g.test_loader()
        else:
            # PTQ4ViT path: Hessian calibration (W8A8 / W6A6)
            # NOTE the bit of quantization can be modified in quant_library/configs/PTQ4ViT.py
            quant_cfg = init_config("PTQ4ViT")
            quant_cfg.bit = int(args.quant_bit)
            weight_bit = int(args.quant_w_bit) if args.quant_w_bit is not None else int(args.quant_bit)
            activation_bit = int(args.quant_a_bit) if args.quant_a_bit is not None else int(args.quant_bit)
            bit_overrides = {
                "w_bit": weight_bit,
                "a_bit": activation_bit,
                "A_bit": activation_bit,
                "B_bit": activation_bit,
            }
            for dname, bit_value in bit_overrides.items():
                if hasattr(quant_cfg, dname):
                    d = getattr(quant_cfg, dname)
                    for k in d:
                        d[k] = bit_value
            net = get_net(args.arch, num_classes=_bb_nc, checkpoint=args.arch_checkpoint)
            wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)
            g = datasets.ViTImageNetLoaderGenerator(args.data, 'imagenet', 32, 32, 16, kwargs={"model": net, "transform": calib_transform_override(args)})
            test_loader = g.test_loader()
            if args.ptq_checkpoint and os.path.isfile(args.ptq_checkpoint):
                logger.info(f"Loading PTQ4ViT checkpoint from {args.ptq_checkpoint} (skipping Hessian calibration)")
                from quant_library.quant_utils.models import attention_forward as _attn_fwd, window_attention_forward as _win_attn_fwd
                from timm.models.vision_transformer import Attention as _Attn
                from timm.models.swin_transformer import WindowAttention as _WinAttn
                _Attn.attention_forward = _attn_fwd
                _WinAttn.window_attention_forward = _win_attn_fwd
                net = torch.load(args.ptq_checkpoint, map_location='cpu', weights_only=False)
                for _m in net.modules():
                    if hasattr(_m, 'mode'):
                        _m.mode = 'quant_forward'
            else:
                calib_loader = g.calib_loader(num=32)
                quant_calibrator = HessianQuantCalibrator(net, wrapped_modules, calib_loader, sequential=False, batch_size=4)
                quant_calibrator.batching_quant_calib()
                if getattr(args, 'save_ptq_checkpoint', None):
                    torch.save(net, args.save_ptq_checkpoint)
                    logger.info(f"[save_ptq_checkpoint] saved freshly-calibrated PTQ4ViT net to {args.save_ptq_checkpoint}")
    else:
        # full precision model
        net = create_backbone(args.arch, num_classes=_bb_nc, checkpoint=args.arch_checkpoint)
        
    net = net.cuda()
    # Move plain-attr tensors that .cuda() doesn't reach (w_interval, a_interval, etc.)
    if args.quant and hasattr(args, 'ptq_checkpoint') and args.ptq_checkpoint and os.path.isfile(args.ptq_checkpoint):
        _plain_attrs = ('w_interval', 'a_interval', 'A_interval', 'B_interval', 'qzero_z')
        for _m in net.modules():
            for _attr in _plain_attrs:
                _t = getattr(_m, _attr, None)
                if isinstance(_t, torch.Tensor) and not _t.is_cuda:
                    setattr(_m, _attr, _t.cuda())
    net.eval()
    net.requires_grad_(False)

    

    if args.algorithm in ['fozo', 'quest']:
        # Optional Ours composition: install the local branch + CLS hooks on raw quantized ViT
        # BEFORE PromptViT wrap. QuARLocalMixin/CLSShiftWrapper register hooks on the
        # inner Linear/norm/head modules, so they fire during FOZO's forward
        # passes (used for SPSA gradient + final prediction).
        _fozo_qac_holder = None
        _fozo_cls_holder = None
        if getattr(args, 'quar_local_enable', False):
            _fozo_qac_holder = build_no_adapt_qac_wrapper(args, net, logger)
        if getattr(args, 'quar_global_enable', False):
            _fozo_cls_holder = build_cls_shift_wrapper(args, net, logger, local_wrapper=_fozo_qac_holder)
        net = PromptViT(net, args.num_prompts).cuda()
        adapt_model = FOZO(
            net,
            zo_eps=args.zo_eps,
            lr=args.lr,
            fitness_lambda=args.fitness_lambda,
            n_spsa=args.n_spsa,
            log_dir=args.output,
        )
        # Attach the local branch/CLS holders so per-corruption reset cascades to their EMA state.
        if _fozo_qac_holder is not None:
            adapt_model._quar_holder = _fozo_qac_holder
        if _fozo_cls_holder is not None:
            adapt_model._cls_shift_holder = _fozo_cls_holder
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(
            train_loader,
            stats_file=args.fozo_stats_file,
            max_batches=args.max_source_stat_batches,
        )
    elif args.algorithm == 'foa':
        # Official CMA-ES FOA (Niu et al., ICML 2024).
        # Uses ZOA_official PromptViT (self.model attribute) for compatibility.
        from models.vpt_official import PromptViT as PromptViTOfficial
        net = PromptViTOfficial(net, args.num_prompts).cuda()
        adapt_model = FOA(args, net, fitness_lambda=args.fitness_lambda).cuda()
        _, train_loader = obtain_train_loader(args)
        adapt_model.obtain_origin_stat(train_loader)
    elif args.algorithm == 't3a':
        # T3A baseline (Iwasawa & Matsuo, NeurIPS 2021).
        # Verbatim port from ZOA_official; works on quantized ViT-B because
        # MinMaxQuantLinear(nn.Linear) preserves isinstance(head, nn.Linear).
        from tta_library.t3a import T3A, configure_model as t3a_configure_model
        net = t3a_configure_model(net)
        adapt_model = T3A(args, net, num_classes=int(getattr(args, 'num_classes', 1000)), filter_K=args.t3a_filter_k).cuda()
    elif args.algorithm in ['tent', 'eata', 'sar']:
        # Backprop-based TTA baselines (TENT/EATA/SAR) on the quantized
        # ViT. The straight-through activation fake-quant lets gradients reach
        # the FP LayerNorm affine params (TENT/EATA/SAR) and all params
        # The official SAR-repo wrappers run forward under
        # @torch.enable_grad(), so adaptation works inside validate_adapt's
        # no_grad loop. Each wrapper exposes .reset() for the episodic
        # per-corruption protocol. See tta_library/backprop_tta.py and
        # runs/backprop_tta_MODIFICATIONS.md.
        from tta_library.backprop_tta import (
            build_tent, build_eata, build_sar,
        )
        if args.algorithm == 'tent':
            adapt_model, _bp_names, _bp_lr = build_tent(args, net)
        elif args.algorithm == 'eata':
            adapt_model, _bp_names, _bp_lr = build_eata(
                args, net, obtain_train_loader, logger=logger)
        elif args.algorithm == 'sar':
            adapt_model, _bp_names, _bp_lr = build_sar(args, net)
        adapt_model = adapt_model.cuda()
        logger.info(f"[{args.algorithm}] built backprop-TTA adapter (lr={_bp_lr:.6g})")
    elif args.algorithm in ['spa', 'spa+tent', 'spa+eta', 'spa+eata']:
        # SPA (Niu et al., ICML 2025), official code vendored in spa_official.py.
        # Backprop-based like TENT/EATA/SAR (same added STE), but 3 forwards +
        # 1 backward per batch and two optimizers. See tta_library/spa_tta.py.
        from tta_library.spa.tta import build_spa
        adapt_model, _spa_names = build_spa(
            args, net, obtain_train_loader=obtain_train_loader, logger=logger)
        adapt_model = adapt_model.cuda()
    elif args.algorithm == 'zoa':
        adapt_model = build_official_zoa_adapter(args, net, args.output, enable_qkzoa=False)
        if getattr(args, "quar_global_enable", False):
            adapt_model._cls_shift_holder = build_cls_shift_wrapper(args, net, logger)
            adapt_model._cls_shift_holder._freeze_owner = adapt_model
    elif args.algorithm == 'no_adapt':
        # Stack: NEO (outermost) → CLS-shift → the local branch → raw net.
        # Each wrapper installs hooks on the raw net's modules, so all hooks
        # fire on every forward through the outermost wrapper. The outermost
        # wrapper's .reset() cascades to inner wrappers.
        local_wrapper = None
        if getattr(args, "quar_local_enable", False):
            local_wrapper = build_no_adapt_qac_wrapper(args, net, logger)
            adapt_model = local_wrapper
        else:
            adapt_model = net
        if getattr(args, "aqr_enable", False):
            # AQR baseline (WACV 2026): standalone quantile recalibration.
            # Additive path -- when --aqr_enable is absent (default) this branch
            # is skipped and the QuAR/no_adapt behavior is unchanged.
            from tta_library.aqr import build_no_adapt_aqr_wrapper
            adapt_model = build_no_adapt_aqr_wrapper(args, net, logger)
        if getattr(args, "quar_global_enable", False):
            adapt_model = build_cls_shift_wrapper(args, net, logger,
                                                  local_wrapper=local_wrapper)
        if getattr(args, "neo_enable", False):
            # If CLS-shift is also active, NEO wraps it; otherwise NEO wraps
            # the local_wrapper (or raw net via the build helper's fallback).
            _inner = adapt_model if (local_wrapper is not None or getattr(args, "quar_global_enable", False)) else None
            adapt_model = build_neo_wrapper(args, net, logger, local_wrapper=_inner)
    else:
        assert False, NotImplementedError

    # Quantizer-output mismatch diagnostic: register AFTER the wrappers so the
    # stat hooks observe the post-recalibration activation (Local/Ours) or the
    # raw activation (No-Adapt/Final/NEO). Read-only; never modifies the forward.
    if getattr(args, 'quar_local_oracle_target_file', ''):
        _ot = torch.load(args.quar_local_oracle_target_file, map_location='cpu')
        for _tgt in (adapt_model, net):
            if hasattr(_tgt, '_quar_alpha'):
                _tgt._quar_oracle_tgt = _ot
                logger.info(f"[local branch-oracle] fixed target moments for {len(_ot)} layers from "
                            f"{args.quar_local_oracle_target_file}")
                break
        else:
            logger.warning("[local branch-oracle] no local-branch wrapper found; oracle moments NOT applied")

    _mismatch_collector = None
    if getattr(args, "quar_mismatch_diag", False):
        from tools.analysis.quar_mismatch_diag import MismatchCollector
        _mm_kw = dict(scope=getattr(args, "quar_mismatch_diag_scope", "block8_9_10_11_all"),
                      exclude_fc2=not getattr(args, "quar_mismatch_diag_include_fc2", False),
                      logger=logger)
        _mismatch_collector = MismatchCollector(net, **_mm_kw)
        if not _mismatch_collector.hooks and adapt_model is not net:
            # T3A moves the quantized head out of net's module tree: get_vit_featurer
            # sets net.head = nn.Identity() and the head object survives only as
            # T3A.classifier. Retry on the wrapper, mirroring the codedist hook.
            _mismatch_collector.remove()
            _mismatch_collector = MismatchCollector(adapt_model, **_mm_kw)

    corrupt_acc, corrupt_ece = [], []
    corruption_metrics_path = os.path.join(args.output, "corruption_metrics.csv")
    with open(corruption_metrics_path, "w") as f:
        f.write("round,order_index,corruption,top1,top5,ece_percent,steady_forward_ms\n")
    # Per-round CSV is only emitted when rounds_per_corruption > 1 (Table 3 style).
    rounds_per_corr = max(1, int(getattr(args, "rounds_per_corruption", 1)))
    rounds_csv_path = os.path.join(args.output, "rounds_metrics.csv")
    if rounds_per_corr > 1:
        with open(rounds_csv_path, "w") as f:
            f.write("stream_round,order_index,corruption_round,corruption,top1,top5,steady_forward_ms\n")
    _cd = {}
    if getattr(args, 'codedist_diag', False):
        import re as _re
        from build_qac_stats import (_is_quant_act as _isq, _get_quant_scale_qmax as _gsq,
                                     _compute_code_hist as _cch, _get_neg_scale as _gns)
        _cd['src'] = torch.load(args.codedist_src, map_location='cpu').get('code_hist', {})
        _cd['accum'] = {}; _cd['KEY'] = _re.compile(r"(blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.fc1))")
        # b3_all additionally covers mlp.fc2, the post-GELU interface the local
        # branch also recalibrates; its twin-uniform codes are handled below.
        _cd['KEY16'] = _re.compile(r"(blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2)))")
        _cd_scope = getattr(args, 'codedist_scope', 'b3')
        def _cd_canon(nm):
            if _cd_scope == 'all17':
                # 16 intermediate interfaces the local branch acts on, plus the
                # classifier head input the global branch acts on: one run yields
                # every aggregate, and the per-interface log lets any subset be
                # recovered afterwards.
                _leaf = nm.split('.')[-1]
                if _leaf in ('head', 'classifier'):
                    return 'head' if 'head' in _cd['src'] else None
                m16 = _cd['KEY16'].search(nm)
                if not m16:
                    return None
                k16 = m16.group(1)
                return k16 if (int(k16.split('.')[1]) in (8, 9, 10, 11) and k16 in _cd['src']) else None
            if _cd_scope == 'head':
                # T3A swaps net.head for an Identity and keeps the quantized head
                # as its own .classifier, so accept either name.
                _leaf = nm.split('.')[-1]
                return 'head' if (_leaf in ('head', 'classifier') and 'head' in _cd['src']) else None
            m = (_cd['KEY16'] if _cd_scope == 'b3_all' else _cd['KEY']).search(nm)
            if not m: return None
            k = m.group(1)
            return k if (int(k.split('.')[1]) in (8, 9, 10, 11) and k in _cd['src']) else None
        _cd['scale'] = {}; _cd['neg'] = {}
        def _cd_mkh(k, mod):
            sc, qm, nb = _gsq(mod)
            ns = _gns(mod)
            _cd['scale'][k] = (sc, qm, nb); _cd['neg'][k] = ns
            def hk(mm, inp):
                if sc is None or not inp: return
                x = inp[0]
                if not torch.is_tensor(x): return
                with torch.no_grad():
                    h = _cch(x.detach().flatten(), sc, qm, nb, neg_scale=ns).cpu()
                _cd['accum'][k] = h if k not in _cd['accum'] else _cd['accum'][k] + h
            return hk
        _ncd = 0
        # Some wrappers (the local branch-only holder) do not expose the backbone's own
        # submodules, so the head is unreachable from adapt_model; walk the
        # quantized net directly for that scope. Same module objects either way,
        # and this loop still runs after every wrapper has registered its hooks,
        # so what the hook sees is the post-recalibration input.
        # Try adapt_model first, which is what the measured runs use.
        # Some wrappers (the local branch-only holder) do not expose the backbone's own
        # submodules, leaving zero hooks; fall back to the quantized net in that
        # case only, so configs that already worked are untouched. Either way this
        # runs after every wrapper has registered its hooks, so the tensor the hook
        # sees is the one the module actually consumes.
        for _root in (adapt_model, net):
            for _nm, _mod in _root.named_modules():
                if not _isq(_mod): continue
                _k = _cd_canon(_nm)
                if _k is not None:
                    _mod.register_forward_pre_hook(_cd_mkh(_k, _mod)); _ncd += 1
            if _ncd: break
        logger.info(f"[codedist] scope={_cd_scope} hooks on {_ncd} interfaces")

    # Per-channel code histogram at a single interface, for the figure that shows the
    # displacement itself rather than the fraction of it a method removes. Same grid and
    # same rounding rule as the code-distribution metric; the only difference is that the
    # counts are kept per channel instead of pooled over the tensor, because pooling lets
    # channels moving in opposite directions cancel and the narrowing is what we want to show.
    _chd = {}
    if getattr(args, 'codehist_dump', ''):
        from build_qac_stats import (_is_quant_act as _isq_h, _get_quant_scale_qmax as _gsq_h,
                                     _get_neg_scale as _gns_h)
        _want = args.codehist_dump
        def _mk_chd(mod):
            sc, qm, nb = _gsq_h(mod); ns = _gns_h(mod); ncode = 1 << nb
            _iv = getattr(mod, 'a_interval', None)
            logger.info(f"[codehist] a_interval shape={tuple(_iv.shape) if torch.is_tensor(_iv) else None} "
                        f"numel={_iv.numel() if torch.is_tensor(_iv) else 0} "
                        f"min={float(_iv.min()) if torch.is_tensor(_iv) else 0:.6f} "
                        f"max={float(_iv.max()) if torch.is_tensor(_iv) else 0:.6f} mean_used={sc:.6f}")
            _chd['meta'] = dict(scale=sc, qmax=qm, n_bits=nb, neg_scale=ns, name=_want,
                                a_interval=(_iv.detach().cpu().clone() if torch.is_tensor(_iv) else None))
            def hk(mm, inp):
                if sc is None or not inp: return
                x = inp[0]
                if not torch.is_tensor(x): return
                with torch.no_grad():
                    xf = x.detach().float().reshape(-1, x.shape[-1])
                    if ns is not None:
                        pos = (xf / max(sc, 1e-8)).round_().clamp_(0, qm - 1)
                        neg = (xf / max(ns, 1e-8)).round_().clamp_(-qm, 0)
                        codes = (pos + neg).long()
                    else:
                        codes = (xf / max(sc, 1e-8)).round_().clamp_(-qm, qm - 1).long()
                    codes = (codes + qm).clamp_(0, ncode - 1)          # [N, C]
                    C = codes.shape[1]
                    if 'h' not in _chd:
                        _chd['h'] = torch.zeros(C, ncode, dtype=torch.float64)
                    flat = (torch.arange(C, device=codes.device).unsqueeze(0) * ncode + codes)
                    cnt = torch.bincount(flat.reshape(-1), minlength=C * ncode).double().cpu()
                    _chd['h'] += cnt.view(C, ncode)
            return hk
        _nch = 0
        for _root in (adapt_model, net):
            for _nm, _mod in _root.named_modules():
                if _isq_h(_mod) and _nm.endswith(_want):
                    _mod.register_forward_pre_hook(_mk_chd(_mod)); _nch += 1
            if _nch: break
        logger.info(f"[codehist] dumping per-channel histogram at '{_want}': {_nch} hook(s)")

    _sq = {}
    if getattr(args, 'sqnr_diag', False):
        import re as _re_sq
        from build_qac_stats import _is_quant_act as _isq_sq, _get_quant_scale_qmax as _gsq_sq
        # per-channel running sums on CPU float64: key -> [n(float), sum_x[C], sum_x2[C], sum_e2[C]]
        _sq['accum'] = {}     # reset each corruption
        _sq['overall'] = {}   # accumulated across all corruptions
        _sq['KEY'] = _re_sq.compile(r"(blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.fc1))")
        _sq['KEY16'] = _re_sq.compile(r"(blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2)))")
        _sq_scope = getattr(args, 'sqnr_scope', 'b3')
        def _sq_canon(nm):
            m = (_sq['KEY16'] if _sq_scope == 'b3_all' else _sq['KEY']).search(nm)
            if not m: return None
            k = m.group(1)
            return k if int(k.split('.')[1]) in (8, 9, 10, 11) else None
        def _sq_mkh(k, mod):
            sc, qm, nb = _gsq_sq(mod)
            def hk(mm, inp):
                if sc is None or not inp: return
                x = inp[0]
                if not torch.is_tensor(x): return
                with torch.no_grad():
                    xf = x.detach().float().reshape(-1, x.shape[-1])  # [N, C]
                    # Use the module's own quantizer when it exposes one: the
                    # post-GELU interfaces are twin-uniform, so a single-scale
                    # round() would not be the map they actually apply. SQNR is
                    # grid-agnostic, which is exactly why it can adjudicate the
                    # interfaces where the code-histogram metric cannot.
                    _qi = getattr(mod, 'quant_input', None)
                    if callable(_qi):
                        qx = _qi(x.detach()).float().reshape(-1, x.shape[-1])
                    else:
                        _s = sc if sc > 0 else 1e-8
                        qx = (xf / _s).round().clamp_(-qm, qm - 1) * _s
                    e = qx - xf
                    n = float(xf.shape[0])
                    s_x  = xf.sum(dim=0).double().cpu()
                    s_x2 = xf.pow(2).sum(dim=0).double().cpu()
                    s_e2 = e.pow(2).sum(dim=0).double().cpu()
                for _d in (_sq['accum'], _sq['overall']):
                    if k not in _d:
                        _d[k] = [n, s_x.clone(), s_x2.clone(), s_e2.clone()]
                    else:
                        _d[k][0] += n; _d[k][1] += s_x; _d[k][2] += s_x2; _d[k][3] += s_e2
            return hk
        _nsq = 0
        for _nm, _mod in adapt_model.named_modules():
            if not _isq_sq(_mod): continue
            _k = _sq_canon(_nm)
            if _k is not None: _mod.register_forward_pre_hook(_sq_mkh(_k, _mod)); _nsq += 1
        logger.info(f"[sqnr] per-channel SQNR hooks on {_nsq} interfaces")
    def _sqnr_summarize(accum):
        # returns (per_if_db{k}, per_if_lin{k}, if_mean_db, overall_mean_lin, overall_mean_db)
        import statistics as _stsq
        per_if_db = {}; per_if_lin = {}
        for k in sorted(accum.keys()):
            n, s_x, s_x2, s_e2 = accum[k]
            mu = s_x / n
            var = (s_x2 / n) - mu.pow(2)
            var = var.clamp_min(0.0)
            mse = (s_e2 / n).clamp_min(1e-30)
            sqnr_c = (var / mse).clamp_min(1e-30)          # per channel [C]
            per_if_lin[k] = float(sqnr_c.mean())            # mean over channels (linear)
            per_if_db[k]  = float((10.0 * sqnr_c.log10()).mean())  # mean over channels (dB)
        if not per_if_lin:
            return per_if_db, per_if_lin, float('nan'), float('nan'), float('nan')
        if_mean_db = _stsq.mean(per_if_db.values())         # dB-domain mean over interfaces
        overall_mean_lin = _stsq.mean(per_if_lin.values())  # linear mean over interfaces
        import math as _msq
        overall_mean_db = 10.0 * _msq.log10(max(overall_mean_lin, 1e-30))
        return per_if_db, per_if_lin, if_mean_db, overall_mean_lin, overall_mean_db
    if getattr(args, "mem_profile", False):
        # Baseline = model + adapter state already resident; peak during eval is
        # measured from here so the number is adaptation cost, not loader warmup.
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"[MEMPROF] post-build allocated_MiB="
                    f"{torch.cuda.memory_allocated() / (1024 ** 2):.1f}")
    for stream_round, stream_order_index, corrupt in stream_entries:
        args.corruption = corrupt
        if _cd: _cd['accum'].clear()
        if _sq: _sq['accum'].clear()
        logger.info(f"{args.corruption} (round={stream_round}, order_index={stream_order_index})")
        if (getattr(args, "qaffine_reset_each_corruption", False) or getattr(args, "qzero_reset_each_corruption", False)) and hasattr(adapt_model, "reset"):
            adapt_model.reset()
        # T3A maintains a support set that is implicitly episodic per corruption;
        # without reset, supports leak across domains and make results order-dependent.
        if args.algorithm == 't3a' and hasattr(adapt_model, "reset"):
            adapt_model.reset()
        # Cascade reset to attached the local branch/CLS holders (for FOZO/ZOA + Ours composition).
        # Skip when --continual: state must persist across corruptions in continual mode.
        if not getattr(args, "continual", False):
            for _holder_attr in ("_quar_holder", "_cls_shift_holder"):
                _holder = getattr(adapt_model, _holder_attr, None)
                if _holder is not None and hasattr(_holder, "reset"):
                    _holder.reset()

        if args.corruption == 'rendition':
            adapt_model.imagenet_mask = imagenet_r_mask
        elif args.corruption == 'imagenet_a':
            adapt_model.imagenet_mask = imagenet_a_mask
        else:
            adapt_model.imagenet_mask = None
        if hasattr(adapt_model, "current_corruption"):
            adapt_model.current_corruption = args.corruption
        elif hasattr(adapt_model, "_log_fozo_qreg_metrics"):
            setattr(adapt_model, "current_corruption", args.corruption)

        val_dataset, val_loader = prepare_test_data(args)

        # Rounds within corruption: state carries across rounds (no inter-round reset).
        # First round is mirrored to corruption_metrics.csv for backward compat;
        # all rounds are logged to rounds_metrics.csv when rounds_per_corruption > 1.
        last_top1 = None; last_top5 = None; last_ece = None; last_ms = None
        for corr_round in range(1, rounds_per_corr + 1):
            torch.cuda.empty_cache()
            top1, top5, ece_loss, steady_forward_ms = validate_adapt(val_loader, adapt_model, args)
            if _cd and _cd['accum']:
                import statistics as _st
                _ds = []; _per = {}; _w1s = []; _w1per = {}
                for _k, _h in _cd['accum'].items():
                    _ht = _h / (_h.sum() + 1e-12); _hs = _cd['src'][_k].float(); _hs = _hs / (_hs.sum() + 1e-12)
                    _d = float((_ht - _hs).abs().sum()); _ds.append(_d); _per[_k] = _d
                    # Value-space Wasserstein-1. The L1 above counts "moved one
                    # bin" the same everywhere, but on the twin-uniform post-GELU
                    # grid a negative bin is 0.0053 wide and a positive one up to
                    # 0.9, so a 170x smaller value change scores the same. W1
                    # de-quantizes the code axis first, so every move is weighted
                    # by how far the value actually travelled.
                    _sc, _qm, _nb = _cd['scale'][_k]
                    _ns = _cd['neg'][_k]
                    _codes = torch.arange(-_qm, _qm, dtype=torch.float64)
                    _vals = torch.where(_codes >= 0, _codes * _sc,
                                        _codes * (_ns if _ns is not None else _sc))
                    _cdf = torch.cumsum(_ht.double() - _hs.double(), dim=0)[:-1]
                    _w1 = float((_cdf.abs() * _vals.diff()).sum())
                    _w1s.append(_w1); _w1per[_k] = _w1
                logger.info(f"[CODEDIST] {args.corruption} = {_st.mean(_ds):.4f} (n={len(_ds)})")
                # Per-interface distances, so one b3_all run yields both the
                # 12-interface aggregate (excluding mlp.fc2) and the 16-interface one.
                logger.info("[CODEDIST-PER] " + args.corruption + " " +
                            " ".join(f"{_k}={_v:.6f}" for _k, _v in sorted(_per.items())))
                # Edge-bin occupancy: mass in the lowest and highest code bins,
                # test vs source. A per-channel affine that pushes post-GELU
                # values past the negative grid's floor shows up as a spike in
                # the lowest bin that the source does not have.
                _sat = []
                for _k, _h in sorted(_cd['accum'].items()):
                    _ht = _h / (_h.sum() + 1e-12); _hs = _cd['src'][_k].float(); _hs = _hs / (_hs.sum() + 1e-12)
                    _sat.append(f"{_k}=lo{float(_ht[0]):.5f}/{float(_hs[0]):.5f}"
                                f",hi{float(_ht[-1]):.5f}/{float(_hs[-1]):.5f}")
                logger.info("[CODEDIST-SAT] " + args.corruption + " " + " ".join(_sat))
                logger.info(f"[CODEDIST-W1] {args.corruption} = {_st.mean(_w1s):.6f} (n={len(_w1s)})")
                logger.info("[CODEDIST-W1-PER] " + args.corruption + " " +
                            " ".join(f"{_k}={_v:.6f}" for _k, _v in sorted(_w1per.items())))
            if _chd.get('h') is not None:
                _chp = os.path.join(args.output, f"codehist_{_chd['meta']['name'].replace('.', '_')}.pt")
                torch.save({'hist': _chd['h'], **_chd['meta'], 'corruption': args.corruption}, _chp)
                logger.info(f"[CODEHIST] saved {tuple(_chd['h'].shape)} to {_chp}")
                _chd.pop('h')
            if _sq and _sq['accum']:
                _db, _lin, _ifdb, _ovlin, _ovdb = _sqnr_summarize(_sq['accum'])
                logger.info(
                    f"[SQNR] {args.corruption} interface_mean_dB={_ifdb:.4f} "
                    f"overall_mean_dB={_ovdb:.4f} overall_mean_linear={_ovlin:.6f} (n_if={len(_db)})")
                logger.info(
                    f"[SQNR-BREAKDOWN] {args.corruption} "
                    + " ".join(f"{k}={_db[k]:.2f}dB" for k in sorted(_db)))
            if rounds_per_corr > 1:
                logger.info(
                    f"Under shift type {args.corruption} corr_round={corr_round}/{rounds_per_corr} "
                    f"After {args.algorithm} Top-1 Accuracy: {top1:.6f}"
                )
                with open(rounds_csv_path, "a") as f:
                    f.write(
                        f"{stream_round},{stream_order_index},{corr_round},{args.corruption},"
                        f"{float(top1):.6f},{float(top5):.6f},{float(steady_forward_ms):.6f}\n"
                    )
            last_top1, last_top5, last_ece, last_ms = top1, top5, ece_loss, steady_forward_ms

        top1, top5, ece_loss, steady_forward_ms = last_top1, last_top5, last_ece, last_ms
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.6f} and Top-5 Accuracy: {top5:.6f} and ECE: {ece_loss:.6f}")
        logger.info(f"Under shift type {args.corruption} steady batch time: {steady_forward_ms:.3f} ms")
        if getattr(args, "mem_profile", False):
            logger.info(
                f"[MEMPROF] {args.corruption} "
                f"peak_allocated_MiB={torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} "
                f"peak_reserved_MiB={torch.cuda.max_memory_reserved() / (1024 ** 2):.1f}")
        with open(corruption_metrics_path, "a") as f:
            f.write(
                f"{stream_round},{stream_order_index},{args.corruption},{float(top1):.6f},{float(top5):.6f},"
                f"{float(ece_loss) * 100.0:.6f},{float(steady_forward_ms):.6f}\n"
            )
        corrupt_acc.append(top1)
        corrupt_ece.append(ece_loss)

        # reset model before adapting on the next domain
        if args.continual == True:
            pass
        elif args.algorithm == 'no_adapt' and not isinstance(adapt_model, NoAdaptQACWrapper) \
                and not getattr(args, "quar_global_enable", False) \
                and not getattr(args, "neo_enable", False):
            pass  # raw no_adapt: no state to reset
        else:
            adapt_model.reset()
            # pass
        
        logger.info(f'mean acc of corruption: {sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0}')
        logger.info(f'mean ece of corruption: {sum(corrupt_ece)/len(corrupt_ece)*100 if len(corrupt_ece) else 0}')
        logger.info(f'corrupt acc list: {[_.item() for _ in corrupt_acc]}')
        logger.info(f'corrupt ece list: {[_*100 for _ in corrupt_ece]}')

    if _sq and _sq['overall']:
        _db, _lin, _ifdb, _ovlin, _ovdb = _sqnr_summarize(_sq['overall'])
        logger.info(
            f"[SQNR-OVERALL] ALL_CORRUPTIONS interface_mean_dB={_ifdb:.4f} "
            f"overall_mean_dB={_ovdb:.4f} overall_mean_linear={_ovlin:.6f} (n_if={len(_db)})")
        logger.info(
            f"[SQNR-OVERALL-BREAKDOWN] "
            + " ".join(f"{k}={_db[k]:.2f}dB" for k in sorted(_db)))

    if _mismatch_collector is not None:
        _mp = os.path.join(args.output, "qac_mismatch_stats.pt")
        _mismatch_collector.dump(_mp)
        _mismatch_collector.remove()
        logger.info(f"[mismatch-diag] saved per-layer pre-quant + post-DQ stats to {_mp}")

    mean_acc = sum(corrupt_acc)/len(corrupt_acc)
    mean_ece = sum(corrupt_ece)/len(corrupt_ece)*100
    new_log_name = f"{log_name.replace('-log.txt', '')}_acc={mean_acc:.2f}_ece={mean_ece:.2f}.txt"
    new_log_path = os.path.join(args.output, new_log_name)
    os.rename(original_log_path, new_log_path)
