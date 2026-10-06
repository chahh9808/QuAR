"""SPA (Self-Bootstrapping for Versatile TTA, Niu et al., ICML 2025) on our
quantized ViT.

Upstream: https://github.com/mr-eggplant/SPA. The algorithm itself is vendored
byte-for-byte in tta_library/spa_official.py (plus spa_maskers.py, spa_byol.py);
this file only reproduces the wiring that upstream main.py does at lines
268-281, against our W6A6 PTQ4ViT ViT-B instead of the FP timm ViT.

What SPA costs per batch (spa_official.forward_and_adapt_eata):
    1. outputs = model(x)                              <- the returned prediction
    2. model(low-freq amplitude-masked x, predictor)   <- consistency branch
    3. model(learnable-patch-erased x, predictor)      <- consistency branch
    loss.backward() through 2+3 (+ the optional TENT/EATA auxiliary on 1),
    then optimizer.step() on {LN affine, BYOL predictor} and a second SGD step
    on the patch-erasing alpha.
So: 3F + 1B per batch, two optimizers, and an auxiliary 768x768 FP predictor.
The predictor is used ONLY by the consistency branches; the returned logits go
through the plain path, so it does not alter the deployed forward structure.

Deviations from upstream, all forced by the quantized graph or by consistency
with our existing backprop baselines, and none touching SPA's algorithm:

  * install_ste(net) before wrapping. PTQ4ViT's activation fake-quant hard-rounds
    with no straight-through estimator, so gradients would be exactly zero
    (see backprop_tta.py). TENT/EATA/SAR in Table 8 get the same treatment.
  * Fisher for the 'spa+eata' variant is computed with our build_eata procedure
    (on the live net, via build_clean_source_loader) rather than upstream
    get_fisher's deepcopy path, so the EATA term matches the EATA row exactly.
    Same official constants: fisher_alpha, e_margin, d_margin.
  * projector_dim is read from net.head.in_features. On the quantized net head
    is a PTQSLBatchingQuantLinear, which subclasses nn.Linear, so .in_features
    is present and equals 768.

Verified on the W6A6 checkpoint before wiring this up: net exposes
forward_features / fc_norm (Identity) / head (in_features=768); net.global_pool
is a str so BYOLWrapper takes its ViT branch; .train() leaves all 74 quantizer
.mode attributes at 'quant_forward' and the model has no dropout with p>0; and
with the eye-initialised predictor the use_predictor path reproduces the plain
logits to max|diff| = 0.
"""

import torch

from tta_library.spa import official as spa_official
from tta_library.spa.byol import BYOLWrapper
from tta_library.backprop_tta import install_ste, build_clean_source_loader

# Upstream ImageNet/ViT-B defaults (SPA main.py lines 274-276).
_VIT_LR_LN = 0.01
_VIT_LR_PREDICTOR = 0.05
_VIT_MASK_RATIO = 0.2
_VIT_NOISE_RATIO = 0.4


def _compute_fisher(args, net, obtain_train_loader, logger=None):
    """Official EATA Fisher on clean source data, same procedure as build_eata."""
    import torch.nn as nn

    _, fisher_loader = build_clean_source_loader(args, obtain_train_loader,
                                                 args.eata_fisher_size)
    params = [p for p in net.parameters() if p.requires_grad]
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    n_batches = len(fisher_loader)
    with torch.enable_grad():
        for iter_, (images, _targets) in enumerate(fisher_loader, start=1):
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
        logger.info(f"[SPA] computed EATA Fisher over {n_batches} clean batches "
                    f"({len(fishers)} param tensors)")
    del ewc_optimizer
    return fishers


def build_spa(args, net, obtain_train_loader=None, logger=None):
    """Wrap the quantized ViT with official SPA.

    args.algorithm selects the variant, matching upstream use_spa_with_others:
    'spa', 'spa+tent', 'spa+eta', 'spa+eata'.
    """
    variant = args.algorithm
    assert variant in ('spa', 'spa+tent', 'spa+eta', 'spa+eata'), \
        f"unsupported SPA variant: {variant}"

    install_ste(net)  # gradients through PTQ4ViT activation quant

    projector_dim = net.head.in_features
    net = BYOLWrapper(net, projector_dim=projector_dim).cuda()
    net = spa_official.configure_model(net)
    params, param_names = spa_official.collect_params(net)
    # Upstream's learning rates are set for its protocol batch size of 64, and SPA steps its
    # optimizers once per batch. Run at a smaller batch and the same rates are applied over
    # 64/bs as many steps on the same stream, which diverges: at batch 1 on Jetson this drove
    # W6A6 ViT-B to 12.3% top-1 against 42.7% unadapted, and 45.9% once scaled. Scale linearly
    # with the batch, exactly as backprop_tta.official_vit_lr already does for TENT/EATA/SAR.
    # The factor is 1.0 at the protocol batch, so nothing about batch 64 changes.
    _bs_scale = float(getattr(args, 'batch_size', 64)) / 64.0
    optimizer = torch.optim.SGD([
        {'params': params, 'lr': _VIT_LR_LN * _bs_scale, 'momentum': 0.9},
        {'params': net.predictor.parameters(), 'lr': _VIT_LR_PREDICTOR * _bs_scale, 'momentum': 0.9},
    ])
    model = spa_official.SPA(net, optimizer,
                             noise_ratio=_VIT_NOISE_RATIO,
                             freq_mask_ratio=_VIT_MASK_RATIO)

    # upstream use_spa_with_others (main.py lines 110-124)
    if 'tent' in variant:
        model.use_tent = True
    if 'eata' in variant or 'eta' in variant:
        model.use_eata = True
        fishers = None
        if 'eata' in variant:
            assert obtain_train_loader is not None, "spa+eata needs the source loader"
            fishers = _compute_fisher(args, net, obtain_train_loader, logger=logger)
        model.fishers = fishers
        model.fisher_alpha = args.eata_fisher_alpha
        model.current_model_probs = None
        model.e_margin = args.eata_e_margin
        model.d_margin = args.eata_d_margin

    model.imagenet_mask = None  # the eval loop overwrites this per corruption
    if logger is not None:
        n_ln = len(param_names)
        n_pred = sum(p.numel() for p in net.predictor.parameters())
        logger.info(f"[SPA] variant={variant} ln_params={n_ln} "
                    f"predictor_params={n_pred} lr_ln={_VIT_LR_LN * _bs_scale} "
                    f"lr_pred={_VIT_LR_PREDICTOR * _bs_scale} mask_ratio={_VIT_MASK_RATIO} "
                    f"noise_ratio={_VIT_NOISE_RATIO}")
    return model, param_names
