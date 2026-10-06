#!/usr/bin/env bash
# Source statistics + PTQ4ViT checkpoints for the CIFAR-10/100 extension (non-ImageNet benchmark).
# Per (dataset, bit-width): Hessian-calibrate PTQ4ViT on 32 CIFAR TRAIN images (saved as a full-net
# checkpoint), then collect the all-layer local stats and the CLS (global) stats from a seeded 12,800-image
# subset of the TRAIN split (200 batches x 64, the same count as the ImageNet protocol; train split so the
# source statistics never see the test images that CIFAR-C is built from).
# Backbones: public HF ViT-B/16 fine-tuned on CIFAR (converted by tools/models/convert_hf_vit_to_timm.py).
# BWS="6 8" (default, PTQ4ViT Hessian calibration here) or BWS="4 3" (AdaLog: checkpoints must already exist from
# tools/stats/build_adalog_ckpt.py as checkpoints/w{4,3}a{4,3}_checkpoint_opt_<ds>.pt; stats are then collected from them).
set -uo pipefail
PY="${PY:-python}"; QUAR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QUAR"
D="${DATA_ROOT:-/path/to/data}"
DATA_CIFAR10="${DATA_CIFAR10:-$D/cifar10}";       DATA_CIFAR10_C="${DATA_CIFAR10_C:-$D/cifar10-c}"
DATA_CIFAR100="${DATA_CIFAR100:-$D/cifar100}";     DATA_CIFAR100_C="${DATA_CIFAR100_C:-$D/cifar100-c}"
DATA_DN126="${DATA_DN126:-$D/domainnet126/real_src}"; DATA_DN126_C="${DATA_DN126_C:-$D/domainnet126/targets}"   # source domain (real) / target domains
LOG="${LOG:-runs/cifar_stats_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$LOG"
declare -A NC=( [cifar10]=10 [cifar100]=100 [domainnet126]=126 )
declare -A CK=( [cifar10]=checkpoints/vit_base_patch16_224_cifar10_nateraw_timm.pth [cifar100]=checkpoints/vit_base_patch16_224_cifar100_pkr7098_timm.pth
                [domainnet126]=checkpoints/vit_base_patch16_224_domainnet126_real_timm.pth )
declare -A DDATA=( [cifar10]=$DATA_CIFAR10 [cifar100]=$DATA_CIFAR100 [domainnet126]=$DATA_DN126 )
DSS="${DSS:-cifar10 cifar100}"
BWS="${BWS:-6 8}"
qflags(){ local ds=$1 bw=$2; case "$bw" in
  8|6) echo "--quant --quant_w_bit $bw --quant_a_bit $bw --save_ptq_fullnet checkpoints/ptq4vit_w${bw}a${bw}_${ds}.pt";;
  4|3) echo "--quant_w_bit $bw --quant_a_bit $bw --adalog_checkpoint checkpoints/w${bw}a${bw}_checkpoint_opt_${ds}.pt";;
  *) echo BADBW; return 1;; esac; }
cflags(){ local ds=$1 bw=$2; case "$bw" in
  8|6) echo "--quant --quant_w_bit $bw --quant_a_bit $bw --ptq_checkpoint checkpoints/ptq4vit_w${bw}a${bw}_${ds}.pt";;
  4|3) echo "--quant_w_bit $bw --quant_a_bit $bw --adalog_checkpoint checkpoints/w${bw}a${bw}_checkpoint_opt_${ds}.pt";;
  *) echo BADBW; return 1;; esac; }
gpu=${GPU0:-0}
for ds in $DSS; do for bw in $BWS; do
  tag=${ds}_w${bw}a${bw}
  (
    CUDA_VISIBLE_DEVICES=$gpu $PY -u tools/stats/build_qac_stats.py --data ${DDATA[$ds]} --dataset $ds --num_classes ${NC[$ds]} \
      --arch vit_base_patch16_224 --arch_checkpoint ${CK[$ds]} $(qflags $ds $bw) \
      --quar_local_scope all --max_batches 200 --batch_size 64 --workers 8 --source_split train --source_samples 12800 --seed 0 \
      --output stats/qac_stats_w${bw}a${bw}_all_layers_${ds}_seed0.pt > "$LOG/qac_$tag.log" 2>&1 \
    && CUDA_VISIBLE_DEVICES=$gpu $PY -u tools/stats/build_cls_stats.py --data ${DDATA[$ds]} --dataset $ds --num_classes ${NC[$ds]} \
      --arch vit_base_patch16_224 --arch_checkpoint ${CK[$ds]} $(cflags $ds $bw) \
      --max_batches 200 --batch_size 64 --workers 8 --source_split train --source_samples 12800 --seed 0 \
      --output stats/cls_stats_w${bw}a${bw}_${ds}_seed0.pt > "$LOG/cls_$tag.log" 2>&1 \
    && echo "[$tag] done" >> "$LOG/status.txt" || echo "[$tag] FAIL" >> "$LOG/status.txt"
  ) &
  gpu=$((gpu+1))
done; done
wait; echo "ALL_DONE" >> "$LOG/status.txt"; echo "$LOG"
