#!/usr/bin/env python
"""Convert a HuggingFace ViTForImageClassification checkpoint to a timm vit_*_patch16_224 state_dict.

Used for the public CIFAR-10/100 fine-tuned ViT-B/16 checkpoints (google/vit-base-patch16-224-in21k
lineage, mean/std 0.5, 224 input). The converted state_dict loads strictly into
timm.create_model(arch, pretrained=False, num_classes=N). Note: the HF models use LayerNorm eps 1e-12
while timm uses 1e-6; the difference is numerically negligible and is validated by the clean-accuracy
check below.

Example:
    python tools/models/convert_hf_vit_to_timm.py --repo nateraw/vit-base-patch16-224-cifar10 \
        --num_classes 10 --out checkpoints/vit_base_patch16_224_cifar10_nateraw_timm.pth \
        --val_dir /path/to/data/cifar10/val
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def download_state_dict(repo):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(repo)
    if 'model.safetensors' in files:
        p = hf_hub_download(repo, 'model.safetensors')
        from safetensors.torch import load_file
        return load_file(p)
    p = hf_hub_download(repo, 'pytorch_model.bin')
    return torch.load(p, map_location='cpu')


def convert(hf_sd, depth=12):
    sd = {}
    g = lambda k: hf_sd[k]
    pre = 'vit.' if any(k.startswith('vit.') for k in hf_sd) else ''
    sd['cls_token'] = g(f'{pre}embeddings.cls_token')
    sd['pos_embed'] = g(f'{pre}embeddings.position_embeddings')
    sd['patch_embed.proj.weight'] = g(f'{pre}embeddings.patch_embeddings.projection.weight')
    sd['patch_embed.proj.bias'] = g(f'{pre}embeddings.patch_embeddings.projection.bias')
    for i in range(depth):
        b = f'{pre}encoder.layer.{i}.'
        sd[f'blocks.{i}.norm1.weight'] = g(b + 'layernorm_before.weight')
        sd[f'blocks.{i}.norm1.bias'] = g(b + 'layernorm_before.bias')
        sd[f'blocks.{i}.attn.qkv.weight'] = torch.cat([g(b + f'attention.attention.{n}.weight') for n in ('query', 'key', 'value')], 0)
        sd[f'blocks.{i}.attn.qkv.bias'] = torch.cat([g(b + f'attention.attention.{n}.bias') for n in ('query', 'key', 'value')], 0)
        sd[f'blocks.{i}.attn.proj.weight'] = g(b + 'attention.output.dense.weight')
        sd[f'blocks.{i}.attn.proj.bias'] = g(b + 'attention.output.dense.bias')
        sd[f'blocks.{i}.norm2.weight'] = g(b + 'layernorm_after.weight')
        sd[f'blocks.{i}.norm2.bias'] = g(b + 'layernorm_after.bias')
        sd[f'blocks.{i}.mlp.fc1.weight'] = g(b + 'intermediate.dense.weight')
        sd[f'blocks.{i}.mlp.fc1.bias'] = g(b + 'intermediate.dense.bias')
        sd[f'blocks.{i}.mlp.fc2.weight'] = g(b + 'output.dense.weight')
        sd[f'blocks.{i}.mlp.fc2.bias'] = g(b + 'output.dense.bias')
    sd['norm.weight'] = g(f'{pre}layernorm.weight')
    sd['norm.bias'] = g(f'{pre}layernorm.bias')
    sd['head.weight'] = g('classifier.weight')
    sd['head.bias'] = g('classifier.bias')
    used = set()
    unused = [k for k in hf_sd if not (k.startswith(f'{pre}embeddings') or k.startswith(f'{pre}encoder.layer')
                                       or k.startswith(f'{pre}layernorm') or k.startswith('classifier'))]
    return sd, unused


@torch.no_grad()
def clean_accuracy(net, val_dir, batch_size=128, workers=8):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
                             transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.ImageFolder(val_dir, tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    net = net.cuda().eval()
    correct = n = 0
    for x, y in dl:
        p = net(x.cuda(non_blocking=True)).argmax(1).cpu()
        correct += (p == y).sum().item(); n += y.numel()
    return 100.0 * correct / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--arch', default='vit_base_patch16_224')
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_dir', default=None, help='ImageFolder of the clean test split for an accuracy check')
    args = ap.parse_args()

    import timm
    hf_sd = download_state_dict(args.repo)
    sd, unused = convert(hf_sd)
    print(f'[convert] {args.repo}: {len(hf_sd)} HF tensors -> {len(sd)} timm tensors; unused HF keys: {unused}')
    net = timm.create_model(args.arch, pretrained=False, num_classes=args.num_classes)
    net.load_state_dict(sd, strict=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(sd, args.out)
    print(f'[save] {args.out}')
    if args.val_dir:
        acc, n = clean_accuracy(net, args.val_dir)
        print(f'[check] clean top-1 on {args.val_dir}: {acc:.2f}% ({n} images)')


if __name__ == '__main__':
    main()
