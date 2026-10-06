"""ImageNet / ImageNet-C loaders with the RobustBench subset selection.

Source: https://github.com/mala-lab/ZOA, dataset/selectedRotateImageFolder.py
(MIT, (c) 2025 DengZeshuai), itself derived from the SAR and EATA releases. Extended here with
the ImageNet-R, Sketch, C-bar and 3DCC paths the generalization appendix needs.
"""
import os
import copy
import random
import math
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from .ImagenetV2 import ImageNetV2Dataset


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
									transforms.ToTensor(),
									normalize])
te_transforms = transforms.Compose([transforms.Resize(256),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize])
te_transforms_imageC = transforms.Compose([transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize])

rotation_tr_transforms = tr_transforms
rotation_te_transforms = te_transforms

# Non-ImageNet ImageFolder datasets (additive). Images are stored at their
# native size (CIFAR: 32x32) and resized straight to 224x224 at load time, matching the
# HF ViTImageProcessor used to fine-tune the public CIFAR ViT checkpoints (bilinear,
# mean/std 0.5). Selected by args.dataset; 'imagenet' (default) keeps every existing path.
small_image_datasets = ('cifar10', 'cifar100')
te_transforms_small = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
									transforms.ToTensor(),
									normalize])

# Per-backbone input normalization. The augreg ViT-B/ViT-S expect mean/std 0.5, DeiT-B expects the
# ImageNet statistics; the evaluation loader follows each backbone's own transform via timm's
# resolve_data_config, as the PTQ calibration loader does. `_with_normalization` returns the transform
# UNCHANGED whenever the backbone's statistics are the 0.5 convention.
_ARCH_NORM_CACHE = {}

def normalization_for_arch(arch):
	"""(mean, std) the backbone was trained with, from timm's pretrained config.

	Anything unresolvable falls back to the 0.5 convention this file used for all backbones before."""
	key = str(arch)
	if key not in _ARCH_NORM_CACHE:
		mean = std = (0.5, 0.5, 0.5)
		try:
			import timm
			cfg = timm.models.get_pretrained_cfg(key)
			mean, std = tuple(cfg.mean), tuple(cfg.std)
		except Exception:
			pass
		_ARCH_NORM_CACHE[key] = (mean, std)
	return _ARCH_NORM_CACHE[key]

def _with_normalization(tf, mean, std):
	"""Rebuild a Compose with a different Normalize. No-op for the 0.5 convention."""
	if tf is None or (tuple(mean) == (0.5, 0.5, 0.5) and tuple(std) == (0.5, 0.5, 0.5)):
		return tf
	steps = [t for t in tf.transforms if not isinstance(t, transforms.Normalize)]
	return transforms.Compose(steps + [transforms.Normalize(mean=list(mean), std=list(std))])

def arch_normalized(args, tf):
	"""Apply the backbone's own normalization on the ImageNet-family paths.

	CIFAR and DomainNet keep 0.5: their fine-tuned checkpoints were produced with it
	(tools/models/convert_hf_vit_to_timm.py, tools/prep/finetune_vit_imagefolder.py)."""
	if getattr(args, 'dataset', 'imagenet') != 'imagenet':
		return tf
	return _with_normalization(tf, *normalization_for_arch(getattr(args, 'arch', 'vit_base_patch16_224')))

def eval_transform_for(args, default):
	"""Return the eval transform for args.dataset; ImageNet-family datasets keep `default`."""
	ds = getattr(args, 'dataset', 'imagenet')
	if ds in small_image_datasets:
		return te_transforms_small
	if ds == 'domainnet126':
		# natural images of varying size: Resize(256)+CenterCrop(224) for source AND target domains
		# (the ImageNet-C branch would otherwise CenterCrop(224) without resizing)
		return te_transforms
	return default

def calib_transform_override(args):
	"""Transform override for the PTQ calibration loader (None = timm's own ImageNet transform)."""
	ds = getattr(args, 'dataset', 'imagenet')
	if ds in small_image_datasets:
		return te_transforms_small
	if ds == 'domainnet126':
		return te_transforms
	return None

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
# ImageNet-C held-out (extra/validation) corruptions — additive; recognized like the standard 15
# so the loader routes them to data_corruption/<corruption>/<level>. Not added to main.py's
# all_corruptions, so they only run when explicitly passed via --corruption.
held_out_corruptions = ['gaussian_blur', 'saturate', 'spatter', 'speckle_noise']
common_corruptions = common_corruptions + held_out_corruptions
# ImageNet-C-bar (algorithmic corruptions, Mintun et al. 2021) — additive; recognized
# like the standard 15 so the loader routes them to data_corruption/<corruption>/<level>.
# Severities are remapped to integer levels 1-5 on disk. Not in main.py's all_corruptions,
# so they only run when explicitly passed via --corruption.
c_bar_corruptions = ['caustic_refraction', 'inverse_sparkles', 'sparkles', 'perlin_noise',
                     'blue_noise_sample', 'plasma_noise', 'checkerboard_cutout',
                     'cocentric_sine_waves', 'single_frequency_greyscale', 'brownish_noise']
common_corruptions = common_corruptions + c_bar_corruptions
# DomainNet-126 target domains (additive). Layout mirrors ImageNet-C: data_corruption/<dn_domain>/<level>/<class>/;
# the source (real) lives at --data/{train,val}. Prefixed dn_ so 'sketch' does not collide with ImageNet-Sketch.
domainnet_domains = ['dn_clipart', 'dn_painting', 'dn_sketch', 'dn_real']
common_corruptions = common_corruptions + domainnet_domains
# ImageNet-3DCC (3D Common Corruptions, Kar et al. CVPR 2022) — additive. Pre-generated 224x224
# images with the SAME on-disk layout as ImageNet-C (<corruption>/<level 1-5>/<wnid>/<file>),
# so the loader routes them to data_corruption/<corruption>/<level> with no remapping. Not in
# main.py's all_corruptions, so they only run when explicitly passed via --corruption.
threed_cc_corruptions = ['near_focus', 'far_focus', 'fog_3d', 'flash', 'color_quant',
                         'low_light', 'xy_motion_blur', 'z_motion_blur', 'iso_noise',
                         'bit_error', 'h265_abr', 'h265_crf']
# Full official ImageNet-3DCC set (12 corruptions). The 3 video-codec ones (bit_error,
# h265_abr, h265_crf) have ~87-96 GB tarballs (temporal frame sequences); downloaded via
# multi-connection aria2c. All sev-level-5, ImageNet-C-style layout, reuse the 5k-eval path.
common_corruptions = common_corruptions + threed_cc_corruptions


class ImagePathFolder(datasets.ImageFolder):
	def __init__(self, traindir, train_transform):
		super(ImagePathFolder, self).__init__(traindir, train_transform)	

	def __getitem__(self, index):
		path, _ = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		path, pa = os.path.split(path)
		path, pb = os.path.split(path)
		return img, 'val/%s/%s' %(pb, pa)


# =========================Rotate ImageFolder Preparations Start======================
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label='rand'):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels


# =========================Rotate ImageFolder Preparations End======================


# The following ImageFolder supports sample a subset from the entire dataset by index/classes/sample number, at any time after the dataloader created. 
class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_transform, original=True, rotation=True, rotation_transform=None):
        super(SelectedRotateImageFolder, self).__init__(root, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform

        self.original_samples = self.samples
        random.shuffle(self.samples)

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path, target = self.samples[index]
        img_input = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input

        results = []
        if self.original:
            results.append(img)
            results.append(target)
        if self.rotation:
            if self.rotation_transform is not None:
                img = self.rotation_transform(img_input)
            target_ssh = np.random.randint(0, 4, 1)[0]
            img_ssh = rotate_single_with_label(img, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        return results

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def set_target_class_dataset(self, target_class_index, logger=None):
        self.target_class_index = target_class_index
        self.samples = [(path, idx) for (path, idx) in self.original_samples if idx in self.target_class_index]
        self.targets = [s[1] for s in self.samples]

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = [self.original_samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]


def reset_data_sampler(sampler, dset_length, dset):
    sampler.dataset = dset
    if dset_length % sampler.num_replicas != 0 and False:
        sampler.num_samples = math.ceil((dset_length - sampler.num_replicas) / sampler.num_replicas)
    else:
        sampler.num_samples = math.ceil(dset_length / sampler.num_replicas)
    sampler.total_size = sampler.num_samples * sampler.num_replicas


def prepare_train_dataset(args):
    print('Preparing data...')
    traindir = os.path.join(args.data, 'train')
    trset = SelectedRotateImageFolder(traindir, tr_transforms, original=True, rotation=args.rotation,
                                                        rotation_transform=rotation_tr_transforms)
    return trset


def prepare_train_dataloader(args, trset=None, sampler=None):
    if sampler is None:
        trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)
        train_sampler = None
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trset)
        trloader = torch.utils.data.DataLoader(
            trset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True) #sampler=None shuffle=True,
    return trloader, train_sampler

def dirichlet_label_shift_order(item_list, delta=0.1, seed=0):
    """Reorder (path, label) items into a non-IID 'online imbalanced label shift' stream.

    Standard wild-TTA protocol (e.g., SAR/NOTE): draw a per-class Dirichlet(delta) weight
    vector; emit samples by repeatedly sampling a still-nonempty class proportional to its
    weight. Small delta => a few classes dominate early => strong temporal class imbalance.
    Labels are used ONLY for ordering, never for adaptation. Deterministic given seed.
    """
    rng = np.random.RandomState(seed)
    cls_to_idxs = {}
    for i, (_, lbl) in enumerate(item_list):
        cls_to_idxs.setdefault(int(lbl), []).append(i)
    classes = sorted(cls_to_idxs.keys())
    for c in classes:
        rng.shuffle(cls_to_idxs[c])  # break intra-class order
    weights = rng.dirichlet([delta] * len(classes))
    w = {c: weights[k] for k, c in enumerate(classes)}
    ptr = {c: 0 for c in classes}
    order = []
    remaining = list(classes)
    while remaining:
        wv = np.array([w[c] for c in remaining], dtype=np.float64)
        s = wv.sum()
        wv = (np.ones_like(wv) / len(wv)) if s <= 0 else (wv / s)
        c = remaining[rng.choice(len(remaining), p=wv)]
        order.append(cls_to_idxs[c][ptr[c]])
        ptr[c] += 1
        if ptr[c] >= len(cls_to_idxs[c]):
            remaining.remove(c)
    return [item_list[i] for i in order]


def sar_label_shift_order(labels, imbalance_ratio=500000, T=100000, seed=2023):
    """SAR (Niu et al., ICLR'23) online imbalanced label-shift sample order.

    Faithful port of SAR's generate_shifted_sample_indices (per_class_shift): the stream is
    1000 'blocks' in a shuffled class order; in block b a 'major' class has prob
    major=ir/(ir+C-1) and every other class has minor=1/(ir+C-1); each block emits T/C
    time-steps, sampling a class from that peaked distribution and then a random image of that
    class (with replacement). ir=1 is uniform (IID); ir=500000 is ~pure class order (sorted).
    Returns a list of length T of indices into `labels`. Labels used only for ordering.
    """
    import random as _random
    rnd = _random.Random(seed)         # SAR: random.seed(seed) for class-order shuffle
    rng = np.random.RandomState(2022)  # SAR: np.random.seed(2022) for ys + image sampling
    cls_to_idxs = {}
    for i, l in enumerate(labels):
        cls_to_idxs.setdefault(int(l), []).append(i)
    classes = sorted(cls_to_idxs.keys())
    C = len(classes)
    minor = 1.0 / (imbalance_ratio + C - 1)
    major = minor * imbalance_ratio
    block_order = list(range(C)); rnd.shuffle(block_order)  # shuffled class order across blocks
    reps = max(1, T // C)
    order = []
    for b in block_order:
        p = np.full(C, minor, dtype=np.float64); p[b] = major; p = p / p.sum()
        ys = rng.choice(C, size=reps, p=p)                  # class (index into `classes`) per step
        for y in ys:
            idxs = cls_to_idxs[classes[int(y)]]
            order.append(idxs[rng.randint(0, len(idxs))])   # random image of that class (w/ replacement)
    return order


class ImageNetSubsetFromList(torch.utils.data.Dataset):
    def __init__(self, item_list, transform=None):
        self.item_list = item_list
        self.transform = transform

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        path, target = self.item_list[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def prepare_test_data(args, use_transforms=True, semi=False):
    if args.corruption == 'original' or args.corruption == 'rendition' or args.corruption == 'v2' or args.corruption == 'sketch' or args.corruption == 'imagenet_a':
        te_transforms_local = te_transforms if use_transforms else None
    elif args.corruption in common_corruptions:
        te_transforms_local = te_transforms_imageC if use_transforms else None
    else:
        assert False, NotImplementedError
    if use_transforms:
        te_transforms_local = arch_normalized(args, eval_transform_for(args, te_transforms_local))
    
    use_imagenet_5k = bool(getattr(args, 'imagenet_5k_eval', False))

    if use_imagenet_5k:
        class_map_path = args.imagenet_class_map_file
        indices_file_path = args.imagenet_5k_indices_file

        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"ImageNet class map file not found at: {class_map_path}")
        with open(class_map_path, 'r') as f:
            class_to_idx = json.load(f)

        if not os.path.exists(indices_file_path):
            raise FileNotFoundError(f"ImageNet 5k indices file not found at: {indices_file_path}")
        with open(indices_file_path, 'r') as f:
            fnames_relative = f.readlines()

        if args.corruption == 'original':
            base_dir_for_current_corruption = os.path.join(args.data, 'val')
        elif args.corruption in common_corruptions:
            base_dir_for_current_corruption = os.path.join(args.data_corruption, args.corruption, str(args.level))
        elif args.corruption == 'rendition':
            teset = datasets.ImageFolder(args.data_rendition, te_transforms_local)
            teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=args.if_shuffle,
                                                    num_workers=args.workers, pin_memory=True)
            return teset, teloader
        elif args.corruption == 'sketch':
            teset = datasets.ImageFolder(args.data_sketch, te_transforms_local)
            teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=args.if_shuffle,
                                                    num_workers=args.workers, pin_memory=True)
            return teset, teloader
        elif args.corruption == 'v2':
            teset = ImageNetV2Dataset(transform=te_transforms_local, location=args.data_v2)
            teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=args.if_shuffle,
                                                    num_workers=args.workers, pin_memory=True)
            return teset, teloader
        else:
            raise Exception(f"Unknown corruption type '{args.corruption}' with 5k subset mode.")

        item_list_for_current_corruption = []
        for fn_rel in fnames_relative:
            fn_rel = fn_rel.strip()
            if not fn_rel:
                continue
            synset_id = fn_rel.split(os.sep)[0]
            if synset_id not in class_to_idx:
                print(f"Warning: Synset ID '{synset_id}' not found in class map for path '{fn_rel}'. Skipping.")
                continue
            label = class_to_idx[synset_id]
            full_image_path = os.path.join(base_dir_for_current_corruption, fn_rel)
            if not os.path.exists(full_image_path):
                print(f"Warning: Image file not found at: {full_image_path}. Skipping sample.")
                continue
            item_list_for_current_corruption.append((full_image_path, label))

        # Non-IID stream order: sort by GT class label so each batch is class-correlated.
        # Labels are used ONLY for ordering, not for adaptation.
        stream_order = getattr(args, 'stream_order', 'default')
        if stream_order == 'class_sorted':
            item_list_for_current_corruption.sort(key=lambda x: x[1])
        elif stream_order == 'imbalanced':
            item_list_for_current_corruption = dirichlet_label_shift_order(
                item_list_for_current_corruption,
                delta=float(getattr(args, 'noniid_dirichlet_delta', 0.1)),
                seed=int(getattr(args, 'noniid_seed', 0)))
        teset = ImageNetSubsetFromList(item_list_for_current_corruption, transform=te_transforms_local)
        print(f"Loaded {len(teset)} samples for 5k evaluation of '{args.corruption}' (level {args.level}) using ImageNet subset.")
        if stream_order in ('class_sorted', 'imbalanced'):
            shuffle_dataloader = False
        else:
            shuffle_dataloader = args.if_shuffle
    elif args.corruption == 'original':
        # source_dir_split is set only by main.obtain_train_loader (--source_split train):
        # source statistics for FOA/FOZO/ZOA then come from the train split instead of val.
        validdir = os.path.join(args.data, 'train' if getattr(args, 'source_dir_split', 'val') == 'train' else 'val')
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
        teset.switch_mode(True, False)
        shuffle_dataloader = args.if_shuffle
    elif args.corruption in common_corruptions:
        if not semi:
            print('Test on %s level %d' %(args.corruption, args.level))
        validdir = os.path.join(args.data_corruption, args.corruption, str(args.level))
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
        teset.switch_mode(True, False)
        shuffle_dataloader = args.if_shuffle
        if getattr(args, 'stream_order', 'default') == 'sar':
            labels = [s[1] for s in teset.original_samples]
            order = sar_label_shift_order(labels,
                                          imbalance_ratio=float(getattr(args, 'sar_imbalance_ratio', 500000)),
                                          T=int(getattr(args, 'sar_T', 100000)),
                                          seed=int(getattr(args, 'noniid_seed', 2023)))
            teset.set_specific_subset(order)
            print(f"SAR online label-shift order: ir={getattr(args,'sar_imbalance_ratio',500000)}, "
                  f"T={len(teset)} (full ImageNet-C, {len(set(labels))} classes)")
            shuffle_dataloader = False
    elif args.corruption == 'rendition':
        validdir = args.data_rendition
        teset = datasets.ImageFolder(validdir, te_transforms_local)
        shuffle_dataloader = args.if_shuffle
    elif args.corruption == 'sketch':
        validdir = args.data_sketch
        teset = datasets.ImageFolder(validdir, te_transforms_local)
        shuffle_dataloader = args.if_shuffle
    elif args.corruption == 'v2':
         teset = ImageNetV2Dataset(transform=te_transforms_local, location=args.data_v2)
         shuffle_dataloader = args.if_shuffle
    elif args.corruption == 'imagenet_a':
        validdir = args.data_a
        teset = datasets.ImageFolder(validdir, te_transforms_local)
        shuffle_dataloader = args.if_shuffle
    else:
        raise Exception('Corruption not found!')
    
    if not hasattr(args, 'workers'):
        args.workers = 1

    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=shuffle_dataloader, 
                                                    num_workers=args.workers, pin_memory=True)
    return teset, teloader
