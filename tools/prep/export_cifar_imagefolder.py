#!/usr/bin/env python
"""Export CIFAR-10/100 and CIFAR-10-C/100-C (severity 5) to an ImageNet-C-style ImageFolder layout.

Layout produced (class directories are zero-padded label indices so ImageFolder's
alphabetical class order equals the label order):

    <out_root>/<name>/train/<ccc>/<idx>.png          CIFAR train split (source data)
    <out_root>/<name>/val/<ccc>/<idx>.png            CIFAR test split (clean)
    <out_root>/<name>-c/<corruption>/5/<ccc>/<idx>.png   CIFAR-C severity 5 (rows 40000:50000)

Images are stored at their native 32x32; the loader resizes to 224 at load time.
Only the 15 standard corruptions are exported (the extra 4 in CIFAR-C are skipped).
"""
import argparse
import os
import pickle
from multiprocessing import Pool

import numpy as np
from PIL import Image

CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
               'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
               'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def _unpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def load_cifar(py_root, name):
    if name == 'cifar10':
        xs, ys = [], []
        for i in range(1, 6):
            d = _unpickle(os.path.join(py_root, f'data_batch_{i}'))
            xs.append(d['data']); ys += list(d['labels'])
        train_x = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_y = np.array(ys)
        d = _unpickle(os.path.join(py_root, 'test_batch'))
        test_x = d['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_y = np.array(d['labels'])
    else:
        d = _unpickle(os.path.join(py_root, 'train'))
        train_x = d['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_y = np.array(d['fine_labels'])
        d = _unpickle(os.path.join(py_root, 'test'))
        test_x = d['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_y = np.array(d['fine_labels'])
    return train_x, train_y, test_x, test_y


def _write_chunk(job):
    out_dir, imgs, labels, start = job
    n = 0
    for k in range(len(imgs)):
        cls_dir = os.path.join(out_dir, f'{int(labels[k]):03d}')
        p = os.path.join(cls_dir, f'{start + k:05d}.png')
        if os.path.exists(p):
            continue
        Image.fromarray(imgs[k]).save(p, compress_level=1)
        n += 1
    return n


def export_split(out_dir, imgs, labels, workers, chunk=500):
    for c in np.unique(labels):
        os.makedirs(os.path.join(out_dir, f'{int(c):03d}'), exist_ok=True)
    jobs = [(out_dir, imgs[i:i + chunk], labels[i:i + chunk], i) for i in range(0, len(imgs), chunk)]
    with Pool(workers) as pool:
        written = sum(pool.map(_write_chunk, jobs))
    print(f'[export] {out_dir}: {len(imgs)} images ({written} newly written)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', choices=['cifar10', 'cifar100'], required=True)
    ap.add_argument('--py_root', required=True, help='cifar-10-batches-py or cifar-100-python directory')
    ap.add_argument('--c_root', required=True, help='CIFAR-10-C or CIFAR-100-C directory with .npy files')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--skip_train', action='store_true')
    args = ap.parse_args()

    train_x, train_y, test_x, test_y = load_cifar(args.py_root, args.name)
    print(f'[load] train {train_x.shape} test {test_x.shape}', flush=True)
    clean_root = os.path.join(args.out_root, args.name)
    export_split(os.path.join(clean_root, 'val'), test_x, test_y, args.workers)

    labels_c = np.load(os.path.join(args.c_root, 'labels.npy'))
    sev5 = slice(40000, 50000)
    assert labels_c.shape[0] == 50000
    if not np.array_equal(labels_c[sev5], test_y):
        print('[warn] CIFAR-C severity-5 labels differ from the test split order; using labels.npy', flush=True)
    for corr in CORRUPTIONS:
        x = np.load(os.path.join(args.c_root, f'{corr}.npy'), mmap_mode='r')[sev5]
        export_split(os.path.join(args.out_root, f'{args.name}-c', corr, '5'), np.ascontiguousarray(x), labels_c[sev5], args.workers)

    if not args.skip_train:
        export_split(os.path.join(clean_root, 'train'), train_x, train_y, args.workers)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
