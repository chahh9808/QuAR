#!/usr/bin/env python3
"""Remap ImageNet-C-bar float-severity dirs to integer levels 1-5 via directory symlinks.

Generator writes:  <gen>/ImageNet-C-Bar/<corruption>/<sev_float>/<class>/<img>
Loader expects:     <root>/<corruption>/<level:int>/<class>/<img>   (level in 1..5)

For each corruption we sort its 5 severities ascending and map -> levels 1..5, then
create a directory symlink  <root>/<corruption>/<N> -> <gen>/.../<sev_float>.
ImageFolder follows the symlink to the per-class image dirs. Idempotent.
"""
# Repository root (this tool lives one level below it).
import os
import os as _os, sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _REPO_ROOT)
import csv, os, sys

CSV = _os.path.join(_REPO_ROOT, "external_repos/augmentation-corruption/imagenet_c_bar/imagenet_c_bar.csv")
GEN = os.environ.get("DATA_CBAR", "") + "-gen/ImageNet-C-Bar"
ROOT = os.environ.get("DATA_CBAR", "")


def read_csv():
    sev = {}
    for line in open(CSV):
        v = line.strip().split(",")
        if len(v) >= 2:
            sev[v[0]] = [float(x) for x in v[1:]]
    return sev


def main():
    # We evaluate at LEVEL 5 (highest severity) only, so only level 5 must exist; levels 1-4
    # are optional (skipped with a note if their gen dirs are absent). Fatal only if the
    # highest-severity dir for any corruption is missing.
    sev = read_csv()
    os.makedirs(ROOT, exist_ok=True)
    missing_hi = []
    skipped = 0
    for corr, sevs in sev.items():
        order = sorted(sevs)  # ascending -> level 1..5
        cdir = os.path.join(ROOT, corr)
        os.makedirs(cdir, exist_ok=True)
        for level, s in enumerate(order, start=1):
            src = os.path.join(GEN, corr, "{:.2f}".format(s))
            if not os.path.isdir(src):
                if level == len(order):           # highest severity (== eval level 5) is required
                    missing_hi.append(src)
                else:
                    skipped += 1
                continue
            nclass = len([d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))])
            link = os.path.join(cdir, str(level))
            if os.path.islink(link) or os.path.exists(link):
                if os.path.realpath(link) != os.path.realpath(src):
                    os.remove(link); os.symlink(src, link)
            else:
                os.symlink(src, link)
            print(f"  {corr} L{level} (sev {s:g}) -> {nclass} class dirs")
    if missing_hi:
        print("\nFATAL: highest-severity (eval level 5) dir missing for:")
        for p in missing_hi:
            print(" ", p)
        sys.exit(1)
    note = f" ({skipped} lower-severity levels not generated, level-5-only run)" if skipped else ""
    print(f"\nRemapped {len(sev)} corruptions; level 5 present for all{note}. Root: {ROOT}")


if __name__ == "__main__":
    main()
