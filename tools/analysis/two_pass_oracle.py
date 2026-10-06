"""Two-pass oracle for the local branch (tab:pooling, "Converged over an extra pass").

The deployed local branch estimates its target moments online, so the first batches of a
corruption are corrected against an estimate that has not converged yet. This tool removes
that transient: pass 1 runs the deployed configuration and saves the moments each corruption
converged to, and pass 2 re-runs the same stream with those moments as a fixed target. The
gap between the two is how much the online estimator costs.

Run it exactly like main.py, with one extra flag in front:

  pass 1   python tools/analysis/two_pass_oracle.py --dump-dir DIR   <main.py args...>
  pass 2   python tools/analysis/two_pass_oracle.py --oracle-dir DIR <main.py args...>

Pass 1 writes DIR/<corruption>.pt holding {module_name: (mu, sigma)} for the local
interfaces, saved at the episodic reset that ends each corruption. Pass 2 reads
DIR/<corruption>.pt and swaps the target as the stream moves to the next corruption, which
is why it cannot be expressed with main.py's --quar_local_oracle_target_file: that flag
takes a single file for the whole run.

Both passes patch tta_library.quar.local at import time and leave the repository code
untouched. The patches are diagnostic only; nothing here runs in the deployed configuration.
"""
import argparse
import os
import runpy
import sys

import torch

QUAR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, QUAR)

from tta_library.quar import local as L  # noqa: E402  (needs QUAR on sys.path)


def _install_dump(dump_dir):
    """Pass 1: save the converged running moments of a corruption at its episodic reset."""
    os.makedirs(dump_dir, exist_ok=True)
    original_reset = L.QuARLocalMixin.reset

    def dump_then_reset(self):
        if getattr(self, "_quar_enable", False) and self._quar_stream_m:
            moments = {}
            for name, mean in self._quar_stream_m.items():
                second = self._quar_stream_q[name]
                sigma = (second.float() - mean.float() ** 2).clamp(min=self._quar_eps).sqrt()
                moments[name] = (mean.float().cpu().clone(), sigma.cpu().clone())
            path = os.path.join(dump_dir, f"{self.args.corruption}.pt")
            torch.save(moments, path)
            self.model.logger.info(
                f"[two-pass oracle] saved converged moments of {len(moments)} interfaces to {path}"
            )
        original_reset(self)

    L.QuARLocalMixin.reset = dump_then_reset


def _install_oracle(oracle_dir):
    """Pass 2: serve the dumped moments of the current corruption as a fixed target."""

    def get_target(self):
        override = self.__dict__.get("_two_pass_override")
        if override is not None:
            return override
        corruption = self.args.corruption
        cache = self.__dict__.setdefault("_two_pass_cache", {})
        if corruption not in cache:
            path = os.path.join(oracle_dir, f"{corruption}.pt")
            cache[corruption] = torch.load(path, map_location="cpu")
            self.model.logger.info(
                f"[two-pass oracle] fixed target moments for {len(cache[corruption])} interfaces from {path}"
            )
        return cache[corruption]

    def set_target(self, value):
        self.__dict__["_two_pass_override"] = value

    # main.py assigns _quar_oracle_tgt when --quar_local_oracle_target_file is given; the
    # property keeps that assignment working while serving a per-corruption file otherwise.
    L.QuARLocalMixin._quar_oracle_tgt = property(get_target, set_target)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dump-dir", default="", help="pass 1: where to save the converged moments")
    parser.add_argument("--oracle-dir", default="", help="pass 2: where to read them back from")
    known, rest = parser.parse_known_args()
    if not known.dump_dir and not known.oracle_dir:
        parser.error("give --dump-dir (pass 1) or --oracle-dir (pass 2)")
    if known.dump_dir and known.oracle_dir:
        parser.error("--dump-dir and --oracle-dir are separate passes; give one")
    if known.dump_dir:
        _install_dump(known.dump_dir)
    else:
        _install_oracle(known.oracle_dir)
    os.chdir(QUAR)
    sys.argv = ["main.py"] + rest
    runpy.run_path(os.path.join(QUAR, "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
