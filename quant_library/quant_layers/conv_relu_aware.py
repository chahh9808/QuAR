"""ReLU-aware conv activation quantizer for convolutional backbones.

Problem this fixes. PTQ4ViT's conv activation quantizer is symmetric and signed,

    x_sim = (x / a_interval).round().clamp(-a_qmax, a_qmax - 1)      # conv.py L64, L574

with ``a_qmax = 2 ** (a_bit - 1)``. That is the right choice for a ViT, whose only
conv is the patch embedding and whose linear-layer inputs are post-LayerNorm and
therefore signed. It is the wrong choice for a ResNet: every conv input except the
stem is post-ReLU and so non-negative, which leaves the entire negative half of the
code range unused and costs one effective bit. Measured on our W6A6 ResNet-50
source statistics, all 11 recalibrated interfaces had zero negative channels.

PTQ4ViT already solves the analogous problem for the one-sided post-GELU
activations with a twin uniform quantizer (PostGeluPTQSLBatchingQuantLinear); this
class applies the same idea to post-ReLU convs.

The fix. For a non-negative input, doubling ``a_qmax`` makes

    clamp(-2 * a_qmax, 2 * a_qmax - 1)

behave exactly as an unsigned ``[0, 2 * a_qmax - 1]`` quantizer, because x/interval
is never negative, so the extended negative half is unreachable. At a_bit=6 that
turns a usable range of [0, 31] into [0, 63], i.e. the full 6 bits. Nothing else
has to change: the interval initialiser and the interval search both derive from
a_qmax, so they rescale consistently. No upstream method is copied or edited.
"""

import torch

from .conv import ChannelwiseBatchingQuantConv2d


class ReLUAwareChannelwiseBatchingQuantConv2d(ChannelwiseBatchingQuantConv2d):
    """Uses the full code range on convs whose calibration input is non-negative."""

    a_unsigned = False

    def _initialize_intervals(self):
        # Runs after _initialize_calib_parameters() has populated raw_input and
        # before any interval is derived from a_qmax, which is the only window in
        # which a_qmax can still be changed consistently.
        try:
            is_nonneg = bool(self.raw_input.min().item() >= 0)
        except Exception:
            is_nonneg = False
        if is_nonneg:
            self.a_qmax = self.a_qmax * 2
            self.a_unsigned = True
        super()._initialize_intervals()


def summarise(net):
    """Report how many conv interfaces took the unsigned path."""
    tot = uns = 0
    for name, m in net.named_modules():
        if isinstance(m, ReLUAwareChannelwiseBatchingQuantConv2d):
            tot += 1
            uns += int(getattr(m, "a_unsigned", False))
    return tot, uns
