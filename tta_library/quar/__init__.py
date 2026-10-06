"""QuAR: Quantizer-Aligned Recalibration.

Two branches, and they used to live apart under names that did not say so. The local branch was a
mixin inside the ZOA baseline's file; the global branch was called cls_shift, after the FOA-style
CLS shifting it started from.

    local.py       per-channel recalibration at 16 quantizer inputs, blocks 8-11
    global_cls.py  recentering at the final head input
"""

from tta_library.quar.local import (  # noqa: F401
    QuARLocalMixin,
    NoAdaptQACWrapper, build_no_adapt_qac_wrapper, _NoAdaptBase, _is_quant_act, _block_idx,
)
from tta_library.quar.global_cls import build_cls_shift_wrapper  # noqa: F401
