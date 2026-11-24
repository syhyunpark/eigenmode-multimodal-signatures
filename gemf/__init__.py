"""Fixed-rank geometric eigenmode multiview factorization (GEMF)."""

from .data import MultiViewData, load_multiview_npz
from .fixed_rank_model import FixedRankGEMF, FixedRankGEMFConfig

__all__ = [
    "MultiViewData",
    "load_multiview_npz",
    "FixedRankGEMF",
    "FixedRankGEMFConfig",
]
