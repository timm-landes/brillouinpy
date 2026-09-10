"""Deprecated - moved to :mod:`brillouinpy.analysis.fit.segmented`.

Kept as a thin re-export; will be removed in a future release.
"""
import warnings as _warnings

from .fit.segmented import *  # noqa: F401,F403
from .fit.segmented import SegmentedFitResult, segmented_fit  # noqa: F401

_warnings.warn(
    "brillouinpy.analysis.segmented has moved to brillouinpy.analysis.fit.segmented "
    "(segmented_fit is also re-exported from brillouinpy.analysis and "
    "brillouinpy.analysis.fit); update your imports.",
    DeprecationWarning, stacklevel=2)
