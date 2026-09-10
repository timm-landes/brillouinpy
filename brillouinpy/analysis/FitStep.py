"""Deprecated - moved to :mod:`brillouinpy.analysis.fit.step`.

Kept as a thin re-export; will be removed in a future release.
"""
import warnings as _warnings

from .fit.step import FitStep  # noqa: F401

_warnings.warn(
    "brillouinpy.analysis.FitStep has moved to brillouinpy.analysis.fit.step; "
    "update your imports.",
    DeprecationWarning, stacklevel=2)
