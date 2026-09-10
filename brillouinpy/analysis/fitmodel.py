"""Deprecated - moved to :mod:`brillouinpy.analysis.fit.core`.

``brillouinpy.analysis.fitmodel`` is kept as a thin re-export so existing
``bp.analysis.fitmodel.DHO`` / ``from brillouinpy.analysis.fitmodel import ...``
code keeps working; it will be removed in a future release.
"""
import warnings as _warnings

from .fit.core import *  # noqa: F401,F403
from .fit.core import (  # noqa: F401  (names not in __all__ but historically imported)
    _ConvolvedModel,
    _DHO_1,
    _DHO_2,
    _DHO_3,
    _Lorentzian_1,
    _Lorentzian_2,
    _Lorentzian_3,
    _dho_line,
    _estimate_p0_from_mean,
    _fit_concurrent,
    _lorentz_line,
    _model_funcs,
)

_warnings.warn(
    "brillouinpy.analysis.fitmodel has moved to brillouinpy.analysis.fit.core "
    "(or the flat brillouinpy.analysis.fit namespace); update your imports.",
    DeprecationWarning, stacklevel=2)
