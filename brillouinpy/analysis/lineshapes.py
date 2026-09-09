"""Deprecated - moved to :mod:`brillouinpy.analysis.fit.lineshapes`.

Kept as a thin re-export; will be removed in a future release.
"""
import warnings as _warnings

from .fit.lineshapes import *  # noqa: F401,F403
from .fit.lineshapes import (  # noqa: F401
    LINESHAPES,
    Lineshape,
    _dho_line,
    _gauss_line,
    _lorentz_line,
    assemble_model,
    model_funcs,
    register_lineshape,
)

_warnings.warn(
    "brillouinpy.analysis.lineshapes has moved to brillouinpy.analysis.fit.lineshapes; "
    "update your imports.",
    DeprecationWarning, stacklevel=2)
