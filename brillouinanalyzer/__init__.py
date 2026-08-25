# -*- coding: utf-8 -*-
"""
Deprecated compatibility shim: 'brillouinanalyzer' was renamed to 'brillouinpy'.

Existing code that does ``import brillouinanalyzer as bp`` keeps working
unchanged - this module re-exports everything from :mod:`brillouinpy`. Update
your imports to ``import brillouinpy as bp`` when convenient; this shim may be
removed in a future release.
"""
import warnings

warnings.warn(
    "The 'brillouinanalyzer' package has been renamed to 'brillouinpy'. "
    "Please update your imports to 'import brillouinpy as bp' (or "
    "'from brillouinpy import ...'). This compatibility shim may be removed "
    "in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from brillouinpy import *  # noqa: E402,F401,F403
from brillouinpy import __getattr__  # noqa: E402,F401
