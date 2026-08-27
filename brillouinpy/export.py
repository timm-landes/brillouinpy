# -*- coding: utf-8 -*-
"""
Deprecated compatibility shim: this module's contents moved to
:mod:`brillouinpy.io.export` (2026-08-27), consolidated under
:mod:`brillouinpy.io` alongside the acquisition-side loaders.

``brillouinpy.export.X(...)`` keeps working unchanged for now, but emits a
DeprecationWarning - update your code to call ``brillouinpy.io.export.X(...)``
(or, for the most commonly used functions, the ``brillouinpy.io.X(...)``
top-level re-export) directly. See CHANGELOG.md for details.
"""
import warnings

warnings.warn(
    "The 'brillouinpy.export' module has moved to 'brillouinpy.io.export'. "
    "Please update your code to call 'brillouinpy.io.export.X(...)' (or "
    "'brillouinpy.io.X(...)' for the commonly used functions) directly - this "
    "compatibility shim may be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from .io.export import *  # noqa: E402,F401,F403
