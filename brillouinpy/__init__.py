# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:31:47 2025

@author: Timm
"""

# from .spectralcontainer import SpectralImageContainer
# from .preprocessing import SpectralImageProcessor
# from .analyzer import SpectralAnalyzer
from .core import Spectrum, SpectralImage, SpectralVolume, SpectralContainer
from . import utils
from . import preprocessing
from . import analysis
from . import io
__all__ = [
    "utils",
    "Spectrum",
    "SpectralImage",
    "SpectralVolume",
    "SpectralContainer",
    "plot",
    "preprocessing",
    "analysis",
    "io",
    "export",
    # 'SpectralImageContainer',
    # 'SpectralImageProcessor',
    # 'SpectralAnalyzer',
]


def __getattr__(name):
    # Lazily import 'plot' (pulls in matplotlib) so that e.g. multiprocessing
    # fit workers, which only need 'analysis.fitmodel', don't pay for it.
    if name == "plot":
        import importlib
        module = importlib.import_module(".plot", __name__)
        globals()["plot"] = module
        return module
    # Lazily import the deprecated 'export' shim (brillouinpy.io.export moved
    # here on 2026-08-27) so that plain 'import brillouinpy' doesn't itself
    # trigger the module's DeprecationWarning - only actually accessing
    # 'brillouinpy.export' does.
    if name == "export":
        import importlib
        module = importlib.import_module(".export", __name__)
        globals()["export"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
