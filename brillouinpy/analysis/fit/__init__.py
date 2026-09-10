"""
Brillouin peak fitting.

Everything for going from a (pre-processed) spectral image to fitted Brillouin
parameters lives here:

* :mod:`~brillouinpy.analysis.fit.lineshapes` - the lineshape models and the
  registry for adding your own (:func:`register_lineshape`).
* :mod:`~brillouinpy.analysis.fit.core` - the pixel-wise fitters
  (:class:`DHO`, :class:`Lorentzian`, :class:`Gaussian`, :class:`PeakFit`) plus
  :func:`estimate_p0`, :func:`estimate_peak_count`, :func:`irf_kernel`.
* :mod:`~brillouinpy.analysis.fit.segmented` - :func:`segmented_fit`, the
  classify-then-fit strategy built on top of the fitters.
* :mod:`~brillouinpy.analysis.fit.step` - the :class:`FitStep` base class.

The names below are re-exported here for convenience, so ``bp.analysis.fit.DHO``,
``bp.analysis.fit.segmented_fit`` etc. all work.
"""
from . import core, lineshapes, segmented, step
from .core import (
    DHO,
    Gaussian,
    Lorentzian,
    MAX_EXPECTED_PEAKS,
    PeakFit,
    estimate_p0,
    estimate_peak_count,
    fitted_peak_count,
    irf_kernel,
    model_funcs,
)
from .lineshapes import LINESHAPES, Lineshape, assemble_model, register_lineshape
from .segmented import SegmentedFitResult, segmented_fit
from .step import FitStep

__all__ = [
    "core", "lineshapes", "segmented", "step",
    "DHO", "Lorentzian", "Gaussian", "PeakFit", "FitStep",
    "estimate_p0", "estimate_peak_count", "fitted_peak_count", "irf_kernel",
    "model_funcs", "MAX_EXPECTED_PEAKS",
    "LINESHAPES", "Lineshape", "assemble_model", "register_lineshape",
    "segmented_fit", "SegmentedFitResult",
]
