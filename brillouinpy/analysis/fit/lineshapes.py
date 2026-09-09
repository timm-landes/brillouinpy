# -*- coding: utf-8 -*-
"""
Lineshape models for the pixel-wise Brillouin fits in
:mod:`brillouinpy.analysis.fit.core`.

Every model here is a *doublet* built from a single-mode core
``f(xs, I0, freqShift, LineWidth)`` (evaluated on the axis-shifted coordinate
``xs = x - axis_shift``) plus a shared baseline. :class:`~brillouinpy.analysis.fit.core.DHO`,
:class:`~brillouinpy.analysis.fit.core.Lorentzian`,
:class:`~brillouinpy.analysis.fit.core.Gaussian` and
:class:`~brillouinpy.analysis.fit.core.PeakFit` pick a family by name.

Adding your own lineshape
-------------------------
Write a single-mode core and register it - no need to touch the fitters::

    import numpy as np
    from brillouinpy.analysis import fit
    from brillouinpy.analysis.fit import lineshapes, PeakFit

    def _pseudo_voigt(xs, I0, freqShift, LineWidth, eta=0.5):
        g = lineshapes.LINESHAPES["gaussian"].core(xs, I0, freqShift, LineWidth)
        l = lineshapes.LINESHAPES["lorentzian"].core(xs, I0, freqShift, LineWidth)
        return eta * l + (1.0 - eta) * g

    lineshapes.register_lineshape("pseudo_voigt", core=_pseudo_voigt)

    fit = PeakFit(model="pseudo_voigt", expected_peaks=1, p0=p0, bounds=bounds)
    params, cov = fit.apply(image)

The n-peak model function ``curve_fit`` sees is assembled automatically:
``f(x, *[I0, freqShift, LineWidth] * n_peaks, *tail_params)``.


Parameter conventions
---------------------
Shared by every model function below and by the parameter vectors
:class:`~brillouinpy.analysis.fit.core.DHO` / ``Lorentzian`` / ``Gaussian`` return:

* ``I0``         - peak-area-like amplitude. The peak *height* above the
                   background is ``I0 / (pi * LineWidth)`` for every family.
* ``freqShift``  - Brillouin shift nu_B (GHz). One mode is a *doublet*: peaks
                   at ``axis_shift +/- freqShift``. ``expected_peaks`` counts
                   modes, so ``expected_peaks=1`` fits one Stokes/anti-Stokes
                   pair from a single ``freqShift``.
* ``LineWidth``  - **half width at half maximum** (HWHM), i.e. Gamma / 2.
                   The full width at half maximum is ``2 * LineWidth``.
                   ``mechanics`` converts this to the FWHM internally where a
                   physical linewidth is needed (loss tangent, viscosity).
* ``Background`` - additive constant, counted **once** for the whole model
                   (not once per peak).
* ``axis_shift`` - rigid shift of the frequency axis (GHz); the doublet is
                   symmetric about this value. Named ``Asymmetry`` up to 0.3.1.
* ``elastic_slope`` - *elastic* families only: slope of a linear background
                   ``Background + elastic_slope * (x - axis_shift)`` that
                   absorbs the residual wing of the (un-blanked) elastic peak.

Flat parameter order: ``[I0, freqShift, LineWidth] * n_peaks`` then the shared
``tail_params`` (``[Background, axis_shift]``, or ``[Background, axis_shift,
elastic_slope]`` for the ``*_elastic`` families).
"""
from typing import Callable, Dict, NamedTuple, Tuple

import numpy as np

#: Largest number of Brillouin modes the fixed-count lineshape models cover.
MAX_EXPECTED_PEAKS = 3

_FWHM_GAUSS_TO_SIGMA = 1.0 / np.sqrt(2.0 * np.log(2.0))  # HWHM -> sigma


# --------------------------------------------------------------------------- #
# Single-mode cores (no background), on xs = x - axis_shift
# --------------------------------------------------------------------------- #

def _dho_line(xs, I0, freqShift, LineWidth):
    """Bare single-mode damped-harmonic-oscillator doublet (no background) on the
    already axis-shifted coordinate ``xs = x - axis_shift``. ``LineWidth`` is the
    HWHM; the peak height is ``I0 / (pi * LineWidth)``."""
    return I0 * 4 * LineWidth * freqShift ** 2 / (
        np.pi * ((xs ** 2 - freqShift ** 2) ** 2 + 4 * (LineWidth * xs) ** 2)
    )


def _lorentz_line(xs, I0, freqShift, LineWidth):
    """Bare single-mode Lorentzian doublet (no background) on ``xs = x -
    axis_shift``: two area-normalised Lorentzians of HWHM ``LineWidth`` centred at
    ``+/- freqShift``, scaled by ``I0`` (so the peak height is ``I0 / (pi *
    LineWidth)``, matching :func:`_dho_line`)."""
    hw = LineWidth
    return (I0 / np.pi) * (hw / ((xs - freqShift) ** 2 + hw ** 2)
                           + hw / ((xs + freqShift) ** 2 + hw ** 2))


def _gauss_line(xs, I0, freqShift, LineWidth):
    """Bare single-mode Gaussian doublet (no background) on ``xs = x -
    axis_shift``: two Gaussians of HWHM ``LineWidth`` centred at ``+/- freqShift``.
    The peak height is ``I0 / (pi * LineWidth)``, matching :func:`_dho_line` /
    :func:`_lorentz_line` so ``I0`` / :func:`estimate_p0` carry over."""
    sigma = LineWidth * _FWHM_GAUSS_TO_SIGMA
    height = I0 / (np.pi * LineWidth)
    return height * (np.exp(-0.5 * ((xs - freqShift) / sigma) ** 2)
                     + np.exp(-0.5 * ((xs + freqShift) / sigma) ** 2))


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def _constant_baseline(x, Background, axis_shift):
    """Default shared baseline: a single additive constant."""
    return Background


def _elastic_baseline(x, Background, axis_shift, elastic_slope):
    """Linear baseline ``Background + elastic_slope * (x - axis_shift)`` - soaks
    up the sloping residual wing of an un-blanked elastic / Rayleigh peak
    (cf. the ``*_elastic`` models in HDF5_BLS_treat)."""
    return Background + elastic_slope * (np.asarray(x, dtype=float) - axis_shift)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class Lineshape(NamedTuple):
    """One registered lineshape family. ``core`` is the single-mode doublet,
    ``baseline`` the shared additive term, ``tail_params`` its parameter names
    (the first must be the additive background; the second must be
    ``axis_shift``)."""
    name: str
    core: Callable
    baseline: Callable
    tail_params: Tuple[str, ...]
    max_peaks: int


LINESHAPES: Dict[str, Lineshape] = {}
_assembled_cache: Dict[str, Dict[int, Callable]] = {}


def register_lineshape(name, *, core, baseline=_constant_baseline,
                       tail_params=("Background", "axis_shift"), max_peaks=MAX_EXPECTED_PEAKS):
    """
    Register a lineshape family so :class:`~brillouinpy.analysis.fit.core.PeakFit`
    (and :func:`model_funcs`) can use it by ``name``.

    Parameters
    ----------
    name : str
        Identifier passed as ``model=`` to ``PeakFit`` / ``segmented_fit``.
    core : callable
        ``core(xs, I0, freqShift, LineWidth) -> ndarray`` - the single-mode
        doublet without background, on ``xs = x - axis_shift``.
    baseline : callable, optional
        ``baseline(x, *tail_params) -> ndarray | float``; default a constant
        ``Background``.
    tail_params : tuple of str, optional
        Names of the shared trailing parameters. ``tail_params[0]`` must be the
        additive background and ``tail_params[1]`` must be ``axis_shift``.
    max_peaks : int, optional
        Largest mode count to assemble a model for (default 3).
    """
    if len(tail_params) < 2:
        raise ValueError("tail_params must start with (background, axis_shift, ...).")
    LINESHAPES[name] = Lineshape(name, core, baseline, tuple(tail_params), int(max_peaks))
    _assembled_cache.pop(name, None)


class _AssembledModel:
    """Picklable flat model callable
    ``f(x, *[I0, freqShift, LineWidth] * n_peaks, *spec.tail_params)`` built from a
    :class:`Lineshape`. A plain closure would not survive being sent to the
    ``ProcessPoolExecutor`` fit workers (cf. ``fit.core._ConvolvedModel``);
    this pickles by carrying ``spec`` (its ``core`` / ``baseline`` must therefore
    be importable module-level functions, not lambdas, to work multiprocess)."""

    def __init__(self, spec, n_peaks):
        self.spec = spec
        self.n_peaks = int(n_peaks)
        self.__name__ = f"_{spec.name}_{n_peaks}"
        self.__qualname__ = self.__name__
        self.__doc__ = (f"{n_peaks}-mode '{spec.name}' model. Parameters: "
                        f"[I0, freqShift, LineWidth] * {n_peaks} + {list(spec.tail_params)}.")

    def __call__(self, x, *params):
        n_peak_params = 3 * self.n_peaks
        n_tail = len(self.spec.tail_params)
        if len(params) != n_peak_params + n_tail:
            raise TypeError(
                f"{self.spec.name} {self.n_peaks}-mode model takes "
                f"{n_peak_params + n_tail} parameters, got {len(params)}.")
        tail = params[n_peak_params:]
        axis_shift = tail[1]
        xs = np.asarray(x, dtype=float) - axis_shift
        out = self.spec.baseline(x, *tail)
        for i in range(self.n_peaks):
            I0, freqShift, LineWidth = params[3 * i:3 * i + 3]
            out = out + self.spec.core(xs, I0, freqShift, LineWidth)
        return out


def assemble_model(spec, n_peaks):
    """Build the flat ``curve_fit`` callable
    ``f(x, *[I0, freqShift, LineWidth] * n_peaks, *spec.tail_params)`` for a
    :class:`Lineshape` (or a registered name)."""
    if isinstance(spec, str):
        spec = LINESHAPES[spec]
    return _AssembledModel(spec, n_peaks)


#: Hand-written concrete models kept for readability, back-compat and exact
#: identity with the historical ``fitmodel`` functions.
_HANDWRITTEN: Dict[str, Dict[int, Callable]] = {}


def model_funcs(name):
    """``{n_peaks: model_function}`` for a registered lineshape family. The
    ``dho`` / ``lorentzian`` families return the hand-written functions; anything
    else is assembled by :func:`assemble_model` (lazily, then cached)."""
    if name not in LINESHAPES:
        raise ValueError(
            f"unknown lineshape {name!r}; registered: {sorted(LINESHAPES)}.")
    if name in _HANDWRITTEN:
        return _HANDWRITTEN[name]
    if name not in _assembled_cache:
        spec = LINESHAPES[name]
        _assembled_cache[name] = {n: assemble_model(spec, n)
                                  for n in range(1, spec.max_peaks + 1)}
    return _assembled_cache[name]


# --------------------------------------------------------------------------- #
# Hand-written DHO / Lorentzian models (parameter order per the conventions above)
# --------------------------------------------------------------------------- #

def _DHO_1(x, I0, freqShift, LineWidth, Background, axis_shift):
    """One-mode DHO doublet. See the module conventions above: ``LineWidth`` is
    the HWHM and ``Background`` is a single additive constant."""
    return _dho_line(x - axis_shift, I0, freqShift, LineWidth) + Background


def _DHO_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, axis_shift):
    """Two-mode DHO model. Shared single ``Background`` and ``axis_shift``."""
    xs = x - axis_shift
    return (_dho_line(xs, I0, freqShift, LineWidth)
            + _dho_line(xs, I02, freqShift2, LineWidth2) + Background)


def _DHO_3(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2,
           I03, freqShift3, LineWidth3, Background, axis_shift):
    """Three-mode DHO model. Shared single ``Background`` and ``axis_shift``."""
    xs = x - axis_shift
    return (_dho_line(xs, I0, freqShift, LineWidth)
            + _dho_line(xs, I02, freqShift2, LineWidth2)
            + _dho_line(xs, I03, freqShift3, LineWidth3) + Background)


def _Lorentzian_1(x, I0, freqShift, LineWidth, Background, axis_shift):
    """One-mode Lorentzian doublet. Same conventions as :func:`_DHO_1`
    (``LineWidth`` is the HWHM, ``Background`` counted once)."""
    return _lorentz_line(x - axis_shift, I0, freqShift, LineWidth) + Background


def _Lorentzian_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, axis_shift):
    """Two-mode Lorentzian model. Shared single ``Background`` and ``axis_shift``."""
    xs = x - axis_shift
    return (_lorentz_line(xs, I0, freqShift, LineWidth)
            + _lorentz_line(xs, I02, freqShift2, LineWidth2) + Background)


def _Lorentzian_3(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2,
                  I03, freqShift3, LineWidth3, Background, axis_shift):
    """Three-mode Lorentzian model. Shared single ``Background`` and ``axis_shift``."""
    xs = x - axis_shift
    return (_lorentz_line(xs, I0, freqShift, LineWidth)
            + _lorentz_line(xs, I02, freqShift2, LineWidth2)
            + _lorentz_line(xs, I03, freqShift3, LineWidth3) + Background)


_HANDWRITTEN["dho"] = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
_HANDWRITTEN["lorentzian"] = {1: _Lorentzian_1, 2: _Lorentzian_2, 3: _Lorentzian_3}


# --------------------------------------------------------------------------- #
# Built-in registrations
# --------------------------------------------------------------------------- #

register_lineshape("dho", core=_dho_line)
register_lineshape("lorentzian", core=_lorentz_line)
register_lineshape("gaussian", core=_gauss_line)

for _name, _core in (("dho", _dho_line), ("lorentzian", _lorentz_line), ("gaussian", _gauss_line)):
    register_lineshape(f"{_name}_elastic", core=_core, baseline=_elastic_baseline,
                       tail_params=("Background", "axis_shift", "elastic_slope"))
del _name, _core
