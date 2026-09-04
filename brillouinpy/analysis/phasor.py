# -*- coding: utf-8 -*-
"""
Spectral phasor analysis (SPA) for Brillouin microspectroscopy.

A model-free alternative to the peak fitting in
:mod:`~brillouinpy.analysis.fitmodel`. Each spectrum is projected onto the first
(or a higher) harmonic of its discrete Fourier transform, giving a single complex
number - the *phasor* ``(G, S)``. Spectra with different Brillouin shifts / line
shapes land at different positions in the phasor plane, so segmentation and
contrast can be generated without assuming a line shape and without a non-linear
fit per pixel.

The approach follows

    Elsayad, K. (2019). "Spectral Phasor Analysis for Brillouin
    Microspectroscopy." Frontiers in Physics 7:62.
    https://doi.org/10.3389/fphy.2019.00062

For a window of ``N`` spectral channels with intensities ``I_k`` the harmonic
``n`` phasor is

    G = (Σ_k I_k cos(2π n k / N)) / Σ_k I_k
    S = (Σ_k I_k sin(2π n k / N)) / Σ_k I_k

i.e. the ``n``-th bin of ``numpy.fft.rfft`` normalised by the DC term. The
modulus ``sqrt(G² + S²)`` reflects how peaked the spectrum is (a single sharp
line sits near the unit circle, a broad/damped one moves toward the origin) and
the phase ``atan2(S, G)`` maps monotonically to the peak position within the
window.
"""
from dataclasses import dataclass

import numpy as np

from .. import core  # noqa: F401  (kept for symmetry with the other analysis modules)


@dataclass
class PhasorResult:
    """
    Result of :func:`phasor`.

    Attributes
    ----------
    G, S : numpy.ndarray
        Real and imaginary phasor coordinates, folded back to the spatial shape
        of the input object (a scalar-shaped ``(1,)`` array for a single
        :class:`~brillouinpy.Spectrum`). ``NaN`` where the spectrum was fully
        masked or had non-positive total intensity.
    harmonic : int
        The DFT harmonic the phasor was computed at.
    spectral_axis : numpy.ndarray
        The (possibly cropped) spectral axis the transform ran on.
    """

    G: np.ndarray
    S: np.ndarray
    harmonic: int
    spectral_axis: np.ndarray

    @property
    def modulus(self) -> np.ndarray:
        """Phasor modulus ``sqrt(G² + S²)`` (per pixel)."""
        return np.sqrt(self.G ** 2 + self.S ** 2)

    @property
    def phase(self) -> np.ndarray:
        """Phasor phase ``atan2(S, G)`` in radians, in ``[0, 2π)`` (per pixel)."""
        return np.mod(np.arctan2(self.S, self.G), 2 * np.pi)

    def features(self) -> np.ndarray:
        """
        ``(n_pixels, 2)`` stack of ``(G, S)`` for feeding into a clustering step,
        e.g. ``sklearn.cluster.KMeans().fit_predict(result.features())`` to
        segment the map directly in the phasor plane.
        """
        return np.column_stack([np.asarray(self.G).reshape(-1),
                                np.asarray(self.S).reshape(-1)])


def phasor(spectral_object, *, harmonic=1, axis_range=None, background=None) -> PhasorResult:
    """
    Compute the spectral phasor of every spectrum in ``spectral_object``.

    Parameters
    ----------
    spectral_object : core.SpectralObject
        The data to transform. Any dimensionality; the phasor is computed
        independently per pixel.
    harmonic : int, optional
        Which DFT harmonic to project onto. ``1`` (default) is the most robust;
        higher harmonics add sensitivity to finer spectral structure at the cost
        of noise. :func:`phase_to_shift` only supports ``harmonic=1``.
    axis_range : tuple[float, float], optional
        ``(low, high)`` bounds (in the units of ``spectral_axis``, i.e. GHz) to
        restrict the transform to - e.g. to a single Stokes peak. The phase then
        maps onto this window. Default: the full axis.
    background : {'min'} or float, optional
        Background handling before the transform, since a constant offset pulls
        the phasor toward the origin. ``'min'`` subtracts each spectrum's own
        minimum; a float subtracts that constant; ``None`` (default) leaves the
        data untouched.

    Returns
    -------
    PhasorResult

    Examples
    --------
    .. code::

        import brillouinpy as bp

        ph = bp.analysis.phasor.phasor(image, axis_range=(4, 12), background='min')
        bp.plot.phasor(ph)

        # fit-free Brillouin-shift map
        shift_map = bp.analysis.phasor.phase_to_shift(ph)

        # or segment directly in the phasor plane
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=3, n_init=10).fit_predict(ph.features())
        labels = labels.reshape(image.shape)
    """
    if harmonic < 1:
        raise ValueError(f"harmonic must be a positive integer, got {harmonic}.")

    flat = spectral_object.flat.spectral_data
    data = np.ma.filled(np.ma.asarray(flat), np.nan).astype(float)  # (n_pixels, n_channels)
    axis = np.asarray(spectral_object.spectral_axis, dtype=float)

    if axis_range is not None:
        low, high = axis_range
        selected = (axis >= low) & (axis <= high)
        data = data[:, selected]
        axis = axis[selected]

    # Drop channels that are NaN/masked in *any* (non-empty) pixel so the DFT
    # basis (which depends on the channel count N) is identical across the map -
    # same rationale as brillouinpy.analysis.Step.AnalysisStep. Fully
    # masked/empty pixels are ignored here and get a NaN phasor below.
    empty_pixels = np.isnan(data).all(axis=1)
    valid_channels = ~np.isnan(data[~empty_pixels]).any(axis=0) if (~empty_pixels).any() \
        else np.zeros(data.shape[1], dtype=bool)
    data = data[:, valid_channels]
    axis = axis[valid_channels]

    min_channels = 2 * harmonic + 1
    if data.shape[1] < min_channels:
        raise ValueError(
            f"Need at least {min_channels} valid spectral channels for harmonic {harmonic}, "
            f"got {data.shape[1]} after cropping/masking.")

    if background == 'min':
        data = data - np.nanmin(data, axis=1, keepdims=True)
    elif background is not None:
        data = data - float(background)

    totals = data.sum(axis=1)
    ft = np.fft.rfft(data, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        G = ft[:, harmonic].real / totals
        S = -ft[:, harmonic].imag / totals

    invalid = ~np.isfinite(totals) | (totals <= 0)
    G[invalid] = np.nan
    S[invalid] = np.nan

    shape = spectral_object.shape
    return PhasorResult(G=G.reshape(shape), S=S.reshape(shape),
                        harmonic=harmonic, spectral_axis=axis)


def phase_to_shift(result: PhasorResult) -> np.ndarray:
    """
    Convert a first-harmonic :class:`PhasorResult`'s phase into an approximate
    Brillouin-shift map, by linearly mapping the phase ``[0, 2π)`` onto the
    spectral window the transform ran on.

    This is a cheap, fit-free shift estimate - exact for an infinitely sharp
    single line, biased for broad or multi-peak spectra. For quantitative work
    still calibrate against a reference or fall back to
    :class:`~brillouinpy.analysis.fitmodel.DHO`.

    Parameters
    ----------
    result : PhasorResult
        Must have ``harmonic == 1``.

    Returns
    -------
    numpy.ndarray
        Estimated shift per pixel, in the units of ``result.spectral_axis``.
    """
    if result.harmonic != 1:
        raise ValueError("phase_to_shift only supports harmonic=1 (phase wraps for higher harmonics).")

    low, high = result.spectral_axis[0], result.spectral_axis[-1]
    return low + (result.phase / (2 * np.pi)) * (high - low)


def phasor_cursor(result: PhasorResult, center, radius) -> np.ndarray:
    """
    Boolean mask of the pixels whose phasor lies within ``radius`` of ``center``
    ``(G, S)`` in the phasor plane - the "cursor" selection from phasor FLIM,
    used here to pull a region of interest (a material, a cell compartment) out
    of a Brillouin map.

    Parameters
    ----------
    result : PhasorResult
    center : tuple[float, float]
        ``(G, S)`` coordinates of the cursor centre.
    radius : float
        Cursor radius in the phasor plane.

    Returns
    -------
    numpy.ndarray of bool
        Same spatial shape as ``result.G``. ``NaN`` phasors are ``False``.
    """
    cg, cs = center
    distance = np.sqrt((result.G - cg) ** 2 + (result.S - cs) ** 2)
    mask = distance <= radius
    return np.asarray(np.where(np.isnan(distance), False, mask))
