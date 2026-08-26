from numbers import Number
from typing import Tuple
import numpy as np
from skimage import restoration

from . import PreprocessingStep
from ..core import Spectrum


# BackgroundSubtractor and Cropper below are adapted, near-verbatim, from
# RamanSPy (https://github.com/barahona-research-group/RamanSPy), Copyright (c)
# 2023 Dimitar Georgiev, licensed under the BSD 3-Clause License (see LICENSE and
# THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt in the repository root).
# IRF_Remover and Deconvoluter_IRF further below are original to this package.
class BackgroundSubtractor(PreprocessingStep):
    """
    Subtract a fixed reference background.

    Parameters
    ----------
    background : Spectrum
        The reference background to subtract.
    """

    def __init__(self, *, background: Spectrum):
        super().__init__(_subtract_background, background=background)


class Cropper(PreprocessingStep):
    """
    Crop the intensity values and the shift axis associated with the band range(s) specified.

    Parameters
    ----------
    region : tuple of two elements
        The band intervals to crop (in GHz).

        For instance:

            - [(None, 300)] - keeps the bands < 300cm-1
            - [(3000, None)] - keeps the bands > 3000cm-1
            - [(700, 1800)] - keeps the bands between 700 and 1800 (i.e. the "fingerprint" region)
    """

    def __init__(self, *, region: Tuple[Number or None, Number or None]): # type: ignore
        if len(region) != 2:
            raise ValueError("The region must be a tuple of two elements")

        super().__init__(_crop, region=region)

class IRF_Remover(PreprocessingStep):
    """
    Remove the intensity values corresponding to the Instrument Response Function (IRF) including an optional little offset.

    Parameters
    ----------
    offset : number of spectral channels which gets removed
        The frequency channels to crop (in channels).

        For instance:
            - None - only the IRF gets detected, saved and removed from the spectral data
            - 10 - the IRF and additional 10 channels get removed
    """
    def __init__(self, *, offset: Number or None): # type: ignore

        super().__init__(_irf_remove, offset=offset)


class Deconvoluter_IRF(PreprocessingStep):
    """
    Deconvolve the Instrument Response Function (IRF) out of the intensity data using the
    Richardson-Lucy algorithm, then remove the (now deconvolved) IRF region from the spectrum.

    Unlike :class:`IRF_Remover`, which just crops the IRF away, this step first uses the
    detected IRF as the point-spread function to sharpen the rest of the spectrum via
    Richardson-Lucy deconvolution, and only then blanks out the IRF channels themselves
    (set to ``numpy.nan``) plus the requested offset around them. Downstream steps must be
    able to handle ``numpy.nan`` values (e.g. :class:`~brillouinpy.analysis.Step.AnalysisStep`
    drops NaN-containing channels automatically).

    Works on any spectral object (``Spectrum``, ``SpectralImage``, ``SpectralVolume``, or the
    raw ``(x, y, z, t, spectral)`` shape returned by
    :func:`brillouinpy.utils.prepare_brillouin_data`) - it loops over whatever spatial
    dimensions are present.

    Parameters
    ----------
    offset : number of spectral channels which gets removed
        The frequency channels to blank out around the detected IRF (in channels), in
        addition to the IRF's own extent.
    iterations : int or None
        The number of Richardson-Lucy deconvolution iterations to run.
    padding : int or None
        Number of channels to reflect-pad the spectrum with on each side before running
        the deconvolution, then crop back off afterwards. Richardson-Lucy deconvolution
        otherwise treats the array edges as a hard boundary, which can produce artefacts
        (a spurious up/down swing) near the first/last few channels; padding with a
        reflected copy of the spectrum gives the algorithm room to work without that
        boundary, at the cost of a bit of extra computation. ``None`` or ``0`` disables
        padding. A value comparable to a few times the IRF's width is a reasonable start;
        it must be smaller than the spectrum's length.
    """

    def __init__(self, *, offset: Number or None, iterations: Number or None, padding: Number or None): # type: ignore
        super().__init__(_deconvolute_irf, offset = offset, iterations = iterations, padding = padding)

def _subtract_background(original_intensity_data, original_spectral_axis, background: Spectrum):
    if not np.array_equal(background.spectral_axis, original_spectral_axis):
        raise ValueError("The spectral axis of the background must match that of the spectral object to process")

    return original_intensity_data[..., :] - background.spectral_data, original_spectral_axis


def _crop(intensity_data, spectral_axis, region):
    indices_to_leave = _get_indices_to_leave(spectral_axis, region)

    return intensity_data[..., indices_to_leave], spectral_axis[indices_to_leave]


def _get_indices_to_leave(spectral_axis, region):
    start = spectral_axis[0] if region[0] is None else region[0]
    end = spectral_axis[-1] if region[1] is None else region[1]

    if start > end:
        # swap
        start, end = end, start

    indices_to_leave = np.logical_and(
        start <= spectral_axis, spectral_axis <= end)

    return indices_to_leave

def _irf_remove(intensity_data, spectral_axis, offset):
    corrected_intensity_data = np.copy(intensity_data)
    spectral_response = np.zeros(intensity_data.shape)

    for x in range(corrected_intensity_data.shape[0]):
        for y in range(corrected_intensity_data.shape[1]):
            start, end = _find_instrumental_response(intensity_data[x, y, :])
            spectral_response[x, y, start:end] = intensity_data[x, y, start:end]
            corrected_intensity_data[x, y, start-offset:end+offset] = 0

    return corrected_intensity_data, spectral_axis

def _find_instrumental_response(pixel_spectrum):
    max_index = np.argmax(pixel_spectrum)

    start = next((i for i in range(max_index, 0, -1) if pixel_spectrum[i] == 0), 0)
    end = next((i for i in range(max_index, len(pixel_spectrum)) if pixel_spectrum[i] == 0), len(pixel_spectrum))

    return start, end

def _deconvolute_irf(intensity_data, spectral_axis, offset, iterations=4, padding=None):
    '''
    Deconvolute the intensity data with the Instrumental Response Function (IRF) using the Richardson-Lucy algorithm.
    Additionally, the IRF gets removed from the spectral data.
    Works on any spatial dimensionality (0D - a single Spectrum - through the 4D (x, y, z, t,
    spectral) shape returned by prepare_brillouin_data).
    '''
    corrected_intensity_data = np.copy(np.asarray(intensity_data))
    spectral_response = np.zeros(intensity_data.shape)
    spatial_shape = intensity_data.shape[:-1]
    n_channels = intensity_data.shape[-1]
    pad = padding or 0

    for idx in np.ndindex(spatial_shape):
        pixel_spectrum = intensity_data[idx + (slice(None),)]
        start, end = _find_instrumental_response(pixel_spectrum)
        spectral_response[idx + (slice(start, end),)] = pixel_spectrum[start:end]

        image = corrected_intensity_data[idx + (slice(None),)]
        psf = spectral_response[idx + (slice(None),)]
        if pad:
            # Richardson-Lucy treats the array edges as a hard boundary, which produces
            # a spurious up/down swing near the first/last few channels; padding with a
            # reflected copy gives it room to work before cropping back to the original
            # range. The PSF is zero-padded to match, so the IRF's position within the
            # padded array stays aligned with the (also padded) spectrum.
            image = np.pad(image, pad, mode='reflect')
            psf = np.pad(psf, pad, mode='constant')

        # Apply Richardson-Lucy deconvolution
        deconvolved = restoration.richardson_lucy(
            image, psf, num_iter=iterations, clip=False, filter_epsilon=None
        )
        corrected_intensity_data[idx + (slice(None),)] = deconvolved[pad:pad + n_channels] if pad else deconvolved
        # Remove the IRF and additional offset
        corrected_intensity_data[
            idx + (slice(max(0, start - offset), min(intensity_data.shape[-1], end + offset)),)
        ] = np.nan

    return corrected_intensity_data, spectral_axis
