from numbers import Number
from typing import Tuple
import numpy as np
from skimage import restoration

from . import PreprocessingStep
from ..core import Spectrum


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

    def __init__(self, *, region: Tuple[Number or None, Number or None]):
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
    def __init__(self, *, offset: Number or None):

        super().__init__(_irf_remove, offset=offset)


class Deconvoluter_IRF(PreprocessingStep):
    
    
    def __init__(self, *, offset: Number or None, iterations: Number or None, padding: Number or None):
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

def _deconvolute_irf(intensity_data, spectral_axis, offset, iterations = 4, padding = 100):
    
    corrected_intensity_data = np.copy(intensity_data)
    spectral_response = np.zeros(intensity_data.shape)
    
    for x in range(corrected_intensity_data.shape[0]):
        for y in range(corrected_intensity_data.shape[1]):
            start, end = _find_instrumental_response(intensity_data[x, y, :])
            spectral_response[x, y, start:end] = intensity_data[x, y, start:end]
            corrected_intensity_data[x,y,:] = restoration.richardson_lucy(corrected_intensity_data[x,y,:], 
                                                                          spectral_response[x, y, :],
                                                                          num_iter=iterations, clip=False, 
                                                                          filter_epsilon=None)
            corrected_intensity_data[x, y, start-offset:end+offset] = 0
    
    return corrected_intensity_data, spectral_axis