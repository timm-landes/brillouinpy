# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:14:03 2025

@author: Timm
"""
import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os

from .FitStep import FitStep

    
class DHO(FitStep):
    """
    Principal component analysis (PCA).


    Parameters
    ----------
    n_components : int, float or 'mle'
        The number of components.
    **kwargs :
        See original `implementation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_ for additional parameters.


    .. note :: Implementation and documentation based on`scikit-learn <https://scikit-learn.org>`_.
    """
    def __init__(self, *, expected_peaks, p0, bounds, padding):
        super().__init__(_fit_concurrent, expected_peaks=expected_peaks, p0=p0, bounds=bounds, padding=padding)


def _fitDHO(x, y, X, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs):
    Y = intensity_data_slice
    Y[Y == 0] = np.nan
    if p0 is None:
        p0 = [1] * (expected_peaks * 3 + 2)  # Default initial guesses
    if bounds is None:
        bounds = (0, np.inf)  # No negative values allowed

    try:
        popt, pcov = curve_fit(fit_funcs[expected_peaks], X, Y, p0=p0, bounds=bounds, nan_policy='omit')
        return x, y, popt, pcov
    except RuntimeError:
        print(f"Fit not successful for pixel ({x},{y})")
        return x, y, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)


def _fit_concurrent(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None, padding=100):
    variables = int(expected_peaks * 3 + 2)
    fit_results = np.zeros((intensity_data.shape[0], intensity_data.shape[1], variables))
    cov_results = np.empty((intensity_data.shape[0], intensity_data.shape[1], variables, variables))

    fit_funcs = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
    X = spectral_axis[0]
    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for x in range(intensity_data.shape[0]):
            for y in range(intensity_data.shape[1]):
                intensity_data_slice = intensity_data[x, y, :]
                task = executor.submit(_fitDHO, x, y, X, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs)
                tasks.append(task)
        
        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Fitting Spectral data'):
            x, y, popt, pcov = future.result()
            fit_results[x, y, :] = popt
            cov_results[x, y, :, :] = pcov

    return fit_results, cov_results


def _data_padding(data, padding):
    freq_step = np.abs(data[0, -1] - data[0, -2])
    pad_freq = np.arange(start=data[0, -1] + freq_step, 
                         stop=data[0, -1] + freq_step * (padding + 1), 
                         step=freq_step, dtype=type(freq_step))
    average = int(np.average(data[1, 0:20]))
    pad_data = [average for _ in range(padding)]
    pad_array = np.vstack((pad_freq, pad_data))
    
    lower_pad = np.flip(pad_array, axis=1)
    lower_pad[0, :] = -lower_pad[0, :]
    return np.hstack((lower_pad, data, pad_array))


def _DHO_1(x, I0, freqShift, LineWidth, Background, Asymmetry):
    return I0 * 4 * LineWidth * freqShift**2 /(np.pi*(((x-Asymmetry)**2 - freqShift**2)**2 + 4*(LineWidth*(x-Asymmetry))**2)) + Background


def _DHO_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry):
    return _DHO_1(x, I0, freqShift, LineWidth, Background, Asymmetry) + _DHO_1(x, I02, freqShift2, LineWidth2, Background, Asymmetry)


def _DHO_3(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, I03, freqShift3, LineWidth3, Background, Asymmetry):
    return _DHO_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry) + _DHO_1(x, I03, freqShift3, LineWidth3, Background, Asymmetry)