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

from multiprocessing import shared_memory

from .FitStep import FitStep


# Worker-Globals
_DHO3_DATA = None
_DHO3_SPECTRAL_AXIS = None
_DHO3_EXPECTED_PEAKS = None
_DHO3_P0 = None
_DHO3_BOUNDS = None
_DHO3_FIT_FUNC = None
_DHO3_SHM = None


def _init_dho3_worker(shm_name, shape, dtype, spectral_axis, expected_peaks, p0, bounds, fit_func):
    global _DHO3_DATA, _DHO3_SPECTRAL_AXIS, _DHO3_EXPECTED_PEAKS, _DHO3_P0, _DHO3_BOUNDS, _DHO3_FIT_FUNC, _DHO3_SHM

    _DHO3_SHM = shared_memory.SharedMemory(name=shm_name)
    _DHO3_DATA = np.ndarray(shape, dtype=dtype, buffer=_DHO3_SHM.buf)
    _DHO3_SPECTRAL_AXIS = spectral_axis
    _DHO3_EXPECTED_PEAKS = expected_peaks
    _DHO3_P0 = p0
    _DHO3_BOUNDS = bounds
    _DHO3_FIT_FUNC = fit_func


def _fit_dho3_worker(idx):
    intensity_data_slice = _DHO3_DATA[idx + (slice(None),)]
    return _fitDHO2(
        idx,
        _DHO3_SPECTRAL_AXIS,
        intensity_data_slice,
        _DHO3_EXPECTED_PEAKS,
        _DHO3_P0,
        _DHO3_BOUNDS,
        _DHO3_FIT_FUNC,
    )


def _fit_concurrent_DHO3(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None):
    variables = int(expected_peaks * 3 + 2)

    if p0 is not None and len(p0) != variables:
        p0 = None
        print("Length of p0 does not match the model. Fallback to p0 = None")

    fit_results = np.full(intensity_data.shape[:-1] + (variables,), np.nan)
    cov_results = np.full(intensity_data.shape[:-1] + (variables, variables), np.nan)

    fit_func = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}[expected_peaks]
    spectral_axis = spectral_axis[0]

    # Masken im Spektrum behalten, aber als NaN speichern, damit curve_fit sie ignoriert
    data = np.asarray(np.ma.filled(intensity_data, np.nan), dtype=np.float64)
    spatial_shape = data.shape[:-1]
    indices = list(np.ndindex(spatial_shape))

    max_workers = max(1, int(os.cpu_count() * 0.75))
    chunksize = max(1, len(indices) // (max_workers * 4))

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    try:
        shared_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shared_data[:] = data

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_dho3_worker,
            initargs=(shm.name, data.shape, data.dtype, spectral_axis, expected_peaks, p0, bounds, fit_func),
        ) as executor:
            for idx, popt, pcov in tqdm(
                executor.map(_fit_dho3_worker, indices, chunksize=chunksize),
                total=len(indices),
                desc="Fitting Spectral data",
            ):
                fit_results[idx] = popt
                cov_results[idx] = pcov

    finally:
        shm.close()
        shm.unlink()

    return fit_results, cov_results
    
class DHO(FitStep):
    """
    Fit of one or multiple density distributions of Damped Harmonic Oscillators (DHO)
    I0 * 4 * LineWidth * freqShift**2 /(np.pi*(((x-Asymmetry)**2 - freqShift**2)**2 + 4*(LineWidth*(x-Asymmetry))**2)) + Background

    Parameters
    ----------
    expected_peaks : int
        The number of peaks to fit.
    p0 : list[floats]
        The initial guess
    bounds : list[list[floats], list[floats]] or None
        Boundaries for the plot
    **kwargs :
        

    """
    def __init__(self, *, expected_peaks, p0, bounds):
        super().__init__(_fit_concurrent_DHO2, expected_peaks=expected_peaks, p0=p0, bounds=bounds)

    
class Lorentzian(FitStep):
    """
    Fit of one or multiple Lorentzian lines
    I0/((((x-Asymmetry)**2 - freqShift**2)**2 + (LineWidth*(x-Asymmetry))**2)) + Background


    Parameters
    ----------
    expected_peaks : int
        The number of peaks to fit.
    p0 : list[floats]
        The initial guess
    bounds : list[list[floats], list[floats]] or None
        Boundaries for the plot
    **kwargs :
        

    """
    def __init__(self, *, expected_peaks, p0, bounds):
        super().__init__(_fit_concurrent_lorentzian, expected_peaks=expected_peaks, p0=p0, bounds=bounds)


def _fitDHO(x, y, z, t, spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs):
    # Convert masked values to nan
    if np.ma.is_masked(intensity_data_slice):
        # Check if all values are masked
        if np.ma.getmask(intensity_data_slice).all():
            return x, y, z, t, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)
        intensity_axis = intensity_data_slice.filled(np.nan)
    else:
        intensity_axis = intensity_data_slice.copy()
    intensity_axis[intensity_axis == 0] = np.nan
    
    # If all values are NaN, return NaN results
    if np.all(np.isnan(intensity_axis)):
        return x, y, z, t, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)
    
    if p0 is None:
        p0 = [1] * (expected_peaks * 3 + 2)  # Default initial guesses
    if bounds is None:
        bounds = (0, np.inf)  # No negative values allowed

    try:
        # Use nan_policy='omit' to handle NaN values
        popt, pcov = curve_fit(fit_funcs[expected_peaks], spectral_axis, intensity_axis, p0=p0, bounds=bounds, nan_policy='omit')
        return x, y, z, t, popt, pcov
    except RuntimeError:
        print(f"Fit not successful for pixel ({x},{y},{z},{t})")
        return x, y, z, t, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)

def _fitDHO2(idx, spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_func):
    # Convert masked values to nan
    if np.ma.is_masked(intensity_data_slice):
        if np.ma.getmask(intensity_data_slice).all():
            return idx, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)
        intensity_axis = intensity_data_slice.filled(np.nan)
    else:
        intensity_axis = intensity_data_slice.copy()
    intensity_axis[intensity_axis == 0] = np.nan

    if np.all(np.isnan(intensity_axis)):
        return idx, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)

    if p0 is None:
        p0 = [1] * (expected_peaks * 3 + 2)
    if bounds is None:
        bounds = (0, np.inf)

    try:
        popt, pcov = curve_fit(fit_func, spectral_axis, intensity_axis, p0=p0, bounds=bounds, nan_policy='omit')
        return idx, popt, pcov
    except RuntimeError:
        print(f"Fit not successful for pixel {idx}")
        return idx, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)
    
def _fit_concurrent_DHO2(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None):
    variables = int(expected_peaks * 3 + 2)
    spatial_shape = intensity_data.shape[:-1]
    fit_results = np.zeros(spatial_shape + (variables,))
    cov_results = np.empty(spatial_shape + (variables, variables))

    fit_funcs = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
    fit_func = fit_funcs[expected_peaks]
    spectral_axis = spectral_axis[0]
    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1,int(os.cpu_count()*0.25))) as executor:
        for idx in np.ndindex(spatial_shape):
            intensity_data_slice = intensity_data[idx + (slice(None),)]
            task = executor.submit(_fitDHO2, idx, spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_func)
            tasks.append(task)

        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Fitting Spectral data'):
            idx, popt, pcov = future.result()
            fit_results[idx] = popt
            cov_results[idx] = pcov
            
    return fit_results, cov_results

def _fitLorentzian(x, y, X, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs):
    # Convert masked values to nan
    if np.ma.is_masked(intensity_data_slice):
        Y = intensity_data_slice.filled(np.nan)
    else:
        Y = intensity_data_slice.copy()
    Y[Y == 0] = np.nan
    
    if p0 is None:
        p0 = [1] * (expected_peaks * 3 + 2)  # Default initial guesses
    if bounds is None:
        bounds = (0, np.inf)  # No negative values allowed

    try:
        # Use nan_policy='omit' to handle NaN values
        popt, pcov = curve_fit(fit_funcs[expected_peaks], X, Y, p0=p0, bounds=bounds, nan_policy='omit')
        return x, y, popt, pcov
    except RuntimeError:
        print(f"Fit not successful for pixel ({x},{y})")
        return x, y, np.full(expected_peaks * 3 + 2, np.nan), np.full((expected_peaks * 3 + 2, expected_peaks * 3 + 2), np.nan)


def _fit_concurrent_DHO(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None):
    variables = int(expected_peaks * 3 + 2)
    if p0 is not None and len(p0) != variables:
        p0 = None
        print('Length of p0 does not match the model. Fallback to p0 = None')
    fit_results = np.zeros((intensity_data.shape[0], intensity_data.shape[1], intensity_data.shape[2], intensity_data.shape[3], variables))
    cov_results = np.empty((intensity_data.shape[0], intensity_data.shape[1], intensity_data.shape[2], intensity_data.shape[3], variables, variables))

    fit_funcs = {1: _DHO_1, 2: _DHO_2, 3: _DHO_3}
    spectral_axis = spectral_axis[0]
    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1,int(os.cpu_count()*0.75))) as executor:
        for x in range(intensity_data.shape[0]):
            for y in range(intensity_data.shape[1]):
                for z in range(intensity_data.shape[2]):
                    for t in range(intensity_data.shape[3]):
                            intensity_data_slice = intensity_data[x, y, z, t, :]
                            # Skip if all values are masked
                            # if np.ma.is_masked(intensity_data_slice) and np.ma.getmask(intensity_data_slice).all():
                            #     fit_results[x, y, z, t, :] = np.nan
                            #     cov_results[x, y, z, t, :] = np.nan
                            #     continue
                            task = executor.submit(_fitDHO, x, y, z, t,  spectral_axis, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs)
                            tasks.append(task)
        
        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Fitting Spectral data'):
            x, y, z, t, popt, pcov = future.result()
            fit_results[x, y, z, t, :] = popt
            cov_results[x, y, z, t, :, :] = pcov

    return fit_results, cov_results

def _fit_concurrent_lorentzian(intensity_data, spectral_axis, expected_peaks, p0=None, bounds=None):
    variables = int(expected_peaks * 3 + 2)
    if p0 is not None and len(p0) != variables:
        p0 = None
        print('Length of p0 does not match the model. Fallback to p0 = None')
    fit_results = np.zeros((intensity_data.shape[0], intensity_data.shape[1], variables))
    cov_results = np.empty((intensity_data.shape[0], intensity_data.shape[1], variables, variables))

    fit_funcs = {1: _Lorentzian_1, 2: _Lorentzian_2, 3: _Lorentzian_3}
    X = spectral_axis[0]
    print('Starting of Multiprocessing can take up to 10 seconds.')
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1,int(os.cpu_count()*0.75))) as executor:
        for x in range(intensity_data.shape[0]):
            for y in range(intensity_data.shape[1]):
                intensity_data_slice = intensity_data[x, y, :]
                # Skip if all values are masked
                if np.ma.is_masked(intensity_data_slice) and np.ma.getmask(intensity_data_slice).all():
                    fit_results[x, y, :] = np.nan
                    cov_results[x, y, :, :] = np.nan
                    continue
                task = executor.submit(_fitLorentzian, x, y, X, intensity_data_slice, expected_peaks, p0, bounds, fit_funcs)
                tasks.append(task)

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(tasks), total=len(tasks), desc='Fitting Spectral data'):
            x, y, popt, pcov = future.result()
            fit_results[x, y, :] = popt
            cov_results[x, y, :, :] = pcov

    return fit_results, cov_results


def _DHO_1(x, I0, freqShift, LineWidth, Background, Asymmetry):
    return I0 * 4 * LineWidth * freqShift**2 /(np.pi*(((x-Asymmetry)**2 - freqShift**2)**2 + 4*(LineWidth*(x-Asymmetry))**2)) + Background


def _DHO_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry):
    return _DHO_1(x, I0, freqShift, LineWidth, Background, Asymmetry) + _DHO_1(x, I02, freqShift2, LineWidth2, Background, Asymmetry)


def _DHO_3(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, I03, freqShift3, LineWidth3, Background, Asymmetry):
    return _DHO_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry) + _DHO_1(x, I03, freqShift3, LineWidth3, Background, Asymmetry)


def _Lorentzian_1(x, I0, freqShift, LineWidth, Background, Asymmetry):
    return I0 * LineWidth /((((x-Asymmetry)**2 - freqShift**2)**2 + (LineWidth*(x-Asymmetry))**2)) + Background


def _Lorentzian_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry):
    return _Lorentzian_1(x, I0, freqShift, LineWidth, Background, Asymmetry) + _Lorentzian_1(x, I02, freqShift2, LineWidth2, Background, Asymmetry)


def _Lorentzian_3(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, I03, freqShift3, LineWidth3, Background, Asymmetry):
    return _Lorentzian_2(x, I0, freqShift, LineWidth, I02, freqShift2, LineWidth2, Background, Asymmetry) + _Lorentzian_1(x, I03, freqShift3, LineWidth3, Background, Asymmetry)