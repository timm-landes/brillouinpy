# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:14:03 2025

@author: Timm
"""
import numpy as np


def fitDHO(self, filename, expected_peaks, p0, bounds, padding=100, full_output=True, deconv=True):
    coord = self.extract_coordinates(filename)
    raw_data, no_of_channels = self.import_DAT_File(filename)
    raw_data = np.vstack([self.frequency_axis, raw_data])
    
    paddata = self.data_padding(raw_data, padding)
    center, fitness, start, end, response = self.find_instrumental_response(paddata)
    pp_data = paddata.copy()
    pp_data[1] = paddata[1] - response[1]
    
    if deconv:
        pp_data[1] = restoration.richardson_lucy(pp_data[1], response[1] / np.max(response[1]), num_iter=20, clip=False, filter_epsilon=0.1)
    
    fit_funcs = {1: self.DHO, 2: self.DHO2, 3: self.DHO3}
    popt, cov = curve_fit(fit_funcs[expected_peaks], pp_data[0, 1:], pp_data[1, 1:], p0=p0, maxfev=10000000, bounds=bounds)
    
    return (coord, pp_data, popt, cov, fitness, center, start, end, response) if full_output else (coord, pp_data, popt, cov)

def fitDHO_wrapper(self, params):
    return self.fitDHO(*params)

def analyze(self, expected_peaks, p0, bounds, padding=100, full_output=True, deconv=True):
    shape = self.get_data_shape()  # Automatische Erkennung der Form der Daten

    # Initialisierung von Numpy-Arrays für jede relevante Messgröße
    results = {
        'amplitude_0': np.zeros(shape),
        'frequency_shift_0': np.zeros(shape),
        'linewidth_0': np.zeros(shape)
    }
    
    if expected_peaks > 1:
        results['amplitude_1'] = np.zeros(shape)
        results['frequency_shift_1'] = np.zeros(shape)
        results['linewidth_1'] = np.zeros(shape)

    if expected_peaks > 2:
        results['amplitude_2'] = np.zeros(shape)
        results['frequency_shift_2'] = np.zeros(shape)
        results['linewidth_2'] = np.zeros(shape)

    param_grid = [
        (f, expected_peaks, p0, bounds, padding, full_output, deconv)
        for f in self.file_list
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for params, future_result in zip(param_grid, executor.map(self.fitDHO_wrapper, param_grid)):
            coord, pp_data, popt, cov, fitness, center, start, end, response = future_result
            x, y, z, t = coord  # Erforderliche Koordinaten; ignorieren von Teilen, falls uninteressant

            # Ergebnisse basierend auf der erwarteten Anzahl an Peaks in die Arrays einsetzen
            results['amplitude_0'][x, y] = popt[0]
            results['frequency_shift_0'][x, y] = popt[1]
            results['linewidth_0'][x, y] = popt[2]

            if expected_peaks > 1:
                results['amplitude_1'][x, y] = popt[5]
                results['frequency_shift_1'][x, y] = popt[6]
                results['linewidth_1'][x, y] = popt[7]

            if expected_peaks > 2:
                results['amplitude_2'][x, y] = popt[10]
                results['frequency_shift_2'][x, y] = popt[11]
                results['linewidth_2'][x, y] = popt[12]

    return results