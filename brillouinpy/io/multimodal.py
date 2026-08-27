# -*- coding: utf-8 -*-
"""
Loading data for modalities other than the group's primary TFP-Brillouin setup
(see :mod:`brillouinpy.io.tfp`) - currently just Raman.
"""
import os
import re
import warnings

import numpy as np
import tqdm


def prepare_raman_data(project_path):
    """
    Loads all Raman CSV files in the project's data directory, extracts the wavelength axis
    from the first file, and returns a masked array of the spectral data and the wavelength
    axis.
    """
    directory = os.path.join(project_path, 'data')
    files = sorted(
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.endswith('.csv') and os.path.isfile(os.path.join(directory, f))
    )
    if not files:
        raise FileNotFoundError("No Raman CSV files found in the data directory.")

    # Extract wavelength axis from the first file (reverse for correct order)
    first_data = np.loadtxt(files[0], delimiter=',', skiprows=1)
    if first_data.ndim == 1:
        wavelength_axis = np.loadtxt(files[0], delimiter=',', skiprows=1, usecols=0)[::-1]
        spectral_dimension = wavelength_axis.shape[0]
    else:
        wavelength_axis = first_data[:, 0][::-1]
        spectral_dimension = wavelength_axis.shape[0]

    # Prepare shape (x, y, z, t, spectral_dimension)
    max_coord = [0, 0, 0, 0]
    pattern = r'Raman_(\d+)_(\d+)_(\d+)_(\d+)'
    for f in files:
        m = re.search(pattern, os.path.basename(f))
        if m:
            coords = list(map(int, m.groups()))
            max_coord = [max(m_, c) for m_, c in zip(max_coord, coords)]
    x_dim, y_dim, z_dim, t_dim = max_coord
    data_array = np.ma.masked_all((x_dim + 1, y_dim + 1, z_dim + 1, t_dim + 1, spectral_dimension))

    if len(files) != (x_dim + 1) * (y_dim + 1) * (z_dim + 1) * (t_dim + 1):
        warnings.warn("The number of points do not match the expected number of (x, y, z, t) points.")

    for f in tqdm.tqdm(files, desc="Processing Raman CSV data"):
        m = re.search(pattern, os.path.basename(f))
        if not m:
            continue
        x, y, z, t = map(int, m.groups())
        try:
            data = np.loadtxt(f, delimiter=',', skiprows=1)
            if data.ndim == 1:
                spectrum = (data[1] if len(data) > 1 else data)
                spectrum = np.array(spectrum)[::-1]
            else:
                spectrum = data[:, 1][::-1]
            data_array[x, y, z, t, :] = spectrum
        except Exception as e:
            warnings.warn(f"Could not load data for coordinates ({x}, {y}, {z}, {t}): {str(e)}")
            # Point remains masked

    return data_array, wavelength_axis
