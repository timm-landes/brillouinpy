# -*- coding: utf-8 -*-
"""
Superseded loaders, kept for reading old datasets that predate
:func:`brillouinpy.io.tfp.prepare_brillouin_data` (which supports the full
``(x, y, z, t, spectral)`` shape instead of a single z-layer/timepoint). Not the
recommended path for new data - use :mod:`brillouinpy.io.tfp` instead.
"""
import os
import warnings

import numpy as np
import tqdm

from .tfp import extract_coordinates, import_DAT_File


def load_spectral_image(project_path, spectral_data_type):
    """
    Loads all Raman/Brillouin data files from ``<project_path>/data`` into a single
    ``(x, y, spectral)`` masked array (a single z-layer/timepoint), inferring the spatial
    extent from the coordinates encoded in the filenames (see
    :func:`brillouinpy.io.tfp.extract_coordinates`).

    Superseded by :func:`brillouinpy.io.tfp.prepare_brillouin_data`, which supports more
    than one z-layer/timepoint; kept for reading old datasets.

    Parameters
    ----------
    project_path : str
        Path to the project directory; files are expected in its ``data`` subfolder.
    spectral_data_type : {'Raman', 'Brillouin'}
        Which kind of data to load. Raman data is read from ``.csv``/``.txt`` files
        (2048 channels assumed), Brillouin data from ``.DAT`` files.

    Returns
    -------
    numpy.ma.MaskedArray
        Array of shape ``(x_dim, y_dim, spectral_dimension)``; points for which no matching
        file was found, or which failed to load, remain masked.
    """
    directory = os.path.join(project_path, 'data')
    # sort files
    if spectral_data_type == 'Raman':
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if (f.endswith('.csv') or f.endswith('.txt')) and os.path.isfile(os.path.join(directory, f))
        )
        spectral_dimension = 2048
    elif spectral_data_type == 'Brillouin':
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if (f.endswith('.DAT')) and os.path.isfile(os.path.join(directory, f))
        )
        _, spectral_dimension = import_DAT_File(files[0])
    else:
        raise ValueError('Spectral data not supported')

    # prepare spectral data: (x, y, spectral_dimension)
    max_coord = [0, 0, 0, 0]
    for f in files:
        coordinates = extract_coordinates(f, spectral_data_type)
        if coordinates:
            max_coord = [max(m, c) for m, c in zip(max_coord, coordinates)]
    x_dim, y_dim, z_dim, timepoint = max_coord

    # Create a masked array to handle missing data points
    spectral_data_array = np.ma.masked_all((x_dim + 1, y_dim + 1, z_dim + 1, timepoint + 1, spectral_dimension))

    if len(files) != (x_dim + 1) * (y_dim + 1) * (z_dim + 1) * (timepoint + 1):
        warnings.warn("The number of points do not match the expected number of (x, y, z, t) points.")

    for f in tqdm.tqdm(files, desc="Processing Spectral data"):
        coordinates = extract_coordinates(f, spectral_data_type)
        if not coordinates:
            continue

        x, y, z, t = coordinates
        try:
            if f.endswith('.csv'):
                data = np.loadtxt(f, delimiter=',')
                spectrum = data[1]
            elif f.endswith('.txt'):
                data = np.loadtxt(f)
                spectrum = data[:-1, 1]
            elif f.endswith('.DAT'):
                spectrum, _ = import_DAT_File(f)

            spectral_data_array[x, y, z, t, :] = spectrum

        except Exception as e:
            warnings.warn(f"Could not load data for coordinates ({x}, {y}, {z}, {t}): {str(e)}")
            # Point remains masked
            continue

    if spectral_data_array.shape[2] != 1 or spectral_data_array.shape[3] != 1:
        raise ValueError(
            f"Found {spectral_data_array.shape[2]} z-layer(s) and {spectral_data_array.shape[3]} "
            "timepoint(s); load_spectral_image only supports a single z-layer/timepoint. Use "
            "brillouinpy.io.tfp.prepare_brillouin_data instead."
        )

    return spectral_data_array[:, :, 0, 0, :]


def load_spectral_image_brio2(project_path, spectral_data_type):
    """
    Variant of :func:`load_spectral_image` for data acquired on the Brio 2 setup, where CSV
    rows are reversed before use (``[::-1]``) to match the spectral axis convention.

    Superseded by :func:`brillouinpy.io.tfp.prepare_brillouin_data`; kept for reading old
    datasets acquired on that setup.

    Parameters
    ----------
    project_path : str
        Path to the project directory; files are expected in its ``data`` subfolder.
    spectral_data_type : {'Raman', 'Brillouin'}
        Which kind of data to load. Raman data is read from ``.csv``/``.txt`` files
        (2048 channels assumed), Brillouin data from ``.DAT`` files.

    Returns
    -------
    numpy.ma.MaskedArray
        Array of shape ``(x_dim, y_dim, spectral_dimension)``; points for which no matching
        file was found, or which failed to load, remain masked.

    Notes
    -----
    The file-count sanity check in this function references ``z_dim``/``timepoint``, which are
    not defined in this 2D variant; that check currently raises ``NameError`` instead of the
    intended warning if the file count doesn't match.
    """
    directory = os.path.join(project_path, 'data')
    # sort files
    if spectral_data_type == 'Raman':
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if (f.endswith('.csv') or f.endswith('.txt')) and os.path.isfile(os.path.join(directory, f))
        )
        spectral_dimension = 2048
    elif spectral_data_type == 'Brillouin':
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if (f.endswith('.DAT')) and os.path.isfile(os.path.join(directory, f))
        )
        _, spectral_dimension = import_DAT_File(files[0])
    else:
        raise ValueError('Spectral data not supported')

    # prepare spectral data: (x, y, spectral_dimension)
    max_coord = [0, 0, 0, 0]
    for f in files:
        coordinates = extract_coordinates(f, spectral_data_type)
        if coordinates:
            max_coord = [max(m, c) for m, c in zip(max_coord, coordinates)]
    x_dim, y_dim, z_dim, timepoint = max_coord

    # Create a masked array to handle missing data points
    spectral_data_array = np.ma.masked_all((x_dim + 1, y_dim + 1, z_dim + 1, timepoint + 1, spectral_dimension))

    if len(files) != (x_dim + 1) * (y_dim + 1) * (z_dim + 1) * (timepoint + 1):
        warnings.warn("The number of points do not match the expected number of (x, y, z, t) points.")

    for f in tqdm.tqdm(files, desc="Processing Spectral data"):
        coordinates = extract_coordinates(f, spectral_data_type)
        if not coordinates:
            continue

        x, y, z, t = coordinates
        try:
            if f.endswith('.csv'):
                data = np.loadtxt(f, delimiter=',', skiprows=1)[::-1]
                spectrum = data[:, 1]
            elif f.endswith('.txt'):
                data = np.loadtxt(f)
                spectrum = data[:-1, 1]
            elif f.endswith('.DAT'):
                spectrum, _ = import_DAT_File(f)

            spectral_data_array[x, y, z, t, :] = spectrum

        except Exception as e:
            warnings.warn(f"Could not load data for coordinates ({x}, {y}, {z}, {t}): {str(e)}")
            # Point remains masked
            continue

    if spectral_data_array.shape[2] != 1 or spectral_data_array.shape[3] != 1:
        raise ValueError(
            f"Found {spectral_data_array.shape[2]} z-layer(s) and {spectral_data_array.shape[3]} "
            "timepoint(s); load_spectral_image_brio2 only supports a single z-layer/timepoint. "
            "Use brillouinpy.io.tfp.prepare_brillouin_data instead."
        )

    return spectral_data_array[:, :, 0, 0, :]
