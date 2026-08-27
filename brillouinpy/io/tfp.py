# -*- coding: utf-8 -*-
"""
Loading data acquired on a tandem Fabry-Perot (TFP) interferometer setup, from this
group's raw-data directory/file conventions (``Bri_x_y_z_t.DAT`` filenames,
``META.json`` metadata). This is the actively used/maintained loading path - for
data acquired differently, see :mod:`brillouinpy.io.legacy` (superseded loaders,
kept for old datasets) and :mod:`brillouinpy.io.multimodal` (Raman).

If your own setup/lab uses a different raw-data layout, this module is meant as a
worked example to model a custom loader on - see the "Writing your own loader"
section of the tutorial (``docs/tutorial.md``).
"""
import json
import os
import re
import warnings

import numpy as np
import tqdm

from ..utils import brillouin_spectral_axis


def read_meta(project_path):
    """
    Loads a measurement's ``META.json`` metadata file.

    Looks for ``META.json`` first inside ``<project_path>/data``, then directly
    inside ``<project_path>`` - both locations occur across existing datasets,
    depending on when the measurement was taken.

    Parameters
    ----------
    project_path : str
        Path to the project directory.

    Returns
    -------
    dict
        The parsed contents of ``META.json``.

    Raises
    ------
    FileNotFoundError
        If no ``META.json`` is found in either location.
    """
    for candidate in (
        os.path.join(project_path, 'data', 'META.json'),
        os.path.join(project_path, 'META.json'),
    ):
        if os.path.isfile(candidate):
            with open(candidate, encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No 'META.json' found in '{project_path}' or its 'data' subfolder."
    )


def _brillouin_scan_parameters_from_meta(meta):
    """
    Extracts ``(mirror_spacing, scan_amplitude, laser_wavelength)``, in meters,
    from a parsed ``META.json`` dict.

    Transparently handles the metadata schema versions found across existing
    datasets:

    - current (nested): ``meta['Brillouin']['Mirror_spacing_mm']`` /
      ``['Scan_amplitude_nm']``, ``meta['Laser']['Wavelength_nm']``.
    - legacy (flat): ``meta['FP1_spacing_mm']`` / ``['FP_scan_amplitude']``,
      ``meta['Laser_wavelenght_nm']`` (sic - the original key is misspelled).

    If the laser wavelength isn't present under either schema, falls back to
    :func:`brillouinpy.utils.brillouin_spectral_axis`'s own default of 532.1 nm.
    """
    if isinstance(meta.get('Brillouin'), dict) and isinstance(meta.get('Laser'), dict):
        brillouin = meta['Brillouin']
        laser = meta['Laser']
        try:
            mirror_spacing_mm = brillouin['Mirror_spacing_mm']
            scan_amplitude_nm = brillouin['Scan_amplitude_nm']
        except KeyError as exc:
            raise KeyError(
                f"META.json's 'Brillouin' section is missing {exc}; "
                f"has keys {sorted(brillouin.keys())}."
            ) from exc
        laser_wavelength_nm = laser.get('Wavelength_nm', 532.1)

    elif 'FP1_spacing_mm' in meta:
        mirror_spacing_mm = meta['FP1_spacing_mm']
        if 'FP_scan_amplitude' not in meta:
            raise KeyError(
                "META.json uses the legacy flat schema and doesn't include "
                "'FP_scan_amplitude' (only present in some older measurements) - "
                "pass 'scan_amplitude' to brillouin_spectral_axis explicitly instead."
            )
        scan_amplitude_nm = meta['FP_scan_amplitude']
        laser_wavelength_nm = meta.get('Laser_wavelenght_nm', 532.1)

    else:
        raise KeyError(
            "Unrecognised META.json schema (top-level keys: "
            f"{sorted(meta.keys())}). Expected either a 'Brillouin'/'Laser' "
            "section (current schema) or a flat 'FP1_spacing_mm' key (legacy schema)."
        )

    return mirror_spacing_mm * 1e-3, scan_amplitude_nm * 1e-9, laser_wavelength_nm * 1e-9


def brillouin_spectral_axis_from_meta(project_path, no_of_channels):
    """
    Like :func:`brillouinpy.utils.brillouin_spectral_axis`, but reads
    ``mirror_spacing``, ``scan_amplitude`` and ``laser_wavelength`` from the
    measurement's ``META.json`` (see :func:`read_meta`) instead of requiring them
    as arguments.

    Parameters
    ----------
    project_path : str
        Path to the project directory (containing ``META.json``, directly or in
        its ``data`` subfolder).
    no_of_channels : int
        Number of spectral channels to generate. Not stored in ``META.json`` -
        get it from the loaded data's spectral dimension instead, e.g.
        ``brillouin_data.shape[-1]``.

    Returns
    -------
    numpy.ndarray
        The symmetric frequency-shift axis, in GHz, of length ``no_of_channels``.

    Raises
    ------
    FileNotFoundError
        If no ``META.json`` is found for ``project_path`` (see :func:`read_meta`).
    KeyError
        If ``META.json`` doesn't match a recognised schema, or is missing a
        required field within the schema it does match.
    """
    meta = read_meta(project_path)
    mirror_spacing, scan_amplitude, laser_wavelength = _brillouin_scan_parameters_from_meta(meta)
    return brillouin_spectral_axis(mirror_spacing, scan_amplitude, no_of_channels, laser_wavelength)


def extract_coordinates(filepath, spectral_data_type):
    """
    Extract coordinates from the filename.

    Parameters:
        filepath (str): The path to the file.

    Returns:
        tuple: Extracted coordinates or None if pattern doesn't match.
    """
    filename = os.path.basename(filepath)
    # Regular expression for extracting coordinates
    if spectral_data_type == 'Raman':
        pattern = r'Raman_(\d+)_(\d+)_(\d+)_(\d+)'
    elif spectral_data_type == 'Brillouin':
        pattern = r'Bri_(\d+)_(\d+)_(\d+)_(\d+)'
    match = re.match(pattern, filename)
    return tuple(map(int, match.groups())) if match else None


def import_DAT_File(file):
    """
    Reads a single Brillouin ``.DAT`` spectrum file (11-line header).

    Parameters
    ----------
    file : str
        Path to the ``.DAT`` file.

    Returns
    -------
    data : numpy.ndarray
        The loaded spectral data.
    spectral_dimension : int
        The number of spectral channels (``len(data)``).
    """
    data = np.loadtxt(file, skiprows=11)
    return data, len(data)


def prepare_brillouin_data(project_path, spectral_data_type):
    """
    Loads all Raman/Brillouin data files from ``<project_path>/data`` into a single
    ``(x, y, z, t, spectral)`` masked array, inferring the spatial extent from the
    coordinates encoded in the filenames (see :func:`extract_coordinates`).

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
        Array of shape ``(x_dim, y_dim, z_dim, timepoint, spectral_dimension)``; points for
        which no matching file was found, or which failed to load, remain masked.
    """
    directory = os.path.join(project_path, 'data')
    # sort files
    if spectral_data_type == 'Raman':
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if (f.endswith('.csv') or f.endswith('.txt')) and os.path.isfile(os.path.join(directory, f))
        )
        spectral_dimension = 2048  # for Raman we assume always 2048 channels as this is the horizontal pixel number of the CCD
    elif spectral_data_type == 'Brillouin':
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if (f.endswith('.DAT')) and os.path.isfile(os.path.join(directory, f))
        )
        _, spectral_dimension = import_DAT_File(files[0])  # for Brillouin we read the spectral dimension from one file
    else:
        raise ValueError('Spectral data not supported')

    # prepare spectral data: (x, y, z, time)
    max_coord = [0,  # x
                 0,  # y
                 0,  # z
                 0]  # time
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
                data = np.loadtxt(f, delimiter=',', skiprows=1)
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
    return spectral_data_array
