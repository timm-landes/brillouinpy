# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:35:14 2025

@author: Timm
"""

import json
import numpy as np
import os
import re
import tqdm
import warnings
import scipy.optimize


# is_aligned() below is adapted, near-verbatim, from RamanSPy
# (https://github.com/barahona-research-group/RamanSPy), Copyright (c) 2023
# Dimitar Georgiev, licensed under the BSD 3-Clause License (see LICENSE and
# THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt in the repository root). The
# rest of this file is original.
def is_aligned(spectral_objects):
    """
    Checks whether a collection of spectral objects all share the same spectral axis.

    Parameters
    ----------
    spectral_objects : list of SpectralObject
        The spectral objects to compare.

    Returns
    -------
    bool
        ``True`` if all objects have an identical ``spectral_axis``, ``False`` otherwise.
    """
    unique_shift_axes = [
        np.array(unique) for unique in set(tuple(spectral_object.spectral_axis) for spectral_object in spectral_objects)]

    return len(unique_shift_axes) == 1


def wavelength_to_wavenumber(wavelengths, laser_excitation):
    """
    Converts wavelengths (nm) to Raman shift wavenumbers (cm :sup:`-1`) relative to a laser
    excitation wavelength.

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths to convert, in nm.
    laser_excitation : float
        Excitation wavelength of the laser used to acquire the data, in nm.

    Returns
    -------
    array_like
        The corresponding Raman shift wavenumbers, in cm :sup:`-1`.
    """
    return 1e7/laser_excitation - 1e7/wavelengths


def wavenumber_to_wavelength(raman_shifts, laser_excitation):
    """
    Converts Raman shift wavenumbers (cm :sup:`-1`) back to wavelengths (nm), the inverse of
    :func:`wavelength_to_wavenumber`.

    Parameters
    ----------
    raman_shifts : array_like
        Raman shift wavenumbers to convert, in cm :sup:`-1`.
    laser_excitation : float
        Excitation wavelength of the laser used to acquire the data, in nm.

    Returns
    -------
    array_like
        The corresponding wavelengths, in nm.
    """
    return 1/(1 / laser_excitation - raman_shifts / 1e7)


def raman_spectral_axis(grating, wavelength):
    """
    Computes the wavelength axis of a Raman spectrometer for a given grating and center
    wavelength, based on the monochromator/CCD geometry hard-coded in this function.

    Parameters
    ----------
    grating : float
        Groove density of the grating, in grooves/mm.
    wavelength : float
        Center wavelength of the monochromator, in nm.

    Returns
    -------
    numpy.ndarray
        The 2048-channel wavelength axis (matching the CCD's horizontal pixel count), in nm.
    """
    # Hab jetzt den Bandpass berechnet, der passt fast immer bis auf 0.2 nm mit der Berechnung von Horiba überein.
    deviationAngle = 21.26 # deviation angle of mono in degree
    deviationAngleRad = deviationAngle * np.pi/180. # deviation angle in rad
    deviationHalfAngleRad = deviationAngleRad/2. # deviation angle in rad
    focalLength = 318.72 # output focal length of mono in mm
    inclineAngle = -5.5 # inclination angle of CCD in degree
    inclineAngleRad = inclineAngle * np.pi/180. # inclination angle of CCD in rad
    order = 1 # diffraction order
    detHPixel = 2048 # horizontal pixels
    pixelSize = 0.014 # in mm
    detWidth = detHPixel * pixelSize # detector width in mm
    alpha = np.arcsin((10**-6 * order * wavelength * grating)/(2 * np.cos(deviationHalfAngleRad))) - deviationAngleRad/2
    beta = deviationAngleRad+alpha
    linearDispersion = 10**6 * np.cos(beta) * np.cos(inclineAngleRad)/(focalLength * grating* order)
    monoBandpass = linearDispersion * detWidth
    pixelBandpass = monoBandpass / detHPixel
    lowerspectrumedge = wavelength - monoBandpass/2
    higherspectrumedge = wavelength + monoBandpass/2
    wavelength_axis = np.linspace(lowerspectrumedge, higherspectrumedge, detHPixel, endpoint=True)
    return wavelength_axis


def fsr(mirror_spacing):
    """
    Computes the Free Spectral Range (FSR) of a tandem Fabry-Perot interferometer.

    Parameters
    ----------
    mirror_spacing : float
        The mirror spacing of the interferometer, in m.

    Returns
    -------
    float
        The free spectral range, in Hz.
    """
    speed_of_light = 299792458  # m/s
    return speed_of_light / (2 * mirror_spacing)


def brillouin_spectral_axis(mirror_spacing, scan_amplitude, no_of_channels, laser_wavelength = 532.1e-9):
    """
    Computes the Brillouin frequency-shift axis of a tandem Fabry-Perot interferometer scan.

    Parameters
    ----------
    mirror_spacing : float
        The mirror spacing of the interferometer, in m.
    scan_amplitude : float
        The mirror scan amplitude, in the same units expected by :func:`fsr`'s
        ``mirror_spacing``/wavelength ratio (i.e. such that ``fsr(mirror_spacing) *
        scan_amplitude / laser_wavelength`` yields a frequency in Hz).
    no_of_channels : int
        Number of spectral channels to generate.
    laser_wavelength : float, optional
        Excitation laser wavelength, in m. Default is 532.1 nm (``532.1e-9``).

    Returns
    -------
    numpy.ndarray
        The symmetric frequency-shift axis, in GHz, of length ``no_of_channels``.
    """
    freq_limits = fsr(mirror_spacing) * scan_amplitude / laser_wavelength
    freq_limits_GHz = freq_limits * 1e-9
    return np.linspace(-freq_limits_GHz, freq_limits_GHz, no_of_channels)


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
    :func:`brillouin_spectral_axis`'s own default of 532.1 nm.
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
    Like :func:`brillouin_spectral_axis`, but reads ``mirror_spacing``,
    ``scan_amplitude`` and ``laser_wavelength`` from the measurement's
    ``META.json`` (see :func:`read_meta`) instead of requiring them as arguments.

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
        spectral_dimension = 2048 # for Raman we assume always 2048 channels as this is the horizontal pixel number of the CCD
    elif spectral_data_type == 'Brillouin':
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory)
            if (f.endswith('.DAT')) and os.path.isfile(os.path.join(directory, f))
        )
        _, spectral_dimension = import_DAT_File(files[0]) # for Brillouin we read the spectral dimension from one file
    else:
        raise ValueError('Spectral data not supported')

    # prepare spectral data: (x, y, z, time)
    max_coord = [0, # x
                 0, # y
                 0, # z
                 0] # time
    for f in files:
        coordinates = extract_coordinates(f, spectral_data_type)
        if coordinates:
            max_coord = [max(m, c) for m, c in zip(max_coord, coordinates)]
    x_dim, y_dim, z_dim, timepoint =  max_coord


        # Create a masked array to handle missing data points
    spectral_data_array = np.ma.masked_all((x_dim+1, y_dim+1, z_dim+1, timepoint+1, spectral_dimension))

    if len(files) != (x_dim+1) * (y_dim+1) * (z_dim+1) * (timepoint+1):
        warnings.warn("The number of points do not match the expected number of (x, y, z, t) points.")

    for f in tqdm.tqdm(files, desc="Processing Spectral data"):
        coordinates = extract_coordinates(f, spectral_data_type)
        if not coordinates:
            continue

        x, y, z, t = coordinates
        try:
            if f.endswith('.csv'):
                data = np.loadtxt(f, delimiter=',',skiprows=1)
                print(data)
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

def load_spectral_image(project_path, spectral_data_type):
    """
    Loads all Raman/Brillouin data files from ``<project_path>/data`` into a single
    ``(x, y, spectral)`` masked array (a single z-layer/timepoint), inferring the spatial
    extent from the coordinates encoded in the filenames (see :func:`extract_coordinates`).

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
    x_dim, y_dim, z_dim, timepoint =  max_coord

        # Create a masked array to handle missing data points
    spectral_data_array = np.ma.masked_all((x_dim+1, y_dim+1, z_dim+1, timepoint+1, spectral_dimension))

    if len(files) != (x_dim+1) * (y_dim+1) * (z_dim+1) * (timepoint+1):
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

    return spectral_data_array


def load_spectral_image_brio2(project_path, spectral_data_type):
    """
    Variant of :func:`load_spectral_image` for data acquired on the Brio 2 setup, where CSV
    rows are reversed before use (``[::-1]``) to match the spectral axis convention.

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
    x_dim, y_dim, z_dim, timepoint =  max_coord

        # Create a masked array to handle missing data points
    spectral_data_array = np.ma.masked_all((x_dim+1, y_dim+1, z_dim+1, timepoint+1, spectral_dimension))

    if len(files) != (x_dim+1) * (y_dim+1) * (z_dim+1) * (timepoint+1):
        warnings.warn("The number of points do not match the expected number of (x, y, z, t) points.")

    for f in tqdm.tqdm(files, desc="Processing Spectral data"):
        coordinates = extract_coordinates(f, spectral_data_type)
        if not coordinates:
            continue

        x, y, z, t = coordinates
        try:
            if f.endswith('.csv'):
                data = np.loadtxt(f, delimiter=',', skiprows=1)[::-1]
                spectrum = data[:,1]
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

    return spectral_data_array


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

def TFP_IRF_Analysis(intensity_data, spectral_axis):
    '''
    Analyze the Instrumental Response Function (IRF) from the intensity data in terms of laser line shape and intenisty
    Assuming the data is 2D (x, y, spectral) or 3D (x, y, z, spectral) or 4D (x, y, z, t, spectral).
    '''
    spectral_response = np.zeros(intensity_data.shape)
    laser_fitness = np.zeros(intensity_data.shape[:-1])
    fwhm_values = np.zeros(intensity_data.shape[:-1])


    for x in range(intensity_data.shape[0]):
        for y in range(intensity_data.shape[1]):
            for z in range(intensity_data.shape[2]):
                for t in range(intensity_data.shape[3]):
                    start, end = _find_instrumental_response(intensity_data[x, y, z, t, :])
                    spectral_response[x, y, z, t, start:end] = intensity_data[x, y, z, t, start:end]
                    try:
                        spectrum = spectral_response[x, y, z, t, :]
                        p0 = [np.max(spectrum), spectral_axis[np.argmax(spectrum)], 0.1, np.min(spectrum)]
                        popt, _ = scipy.optimize.curve_fit(_gauss, spectral_axis, spectrum, p0=p0)
                        a, mu, sigma, c = popt
                        fwhm = 2.3548 * abs(sigma)  # FWHM for Gaussian
                        fwhm_values[x, y, z, t] = fwhm
                        laser_fitness[x, y, z, t] = a
                        # Optional: Save fitted curve if needed
                        # spectral_response[x, y, z, t, :] = _gauss(spectral_axis, *popt)
                    except Exception:
                        fwhm_values[x, y, z, t] = np.nan
                        laser_fitness[x, y, z, t] = np.nan
    return spectral_response, spectral_axis, laser_fitness, fwhm_values


def _gauss(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c


def _find_instrumental_response(pixel_spectrum):
    max_index = np.argmax(pixel_spectrum)

    start = next((i for i in range(max_index, 0, -1) if pixel_spectrum[i] == 0), 0)
    end = next((i for i in range(max_index, len(pixel_spectrum)) if pixel_spectrum[i] == 0), len(pixel_spectrum))

    return start, end


# --- New function for Raman data preparation ---
def prepare_raman_data(project_path):
    """
    Loads all Raman CSV files in the project's data directory, extracts the wavelength axis from the first file,
    and returns a masked array of the spectral data and the wavelength axis.
    """
    import numpy as np
    import os
    import re
    import tqdm
    import warnings

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
    data_array = np.ma.masked_all((x_dim+1, y_dim+1, z_dim+1, t_dim+1, spectral_dimension))

    if len(files) != (x_dim+1)*(y_dim+1)*(z_dim+1)*(t_dim+1):
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
