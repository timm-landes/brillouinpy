# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:35:14 2025

@author: Timm
"""

import numpy as np
import os, re, tqdm, warnings


def is_aligned(raman_objects):
    unique_shift_axes = [
        np.array(unique) for unique in set(tuple(raman_object.spectral_axis) for raman_object in raman_objects)]

    return len(unique_shift_axes) == 1


def wavelength_to_wavenumber(wavelengths, laser_excitation):
    return 1e7/laser_excitation - 1e7/wavelengths


def wavenumber_to_wavelength(raman_shifts, laser_excitation):
    return 1/(1 / laser_excitation - raman_shifts / 1e7)


def raman_spectral_axis(grating, wavelength):
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
    speed_of_light = 299792458  # m/s
    return speed_of_light / (2 * mirror_spacing)


def brillouin_spectral_axis(mirror_spacing, scan_amplitude, no_of_channels, laser_wavelength = 532.1e-9):
    freq_limits = fsr(mirror_spacing) * scan_amplitude / laser_wavelength
    freq_limits_GHz = freq_limits * 1e-9
    return np.linspace(-freq_limits_GHz, freq_limits_GHz, no_of_channels)

    
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
    return spectral_data_array

def load_spectral_image(project_path, spectral_data_type):
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
    x_dim, y_dim, _, _ =  max_coord
    
        # Create a masked array to handle missing data points
    spectral_data_array = np.ma.masked_all((x_dim+1, y_dim+1, spectral_dimension))
    
    if len(files) != (x_dim+1) * (y_dim+1):
        warnings.warn("The number of points do not match the expected number of (x, y) points.")
    
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
                
            spectral_data_array[x, y, :] = spectrum
            
        except Exception as e:
            warnings.warn(f"Could not load data for coordinates ({x}, {y}): {str(e)}")
            # Point remains masked
            continue
    
    return spectral_data_array


def load_spectral_image_brio2(project_path, spectral_data_type):
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
    x_dim, y_dim, _, _ =  max_coord
    
        # Create a masked array to handle missing data points
    spectral_data_array = np.ma.masked_all((x_dim+1, y_dim+1, spectral_dimension))
    
    if len(files) != (x_dim+1) * (y_dim+1):
        warnings.warn("The number of points do not match the expected number of (y, y) points.")
    
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
                
            spectral_data_array[x, y, :] = spectrum
            
        except Exception as e:
            warnings.warn(f"Could not load data for coordinates ({x}, {y}): {str(e)}")
            # Point remains masked
            continue
    
    return spectral_data_array


def import_DAT_File(file):
    data = np.loadtxt(file, skiprows=11)
    return data, len(data)
