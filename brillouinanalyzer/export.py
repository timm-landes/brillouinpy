# -*- coding: utf-8 -*-
"""
Export utilities for interoperability with external tools.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import tifffile

from . import core


def _dho_parameter_names(expected_peaks: int) -> list:
    """Parameter names for the last axis of a DHO/Lorentzian FitStep result."""
    names = []
    for i in range(expected_peaks):
        suffix = "" if i == 0 else f"_{i + 1}"
        names += [f"I0{suffix}", f"freqShift{suffix}", f"LineWidth{suffix}"]
    names += ["Background", "Asymmetry"]
    return names


_MEASUREMENT_GROUP_TYPES = {"Measure", "Calibration_spectrum", "Impulse_response"}


def _open_hdf5_bls(filepath):
    try:
        from HDF5_BLS import Wrapper
    except ImportError as exc:
        raise ImportError(
            "Reading/writing the HDF5_BLS format requires the 'HDF5_BLS' package. "
            "Install it with 'pip install HDF5_BLS'."
        ) from exc
    return Wrapper(filepath)


def _find_measurement_groups(structure: dict, path: str = "Brillouin") -> list:
    groups = []
    if structure.get("Brillouin_type") in _MEASUREMENT_GROUP_TYPES:
        groups.append(path)
    for key, child in structure.items():
        if key == "Brillouin_type" or not isinstance(child, dict):
            continue
        groups.extend(_find_measurement_groups(child, f"{path}/{key}"))
    return groups


def list_measurements(filepath: str) -> list:
    """
    List the measurement-like groups (Brillouin_type "Measure", "Calibration_spectrum"
    or "Impulse_response") contained in an HDF5_BLS file.

    Useful to find the ``measure_group`` path to pass to :func:`from_hdf5_bls` when a
    file contains more than one measurement.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file.

    Returns
    -------
    list[str]
        The full paths (e.g. "Brillouin/Measure") of the measurement groups found.
    """
    wrapper = _open_hdf5_bls(filepath)
    try:
        return _find_measurement_groups(wrapper.get_structure()["Brillouin"])
    finally:
        wrapper.close()


def from_hdf5_bls(filepath: str, measure_group: Optional[str] = None) -> core.SpectralObject:
    """
    Read a measurement stored in the HDF5_BLS format
    (https://github.com/bio-brillouin/HDF5_BLS) back into a brillouinanalyzer
    spectral object, ready for further processing (preprocessing pipelines,
    fitting, ...).

    Requires the ``HDF5_BLS`` package (``pip install HDF5_BLS``).

    Parameters
    ----------
    filepath : str
        Path to the .h5 file.
    measure_group : str, optional
        Path of the group inside the file holding the data to read (e.g.
        "Brillouin/Measure"). If omitted, the file's only measurement group is
        used automatically; if the file contains several (e.g. a calibration
        spectrum next to the actual measurement), you must specify which one to
        read - use :func:`list_measurements` to see the available options.

    Returns
    -------
    SpectralObject
        A :class:`~brillouinanalyzer.Spectrum`/:class:`~brillouinanalyzer.SpectralImage`/
        :class:`~brillouinanalyzer.SpectralVolume`/:class:`~brillouinanalyzer.SpectralContainer`
        instance (chosen automatically based on the dimensionality of the stored
        data), with the group's attributes attached as a ``metadata`` dict.
    """
    wrapper = _open_hdf5_bls(filepath)
    try:
        if measure_group is None:
            candidates = _find_measurement_groups(wrapper.get_structure()["Brillouin"])
            if len(candidates) != 1:
                raise ValueError(
                    "Could not determine which measurement group to read automatically "
                    f"(found: {candidates}). Pass 'measure_group' explicitly - "
                    "use brillouinanalyzer.export.list_measurements(filepath) to see the options."
                )
            measure_group = candidates[0]
        elif measure_group not in _find_measurement_groups(wrapper.get_structure()["Brillouin"]):
            raise ValueError(
                f"'{measure_group}' is not a measurement group in '{filepath}'. "
                "Use brillouinanalyzer.export.list_measurements(filepath) to see the available options."
            )

        psd_names = wrapper.get_children_elements(measure_group, Brillouin_type="PSD")
        if not psd_names:
            psd_names = wrapper.get_children_elements(measure_group, Brillouin_type="Raw_data")
            if not psd_names:
                raise ValueError(f"No PSD or Raw_data dataset found under '{measure_group}'.")

        freq_names = wrapper.get_children_elements(measure_group, Brillouin_type="Frequency")
        if not freq_names:
            raise ValueError(f"No Frequency dataset found under '{measure_group}'.")

        intensity_data = np.asarray(wrapper[f"{measure_group}/{psd_names[0]}"])
        spectral_axis = np.asarray(wrapper[f"{measure_group}/{freq_names[0]}"])

        spectral_object = core._create_data(intensity_data, spectral_axis)
        spectral_object.metadata = wrapper.get_attributes(measure_group)

        return spectral_object
    finally:
        wrapper.close()


def to_hdf5_bls(
    spectral_object: core.SpectralObject,
    filepath: str,
    *,
    sample: Optional[str] = None,
    attributes: Optional[dict] = None,
    fit_result: Optional[np.ndarray] = None,
    expected_peaks: Optional[int] = None,
    parameter_names: Optional[Sequence[str]] = None,
    x_step: Optional[float] = None,
    y_step: Optional[float] = None,
    spatial_unit: str = "um",
    overwrite: bool = False,
) -> None:
    """
    Export a spectral object (and, optionally, a peak-fit result) to an HDF5 file
    compatible with the HDF5_BLS format (https://github.com/bio-brillouin/HDF5_BLS).

    Requires the ``HDF5_BLS`` package (``pip install HDF5_BLS``).

    Parameters
    ----------
    spectral_object : SpectralObject
        The (typically preprocessed) Brillouin data to export. Its ``spectral_data``
        is stored as the PSD and its ``spectral_axis`` as the frequency axis.
    filepath : str
        Destination path of the .h5 file.
    sample : str, optional
        Sample name, stored as a "Sample" attribute on the Measure group.
    attributes : dict, optional
        Additional free-form metadata stored as attributes on the Measure group.
    fit_result : numpy.ndarray of shape (..., n_params), optional
        The array of fitted parameters as returned by ``FitStep.apply`` (e.g.
        ``brillouinanalyzer.analysis.fitmodel.DHO``/``Lorentzian``). The last axis
        is expected to hold, per fitted peak, the triplet (I0, freqShift,
        LineWidth), followed by a single shared (Background, Asymmetry) pair.
    expected_peaks : int, optional
        Number of fitted peaks in ``fit_result``. Required if ``fit_result`` is
        given and ``parameter_names`` isn't.
    parameter_names : list[str], optional
        Explicit names for the last axis of ``fit_result``, overriding the default
        DHO/Lorentzian naming derived from ``expected_peaks``.
    x_step, y_step : float, optional
        Real-world pixel spacing along x/y, used to build the spatial abscissae.
        If omitted, pixel indices are stored instead.
    spatial_unit : str, optional
        Unit of ``x_step``/``y_step``, by default "um".
    overwrite : bool, optional
        Overwrite ``filepath`` if it already exists, by default False.
    """
    wrapper = _open_hdf5_bls(None)
    try:
        wrapper.create_group("Measure", parent_group="Brillouin", brillouin_type="Measure")
        wrapper.add_PSD(np.ma.filled(spectral_object.spectral_data, np.nan), parent_group="Brillouin/Measure")
        wrapper.add_frequency(np.asarray(spectral_object.spectral_axis), parent_group="Brillouin/Measure")

        shape = spectral_object.shape
        if shape != (1,):
            if len(shape) >= 1:
                x_axis = np.arange(shape[0]) * (x_step if x_step else 1)
                wrapper.add_abscissa(
                    x_axis, parent_group="Brillouin/Measure", name="X",
                    unit=spatial_unit if x_step else "px", dim_start=0, dim_end=1,
                )
            if len(shape) >= 2:
                y_axis = np.arange(shape[1]) * (y_step if y_step else 1)
                wrapper.add_abscissa(
                    y_axis, parent_group="Brillouin/Measure", name="Y",
                    unit=spatial_unit if y_step else "px", dim_start=1, dim_end=2,
                )

        meta = dict(attributes or {})
        if sample is not None:
            meta["Sample"] = sample
        if meta:
            wrapper.add_attributes(meta, parent_group="Brillouin/Measure")

        if fit_result is not None:
            fit_result = np.asarray(fit_result)
            if parameter_names is None:
                if expected_peaks is None:
                    raise ValueError(
                        "Provide either 'parameter_names' or 'expected_peaks' together with 'fit_result'."
                    )
                parameter_names = _dho_parameter_names(expected_peaks)

            n_params = fit_result.shape[-1]
            if len(parameter_names) != n_params:
                raise ValueError(
                    f"'parameter_names' has {len(parameter_names)} entries but "
                    f"fit_result's last axis has {n_params}."
                )

            n_peaks = (n_params - 2) // 3
            wrapper.create_group("Treatment", parent_group="Brillouin", brillouin_type="Treatment")

            for i in range(n_peaks):
                wrapper.add_treated_data(
                    parent_group="Brillouin/Treatment",
                    name_group=f"Treat_{i}",
                    amplitude=np.asarray(fit_result[..., 3 * i]),
                    shift=np.asarray(fit_result[..., 3 * i + 1]),
                    linewidth=np.asarray(fit_result[..., 3 * i + 2]),
                )

            for j, name in enumerate(parameter_names[3 * n_peaks:]):
                wrapper.add_other(
                    np.asarray(fit_result[..., 3 * n_peaks + j]),
                    parent_group="Brillouin/Treatment",
                    name=name,
                )

        wrapper.save_as_hdf5(filepath, overwrite=overwrite)
    finally:
        wrapper.close()


def fit_to_tiff(
    fit_result: np.ndarray,
    output_directory: str,
    *,
    expected_peaks: Optional[int] = None,
    parameter_names: Optional[Sequence[str]] = None,
    prefix: str = "fit",
) -> list:
    """
    Export each fitted parameter map of a peak-fit result as a separate 32-bit
    float TIFF image, so students can load them in ImageJ/Fiji or any other tool.

    Parameters
    ----------
    fit_result : numpy.ndarray of shape (x, y, n_params)
        The array of fitted parameters as returned by ``FitStep.apply`` (e.g.
        ``brillouinanalyzer.analysis.fitmodel.DHO``/``Lorentzian``).
    output_directory : str
        Directory the TIFF files are written to. Created if it doesn't exist.
    expected_peaks : int, optional
        Number of fitted peaks in ``fit_result``. Required if ``fit_result`` is
        given and ``parameter_names`` isn't.
    parameter_names : list[str], optional
        Explicit names for the last axis of ``fit_result`` used as filenames,
        overriding the default DHO/Lorentzian naming derived from ``expected_peaks``.
    prefix : str, optional
        Filename prefix, by default "fit". Files are written as
        "{prefix}_{parameter_name}.tif".

    Returns
    -------
    list[str]
        The paths of the written TIFF files.
    """
    import os

    fit_result = np.asarray(fit_result)
    if fit_result.ndim != 3:
        raise ValueError(
            f"'fit_result' must have shape (x, y, n_params), got shape {fit_result.shape}."
        )

    if parameter_names is None:
        if expected_peaks is None:
            raise ValueError("Provide either 'parameter_names' or 'expected_peaks'.")
        parameter_names = _dho_parameter_names(expected_peaks)

    n_params = fit_result.shape[-1]
    if len(parameter_names) != n_params:
        raise ValueError(
            f"'parameter_names' has {len(parameter_names)} entries but "
            f"fit_result's last axis has {n_params}."
        )

    os.makedirs(output_directory, exist_ok=True)

    written_files = []
    for i, name in enumerate(parameter_names):
        parameter_map = np.ma.filled(fit_result[..., i], np.nan).astype(np.float32)
        filepath = os.path.join(output_directory, f"{prefix}_{name}.tif")
        tifffile.imwrite(filepath, parameter_map)
        written_files.append(filepath)

    return written_files
