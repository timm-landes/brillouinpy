import os

import numpy as np
import pytest

from brillouinpy.io import multimodal


def _write_raman_csv(directory, x, y, z, t, wavelengths, intensities):
    # prepare_raman_data's 2-D branch: np.loadtxt(f, delimiter=',', skiprows=1)
    # gives a (rows, 2) array of (wavelength, intensity) pairs when there's more
    # than one data row.
    path = os.path.join(directory, f"Raman_{x}_{y}_{z}_{t}.csv")
    with open(path, "w") as f:
        f.write("wavelength,intensity\n")
        for wl, i in zip(wavelengths, intensities):
            f.write(f"{wl},{i}\n")
    return path


def _write_single_row_raman_csv(directory, x, y, z, t, wavelength, intensity):
    # prepare_raman_data's 1-D branch: exactly one data row after the header
    # collapses np.loadtxt's result to a 1-D array instead of 2-D.
    return _write_raman_csv(directory, x, y, z, t, [wavelength], [intensity])


def test_no_csv_files_raises(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError, match="No Raman CSV files"):
        multimodal.prepare_raman_data(str(tmp_path))


def test_grid_shape_masked_gap_and_axis_reversal(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    wavelengths = [500.0, 550.0, 600.0]
    values = {(0, 0): [1.0, 2.0, 3.0], (1, 0): [4.0, 5.0, 6.0], (0, 1): [7.0, 8.0, 9.0]}
    # (1, 1) is deliberately not written - a missing point in an otherwise full 2x2 grid.
    for (x, y), v in values.items():
        _write_raman_csv(str(data_dir), x, y, 0, 0, wavelengths, v)

    with pytest.warns(UserWarning, match="do not match"):
        data, axis = multimodal.prepare_raman_data(str(tmp_path))

    assert data.shape == (2, 2, 1, 1, 3)
    assert np.allclose(axis, wavelengths[::-1])
    for (x, y), v in values.items():
        assert not np.ma.is_masked(data[x, y, 0, 0])
        assert np.allclose(data[x, y, 0, 0], v[::-1])
    assert np.ma.is_masked(data[1, 1, 0, 0])


def test_single_row_file_loads_correctly(tmp_path):
    # Regression test: prepare_raman_data used to crash (IndexError) extracting the
    # wavelength axis from a file with exactly one data row, since usecols=0
    # collapses to a 0-d array that [::-1] can't slice.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_single_row_raman_csv(str(data_dir), 0, 0, 0, 0, 532.1, 42.0)

    data, axis = multimodal.prepare_raman_data(str(tmp_path))

    assert data.shape == (1, 1, 1, 1, 1)
    assert np.allclose(axis, [532.1])
    assert not np.ma.is_masked(data[0, 0, 0, 0])
    assert np.allclose(data[0, 0, 0, 0], [42.0])


def test_corrupt_file_warns_and_stays_masked(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_raman_csv(str(data_dir), 0, 0, 0, 0, [500.0, 600.0], [1.0, 2.0])
    corrupt_path = os.path.join(str(data_dir), "Raman_1_0_0_0.csv")
    with open(corrupt_path, "w") as f:
        f.write("wavelength,intensity\nnot,numeric\n")

    with pytest.warns(UserWarning, match="Could not load data"):
        data, axis = multimodal.prepare_raman_data(str(tmp_path))

    assert not np.ma.is_masked(data[0, 0, 0, 0])
    assert np.ma.is_masked(data[1, 0, 0, 0])
