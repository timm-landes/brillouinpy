import os

import numpy as np
import pytest

from conftest import write_dat_file
from brillouinpy.io import legacy

# Both load_spectral_image and load_spectral_image_brio2 hardcode Raman's
# spectral_dimension to 2048, regardless of what's actually in the file - so a
# successful Raman load needs a 2048-point spectrum.
_RAMAN_LENGTH = 2048


def _write_raman_csv(directory, x, y, z, t, spectrum):
    # load_spectral_image's Raman/.csv path: np.loadtxt(f, delimiter=',') with no
    # skiprows, then spectrum = data[1] (the second row).
    arr = np.zeros((2, len(spectrum)))
    arr[1] = spectrum
    path = os.path.join(directory, f"Raman_{x}_{y}_{z}_{t}.csv")
    np.savetxt(path, arr, delimiter=",")
    return path


def _write_brio2_raman_csv(directory, x, y, z, t, spectrum):
    # load_spectral_image_brio2's Raman/.csv path: np.loadtxt(f, delimiter=',',
    # skiprows=1)[::-1], then spectrum = data[:, 1]. One header row is skipped, the
    # remaining rows are reversed before column 1 is read - so write column 1
    # pre-reversed, and the function's own [::-1] restores the intended order.
    n = len(spectrum)
    arr = np.zeros((n, 2))
    arr[:, 1] = np.asarray(spectrum)[::-1]
    path = os.path.join(directory, f"Raman_{x}_{y}_{z}_{t}.csv")
    with open(path, "w") as f:
        f.write("header\n")
    with open(path, "a") as f:
        np.savetxt(f, arr, delimiter=",")
    return path


@pytest.mark.parametrize("loader_name", ["load_spectral_image", "load_spectral_image_brio2"])
def test_unknown_spectral_data_type_raises(tmp_path, loader_name):
    loader = getattr(legacy, loader_name)
    with pytest.raises(ValueError, match="Spectral data not supported"):
        loader(str(tmp_path), "Foo")


@pytest.mark.parametrize("loader_name", ["load_spectral_image", "load_spectral_image_brio2"])
def test_brillouin_grid_shape_and_masked_gap(tmp_path, loader_name):
    loader = getattr(legacy, loader_name)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    values = {(0, 0): [1.0, 2.0, 3.0], (1, 0): [4.0, 5.0, 6.0], (0, 1): [7.0, 8.0, 9.0]}
    # (1, 1) is deliberately not written - a missing point in an otherwise full 2x2 grid.
    for (x, y), v in values.items():
        write_dat_file(str(data_dir), "Bri", x, y, 0, 0, v)

    with pytest.warns(UserWarning, match="do not match"):
        result = loader(str(tmp_path), "Brillouin")

    assert result.shape == (2, 2, 3)
    for (x, y), v in values.items():
        assert not np.ma.is_masked(result[x, y])
        assert np.allclose(result[x, y], v)
    assert np.ma.is_masked(result[1, 1])


@pytest.mark.parametrize("loader_name", ["load_spectral_image", "load_spectral_image_brio2"])
def test_raman_full_grid_no_warning(tmp_path, loader_name, recwarn):
    loader = getattr(legacy, loader_name)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    spectrum_00 = np.linspace(0, 1, _RAMAN_LENGTH)
    spectrum_10 = np.linspace(1, 2, _RAMAN_LENGTH)
    write = _write_raman_csv if loader_name == "load_spectral_image" else _write_brio2_raman_csv
    write(str(data_dir), 0, 0, 0, 0, spectrum_00)
    write(str(data_dir), 1, 0, 0, 0, spectrum_10)

    result = loader(str(tmp_path), "Raman")

    assert not any("do not match" in str(w.message) for w in recwarn.list)
    assert result.shape == (2, 1, _RAMAN_LENGTH)
    assert np.allclose(result[0, 0], spectrum_00)
    assert np.allclose(result[1, 0], spectrum_10)


@pytest.mark.parametrize("loader_name", ["load_spectral_image", "load_spectral_image_brio2"])
def test_more_than_one_z_layer_raises(tmp_path, loader_name):
    loader = getattr(legacy, loader_name)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    write_dat_file(str(data_dir), "Bri", 0, 0, 0, 0, [1.0, 2.0])
    write_dat_file(str(data_dir), "Bri", 0, 0, 1, 0, [3.0, 4.0])

    with pytest.raises(ValueError, match="z-layer"):
        loader(str(tmp_path), "Brillouin")


def test_brio2_csv_row_order_is_reversed_relative_to_plain_read(tmp_path):
    # A fixture where forward vs. reversed row reading gives visibly different
    # values, to confirm the [::-1] in load_spectral_image_brio2 actually does
    # something rather than being a no-op.
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    ascending = np.arange(_RAMAN_LENGTH, dtype=float)
    _write_brio2_raman_csv(str(data_dir), 0, 0, 0, 0, ascending)

    result = legacy.load_spectral_image_brio2(str(tmp_path), "Raman")

    assert np.allclose(result[0, 0], ascending)
    # a naive forward (non-reversed) read of the same file would have produced the
    # descending order instead
    assert not np.allclose(result[0, 0], ascending[::-1])
