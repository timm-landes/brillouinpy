from pathlib import Path

import numpy as np
import pytest

try:
    import HDF5_BLS  # noqa: F401
except (ImportError, SyntaxError):
    pytest.skip(
        "HDF5_BLS not importable (missing, or incompatible with this Python "
        "version - e.g. its 3.12+-only f-string syntax on older Pythons)",
        allow_module_level=True,
    )

from brillouinpy.io import export
from brillouinpy.core import SpectralImage, Spectrum


def _dho(x, amplitude, freq_shift, linewidth, background=0.0, axis_shift=0.0):
    xs = x - axis_shift
    return amplitude * 4 * linewidth * freq_shift ** 2 / (
        np.pi * ((xs ** 2 - freq_shift ** 2) ** 2 + 4 * (linewidth * xs) ** 2)
    ) + background


@pytest.fixture
def image():
    axis = np.linspace(-20, 20, 60)
    data = np.stack([
        [_dho(axis, 5e-3, 8.5, 1.0) for _ in range(3)]
        for _ in range(4)
    ])
    return SpectralImage(data, axis)


def test_to_hdf5_bls_from_hdf5_bls_roundtrip(tmp_path, image):
    filepath = str(tmp_path / "image.h5")

    export.to_hdf5_bls(image, filepath, sample="Test sample",
                       x_step=0.5, y_step=0.5, spatial_unit="um")
    reloaded = export.from_hdf5_bls(filepath)

    assert reloaded.shape == image.shape
    assert np.allclose(reloaded.spectral_axis, image.spectral_axis)
    assert np.allclose(
        np.ma.filled(reloaded.spectral_data, np.nan),
        np.ma.filled(image.spectral_data, np.nan),
        equal_nan=True,
    )
    assert reloaded.metadata["Sample"] == "Test sample"


def test_to_hdf5_bls_roundtrip_spectrum(tmp_path, image):
    filepath = str(tmp_path / "spectrum.h5")
    spectrum = image[0, 0]
    assert isinstance(spectrum, Spectrum)

    export.to_hdf5_bls(spectrum, filepath)
    reloaded = export.from_hdf5_bls(filepath)

    assert np.allclose(
        np.ma.filled(reloaded.spectral_data, np.nan).squeeze(),
        np.ma.filled(spectrum.spectral_data, np.nan),
        equal_nan=True,
    )


def test_to_hdf5_bls_with_fit_result(tmp_path, image):
    filepath = str(tmp_path / "fit.h5")
    nx, ny = image.shape
    fitted = np.tile([5e-3, 8.5, 1.0, 1e-4, 0.0], (nx, ny, 1))

    export.to_hdf5_bls(image, filepath, fit_result=fitted, expected_peaks=1)

    assert Path(filepath).exists()
    reloaded = export.from_hdf5_bls(filepath)
    assert reloaded.shape == image.shape


def test_to_hdf5_bls_requires_expected_peaks_with_fit_result(tmp_path, image):
    filepath = str(tmp_path / "fit_missing_peaks.h5")
    fitted = np.zeros(image.shape + (5,))

    with pytest.raises(ValueError, match="expected_peaks"):
        export.to_hdf5_bls(image, filepath, fit_result=fitted)


def test_to_hdf5_bls_raises_without_overwrite(tmp_path, image):
    filepath = str(tmp_path / "exists.h5")
    export.to_hdf5_bls(image, filepath)

    with pytest.raises(Exception):
        export.to_hdf5_bls(image, filepath)
