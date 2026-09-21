import os

import numpy as np
import pytest
import tifffile

from brillouinpy.io import export


def test_dho_parameter_names_one_peak():
    assert export._dho_parameter_names(1) == ["I0", "freqShift", "LineWidth", "Background", "axis_shift"]


def test_dho_parameter_names_three_peaks():
    assert export._dho_parameter_names(3) == [
        "I0", "freqShift", "LineWidth",
        "I0_2", "freqShift_2", "LineWidth_2",
        "I0_3", "freqShift_3", "LineWidth_3",
        "Background", "axis_shift",
    ]


@pytest.fixture
def fit_result():
    # shape (nx, ny, n_params) for a single-peak DHO fit:
    # I0, freqShift, LineWidth, Background, axis_shift
    rng = np.random.default_rng(0)
    return rng.random((4, 3, 5))


def test_fit_to_tiff_writes_one_file_per_parameter(tmp_path, fit_result):
    written = export.fit_to_tiff(fit_result, str(tmp_path), expected_peaks=1)

    expected_names = export._dho_parameter_names(1)
    assert written == [str(tmp_path / f"fit_{name}.tif") for name in expected_names]

    for i, filepath in enumerate(written):
        assert os.path.exists(filepath)
        loaded = tifffile.imread(filepath)
        assert loaded.dtype == np.float32
        assert np.allclose(loaded, fit_result[..., i].astype(np.float32))


def test_fit_to_tiff_parameter_names_override(tmp_path, fit_result):
    custom_names = ["a", "b", "c", "d", "e"]
    written = export.fit_to_tiff(fit_result, str(tmp_path), parameter_names=custom_names)

    assert written == [str(tmp_path / f"fit_{name}.tif") for name in custom_names]


def test_fit_to_tiff_parameter_names_length_mismatch_raises(tmp_path, fit_result):
    with pytest.raises(ValueError, match="entries"):
        export.fit_to_tiff(fit_result, str(tmp_path), parameter_names=["a", "b"])


def test_fit_to_tiff_wrong_ndim_raises(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        export.fit_to_tiff(np.zeros((4, 5)), str(tmp_path), expected_peaks=1)


def test_fit_to_tiff_missing_expected_peaks_raises(tmp_path, fit_result):
    with pytest.raises(ValueError, match="Provide either"):
        export.fit_to_tiff(fit_result, str(tmp_path))


def test_fit_to_tiff_masked_array_writes_nan(tmp_path, fit_result):
    masked = np.ma.masked_array(fit_result, mask=np.zeros_like(fit_result, dtype=bool))
    masked.mask[0, 0, 0] = True

    written = export.fit_to_tiff(masked, str(tmp_path), expected_peaks=1)

    loaded = tifffile.imread(written[0])
    assert np.isnan(loaded[0, 0])
    assert not np.isnan(loaded[1, 0])


def test_fit_to_tiff_prefix(tmp_path, fit_result):
    written = export.fit_to_tiff(fit_result, str(tmp_path), expected_peaks=1, prefix="custom")
    assert all(os.path.basename(f).startswith("custom_") for f in written)
