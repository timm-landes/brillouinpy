import json

import numpy as np
import pytest

from brillouinpy import utils

NESTED_META = {
    "Brillouin": {
        "Mirror_spacing_mm": 6.0,
        "Scan_amplitude_nm": 480,
        "Scanning_cycles": 15,
        "Input_pinhole_um": "150",
        "Output_pinhole_um": "700",
    },
    "Laser": {"Model": "Cobolt Samba", "Power_mW": 1.0, "Wavelength_nm": 532.1},
    "General": {"Date": "2025-01-01T00:00:00", "Sample": "Test"},
}

FLAT_META_WITH_AMPLITUDE = {
    "FP1_spacing_mm": 6,
    "FP_scan_amplitude": 480,
    "Laser_wavelenght_nm": 532.1,
    "Sample": "Test",
}

FLAT_META_WITHOUT_AMPLITUDE = {
    "FP1_spacing_mm": 6,
    "Laser_wavelenght_nm": 532.1,
    "Sample": "Test",
}

UNRECOGNISED_META = {"Foo": "bar"}


def _write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_read_meta_finds_file_in_data_subfolder(tmp_path):
    (tmp_path / "data").mkdir()
    _write_meta(tmp_path / "data" / "META.json", NESTED_META)

    assert utils.read_meta(str(tmp_path)) == NESTED_META


def test_read_meta_finds_file_directly_in_project_path(tmp_path):
    _write_meta(tmp_path / "META.json", NESTED_META)

    assert utils.read_meta(str(tmp_path)) == NESTED_META


def test_read_meta_prefers_data_subfolder_over_project_root(tmp_path):
    (tmp_path / "data").mkdir()
    _write_meta(tmp_path / "data" / "META.json", NESTED_META)
    _write_meta(tmp_path / "META.json", FLAT_META_WITH_AMPLITUDE)

    assert utils.read_meta(str(tmp_path)) == NESTED_META


def test_read_meta_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.read_meta(str(tmp_path))


@pytest.mark.parametrize(
    "meta, expected_mirror_spacing_m, expected_scan_amplitude_m, expected_wavelength_m",
    [
        (NESTED_META, 6e-3, 480e-9, 532.1e-9),
        (FLAT_META_WITH_AMPLITUDE, 6e-3, 480e-9, 532.1e-9),
    ],
)
def test_brillouin_scan_parameters_from_meta(
    meta, expected_mirror_spacing_m, expected_scan_amplitude_m, expected_wavelength_m
):
    mirror_spacing, scan_amplitude, laser_wavelength = utils._brillouin_scan_parameters_from_meta(meta)

    assert mirror_spacing == pytest.approx(expected_mirror_spacing_m)
    assert scan_amplitude == pytest.approx(expected_scan_amplitude_m)
    assert laser_wavelength == pytest.approx(expected_wavelength_m)


def test_brillouin_scan_parameters_from_meta_flat_without_amplitude_raises():
    with pytest.raises(KeyError, match="FP_scan_amplitude"):
        utils._brillouin_scan_parameters_from_meta(FLAT_META_WITHOUT_AMPLITUDE)


def test_brillouin_scan_parameters_from_meta_unrecognised_schema_raises():
    with pytest.raises(KeyError, match="Unrecognised META.json schema"):
        utils._brillouin_scan_parameters_from_meta(UNRECOGNISED_META)


def test_brillouin_spectral_axis_from_meta_matches_manual_call(tmp_path):
    _write_meta(tmp_path / "META.json", NESTED_META)

    axis = utils.brillouin_spectral_axis_from_meta(str(tmp_path), no_of_channels=200)
    expected = utils.brillouin_spectral_axis(
        mirror_spacing=6e-3, scan_amplitude=480e-9, no_of_channels=200, laser_wavelength=532.1e-9
    )

    assert np.allclose(axis, expected)
