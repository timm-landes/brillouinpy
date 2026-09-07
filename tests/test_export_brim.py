from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("brimfile")

from brillouinpy.io import export
from brillouinpy.core import SpectralImage, SpectralVolume, Spectrum


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


def test_to_brim_from_brim_roundtrip(tmp_path, image):
    filepath = str(tmp_path / "image.brim.zarr")

    export.to_brim(
        image, filepath,
        sample="Test sample", laser_wavelength_nm=532.1,
        x_step_um=0.5, y_step_um=0.5,
    )
    reloaded = export.from_brim(filepath)

    assert type(reloaded) is SpectralImage
    assert reloaded.shape == image.shape
    assert np.allclose(reloaded.spectral_axis, image.spectral_axis)
    assert np.allclose(
        np.ma.filled(reloaded.spectral_data, np.nan),
        np.ma.filled(image.spectral_data, np.nan),
        equal_nan=True,
    )
    assert reloaded.metadata["Experiment"]["Sample"][0] == "Test sample"
    assert reloaded.metadata["Optics"]["Wavelength"] == (532.1, "nm")


def test_from_brim_to_brim_carries_over_metadata_and_px_size(tmp_path, image):
    first = str(tmp_path / "first.brim.zarr")
    second = str(tmp_path / "second.brim.zarr")

    export.to_brim(
        image, first,
        sample="Sample A", laser_wavelength_nm=660.0,
        x_step_um=0.25, y_step_um=0.75,
    )

    reloaded = export.from_brim(first)
    assert reloaded.px_size_um["x"] == 0.25
    assert reloaded.px_size_um["y"] == 0.75

    # no metadata / px-size args: everything should ride along on the object
    export.to_brim(reloaded, second)
    again = export.from_brim(second)

    assert again.metadata["Experiment"]["Sample"][0] == "Sample A"
    assert again.metadata["Optics"]["Wavelength"] == (660.0, "nm")
    assert again.px_size_um["x"] == 0.25
    assert again.px_size_um["y"] == 0.75

    # explicit args still win
    third = str(tmp_path / "third.brim.zarr")
    export.to_brim(reloaded, third, x_step_um=1.5)
    assert export.from_brim(third).px_size_um["x"] == 1.5


def test_to_brim_without_metadata_roundtrips_cleanly(tmp_path, image):
    # regression test for a brimfile <= 1.7.0 quirk: reading a data group that
    # never had any metadata written raises when the file is opened read-only.
    filepath = str(tmp_path / "no_meta.brim.zarr")

    export.to_brim(image, filepath)
    reloaded = export.from_brim(filepath)

    # to_brim always writes Experiment.Datetime (some viewers crash without it).
    assert "Datetime" in reloaded.metadata["Experiment"]


def test_to_brim_writes_explicit_acquisition_datetime(tmp_path, image):
    filepath = str(tmp_path / "dt.brim.zarr")

    export.to_brim(image, filepath, acquisition_datetime="2026-09-03T12:34:56")
    reloaded = export.from_brim(filepath)

    assert reloaded.metadata["Experiment"]["Datetime"][0] == "2026-09-03T12:34:56"


def test_to_brim_raises_without_overwrite(tmp_path, image):
    filepath = str(tmp_path / "image.brim.zarr")
    export.to_brim(image, filepath)

    with pytest.raises(FileExistsError):
        export.to_brim(image, filepath, overwrite=False)

    export.to_brim(image, filepath, overwrite=True)  # should not raise


def test_to_brim_as_zip_writes_a_single_file(tmp_path, image):
    filepath = str(tmp_path / "image.brim.zip")

    export.to_brim(image, filepath, as_zip=True, laser_wavelength_nm=532.1)

    assert Path(filepath).is_file()
    assert not Path(filepath).is_dir()

    reloaded = export.from_brim(filepath)
    assert reloaded.shape == image.shape
    assert np.allclose(
        np.ma.filled(reloaded.spectral_data, np.nan),
        np.ma.filled(image.spectral_data, np.nan),
        equal_nan=True,
    )
    assert reloaded.metadata["Optics"]["Wavelength"] == (532.1, "nm")

    # overwrite=True on an existing zip should also not raise
    export.to_brim(image, filepath, as_zip=True, overwrite=True)


def test_to_brim_with_single_peak_fit_result(tmp_path, image):
    filepath = str(tmp_path / "fit.brim.zarr")
    nx, ny = image.shape
    fitted = np.tile([5e-3, 8.5, 1.0, 1e-4, 0.0], (nx, ny, 1))

    export.to_brim(image, filepath, fit_result=fitted, expected_peaks=1)

    import brimfile as brim
    f = brim.File(filepath)
    try:
        data_group = f.get_data(0)
        ar = data_group.get_analysis_results()
        assert ar.fit_model == brim.AnalysisResults.FitModel.DHO
        shift_image, _ = ar.get_image(brim.AnalysisResults.Quantity.Shift, brim.AnalysisResults.PeakType.AntiStokes)
        assert np.allclose(shift_image, 8.5)
    finally:
        f.close()


def test_to_brim_with_two_peak_fit_result(tmp_path, image):
    filepath = str(tmp_path / "fit2.brim.zarr")
    nx, ny = image.shape
    fitted = np.tile([5e-3, 6.0, 1.0, 5e-3, 11.0, 1.2, 1e-4, 0.0], (nx, ny, 1))

    export.to_brim(image, filepath, fit_result=fitted, expected_peaks=2)

    import brimfile as brim
    f = brim.File(filepath)
    try:
        ar = f.get_data(0).get_analysis_results()
        shift_peak0, _ = ar.get_image(brim.AnalysisResults.Quantity.Shift, brim.AnalysisResults.PeakType.AntiStokes, index=0)
        shift_peak1, _ = ar.get_image(brim.AnalysisResults.Quantity.Shift, brim.AnalysisResults.PeakType.AntiStokes, index=1)
        assert np.allclose(shift_peak0, 6.0)
        assert np.allclose(shift_peak1, 11.0)
    finally:
        f.close()


def test_to_brim_requires_expected_peaks_with_fit_result(tmp_path, image):
    filepath = str(tmp_path / "fit_missing_peaks.brim.zarr")
    fitted = np.zeros(image.shape + (5,))

    with pytest.raises(ValueError, match="expected_peaks"):
        export.to_brim(image, filepath, fit_result=fitted)


def test_to_brim_from_brim_roundtrip_spectrum(tmp_path, image):
    filepath = str(tmp_path / "spectrum.brim.zarr")
    spectrum = image[0, 0]
    assert isinstance(spectrum, Spectrum)

    export.to_brim(spectrum, filepath)
    reloaded = export.from_brim(filepath)

    # brim doesn't distinguish a bare spectrum from a 1x1-pixel image
    assert reloaded.shape == (1, 1)
    assert np.allclose(
        np.ma.filled(reloaded.spectral_data, np.nan).squeeze(),
        np.ma.filled(spectrum.spectral_data, np.nan),
    )


def test_to_brim_from_brim_roundtrip_volume(tmp_path):
    axis = np.linspace(-20, 20, 40)
    data = np.random.default_rng(0).random((3, 2, 2, 40))
    volume = SpectralVolume(data, axis)

    filepath = str(tmp_path / "volume.brim.zarr")
    export.to_brim(volume, filepath, x_step_um=1.0, y_step_um=1.0, z_step_um=2.0)
    reloaded = export.from_brim(filepath)

    assert type(reloaded) is SpectralVolume
    assert reloaded.shape == volume.shape
    assert np.allclose(reloaded.spectral_data, volume.spectral_data)


def test_list_brim_measurements(tmp_path, image):
    filepath = str(tmp_path / "image.brim.zarr")
    export.to_brim(image, filepath)

    groups = export.list_brim_measurements(filepath)

    assert len(groups) == 1
    assert groups[0]["index"] == 0


def test_from_brim_missing_brimfile_raises_helpful_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "brimfile":
            raise ImportError("no module named brimfile")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="brimfile"):
        export.from_brim("does_not_matter.brim.zarr")
