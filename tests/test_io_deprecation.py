"""
Structural tests for the brillouinpy.io reorganisation (2026-08-27): the new
brillouinpy.io package exposes the expected submodules/re-exports, and the old
brillouinpy.utils.*/brillouinpy.export locations still work but warn.
"""
import warnings

import pytest

import brillouinpy as bp
from brillouinpy import io, utils


def test_io_exposes_submodules():
    assert io.tfp is not None
    assert io.legacy is not None
    assert io.multimodal is not None
    assert io.export is not None


def test_io_reexports_common_functions_at_top_level():
    assert io.prepare_brillouin_data is io.tfp.prepare_brillouin_data
    assert io.read_meta is io.tfp.read_meta
    assert io.to_brim is io.export.to_brim
    assert io.from_hdf5_bls is io.export.from_hdf5_bls


@pytest.mark.parametrize(
    "old_name, new_module_name, new_func_name",
    [
        ("read_meta", "tfp", "read_meta"),
        ("brillouin_spectral_axis_from_meta", "tfp", "brillouin_spectral_axis_from_meta"),
        ("extract_coordinates", "tfp", "extract_coordinates"),
        ("import_DAT_File", "tfp", "import_DAT_File"),
        ("prepare_brillouin_data", "tfp", "prepare_brillouin_data"),
        ("load_spectral_image", "legacy", "load_spectral_image"),
        ("load_spectral_image_brio2", "legacy", "load_spectral_image_brio2"),
        ("prepare_raman_data", "multimodal", "prepare_raman_data"),
    ],
)
def test_deprecated_utils_wrapper_forwards_to_new_location(monkeypatch, old_name, new_module_name, new_func_name):
    new_module = getattr(io, new_module_name)
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return "sentinel"

    monkeypatch.setattr(new_module, new_func_name, fake)

    with pytest.deprecated_call():
        result = getattr(utils, old_name)(1, two=2)

    assert result == "sentinel"
    assert calls == [((1,), {"two": 2})]


def test_brillouinpy_export_shim_still_works_and_warns():
    with pytest.deprecated_call():
        module = bp.export

    assert module.to_brim is io.export.to_brim


def test_plain_import_does_not_trigger_export_deprecation_warning():
    # A fresh 'import brillouinpy' must not itself touch the deprecated
    # 'brillouinpy.export' shim (only actually accessing 'bp.export' should).
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import importlib

        import brillouinpy
        importlib.reload(brillouinpy)
