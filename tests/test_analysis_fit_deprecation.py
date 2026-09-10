"""
Structural tests for the analysis.fit reorganisation (2026-09): the fitters,
lineshape models, segmented fit and FitStep base class moved into the
brillouinpy.analysis.fit subpackage. The old module locations
(brillouinpy.analysis.fitmodel / .lineshapes / .segmented / .FitStep) must keep
working as thin re-exports that emit a DeprecationWarning, and the flat
brillouinpy.analysis.fit namespace must expose the public API.
"""
import importlib
import sys
import warnings

import pytest

import brillouinpy as bp
from brillouinpy.analysis import fit
from brillouinpy.analysis.fit import core, lineshapes, segmented, step


@pytest.mark.parametrize(
    "old_name, new_module, probe_attr",
    [
        ("fitmodel", core, "DHO"),
        ("fitmodel", core, "estimate_p0"),
        ("lineshapes", lineshapes, "register_lineshape"),
        ("lineshapes", lineshapes, "LINESHAPES"),
        ("segmented", segmented, "segmented_fit"),
        ("FitStep", step, "FitStep"),
    ],
)
def test_deprecated_module_reexports_and_warns(old_name, new_module, probe_attr):
    full = f"brillouinpy.analysis.{old_name}"
    sys.modules.pop(full, None)

    with pytest.deprecated_call():
        old_module = importlib.import_module(full)

    assert getattr(old_module, probe_attr) is getattr(new_module, probe_attr)


def test_old_fitmodel_private_names_still_importable():
    sys.modules.pop("brillouinpy.analysis.fitmodel", None)
    with pytest.deprecated_call():
        from brillouinpy.analysis.fitmodel import (  # noqa: F401
            _ConvolvedModel,
            _DHO_1,
            _fit_concurrent,
            _model_funcs,
        )

    assert _DHO_1 is lineshapes._DHO_1
    assert _model_funcs is lineshapes.model_funcs


def test_flat_fit_namespace_exposes_public_api():
    for name in ("DHO", "Lorentzian", "Gaussian", "PeakFit", "FitStep",
                 "estimate_p0", "estimate_peak_count", "irf_kernel",
                 "segmented_fit", "SegmentedFitResult",
                 "register_lineshape", "LINESHAPES"):
        assert hasattr(fit, name), name

    assert fit.DHO is core.DHO
    assert fit.segmented_fit is segmented.segmented_fit
    assert bp.analysis.segmented_fit is fit.segmented_fit
    assert fit.register_lineshape is lineshapes.register_lineshape


def test_importing_analysis_does_not_trigger_deprecation_warning():
    # Re-running brillouinpy.analysis' __init__ must not import any of the
    # deprecated shim modules (only brillouinpy.analysis.fit).
    for shim in ("fitmodel", "lineshapes", "segmented", "FitStep"):
        sys.modules.pop(f"brillouinpy.analysis.{shim}", None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import brillouinpy.analysis as analysis

        importlib.reload(analysis)
