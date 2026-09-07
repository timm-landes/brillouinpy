import sys

import numpy as np
import pytest

from brillouinpy.analysis.unmix import VCA, _fcls, _nnls, _ucls
from brillouinpy.core import SpectralImage


def test_unmix_module_does_not_import_pysptools_at_load_time():
    # PPI/FIPPI/NFINDR still need pysptools (lazily, inside their __init__/call),
    # but importing the module itself - and using VCA - must not require it.
    sys.modules.pop('pysptools', None)
    sys.modules.pop('brillouinpy.analysis.unmix', None)
    import brillouinpy.analysis.unmix as unmix  # noqa: F401
    assert 'pysptools' not in sys.modules


def _synthetic_mixture(seed=0):
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 20, 60)
    endmember_a = np.exp(-((axis - 6) ** 2) / (2 * 0.5 ** 2))
    endmember_b = np.exp(-((axis - 12) ** 2) / (2 * 0.5 ** 2))
    true_endmembers = np.stack([endmember_a, endmember_b])  # (2, bands)

    abundances_a = np.linspace(0, 1, 25)
    true_abundances = np.stack([abundances_a, 1 - abundances_a], axis=1)  # (25, 2)
    spectral_data = true_abundances @ true_endmembers
    spectral_data += rng.normal(scale=1e-3, size=spectral_data.shape)

    return spectral_data, true_endmembers, true_abundances, axis


def test_ucls_recovers_known_abundances():
    spectral_data, endmembers, true_abundances, _ = _synthetic_mixture()
    abundances = _ucls(spectral_data, endmembers)

    assert abundances.shape == true_abundances.shape
    assert np.allclose(abundances, true_abundances, atol=0.05)


def test_nnls_is_non_negative():
    spectral_data, endmembers, true_abundances, _ = _synthetic_mixture()
    abundances = _nnls(spectral_data, endmembers)

    assert abundances.shape == true_abundances.shape
    assert (abundances >= -1e-8).all()
    assert np.allclose(abundances, true_abundances, atol=0.05)


def test_fcls_is_non_negative_and_sums_to_one():
    spectral_data, endmembers, true_abundances, _ = _synthetic_mixture()
    abundances = _fcls(spectral_data, endmembers)

    assert abundances.shape == true_abundances.shape
    assert (abundances >= -1e-8).all()
    assert np.allclose(abundances.sum(axis=1), 1.0, atol=1e-2)
    assert np.allclose(abundances, true_abundances, atol=0.05)


def test_unmixer_rejects_invalid_abundance_method():
    spectral_data, endmembers, _, axis = _synthetic_mixture()
    image = SpectralImage(spectral_data[:, np.newaxis, :], axis)

    with pytest.raises(ValueError, match="not a valid abundance method"):
        VCA(n_endmembers=2, abundance_method='bogus').apply(image)


@pytest.mark.parametrize('abundance_method', ['ucls', 'nnls', 'fcls'])
def test_vca_end_to_end(abundance_method):
    spectral_data, _, _, axis = _synthetic_mixture()
    image = SpectralImage(spectral_data[:, np.newaxis, :], axis)

    abundances, endmembers = VCA(n_endmembers=2, abundance_method=abundance_method).apply(image)

    assert len(abundances) == 2
    assert len(endmembers) == 2
    assert np.asarray(abundances[0]).shape == (25, 1)
