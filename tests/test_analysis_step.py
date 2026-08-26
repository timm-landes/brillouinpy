import numpy as np
import pytest

from brillouinpy.analysis import decompose, cluster
from brillouinpy.analysis.Step import variance_explained
from brillouinpy.core import SpectralImage


def _two_material_image(nx=10, ny=10, n_channels=100, seed=0):
    rng = np.random.default_rng(seed)
    axis = np.linspace(-20, 20, n_channels)

    def dho(shift, width):
        return np.exp(-0.5 * ((axis - shift) / width) ** 2)

    spectrum_a = dho(-6.0, 1.0)
    spectrum_b = dho(6.0, 1.0)

    abundance_a = np.tile(np.linspace(0, 1, nx)[:, None], (1, ny))
    abundance_b = 1 - abundance_a

    data = (
        abundance_a[..., None] * spectrum_a[None, None, :]
        + abundance_b[..., None] * spectrum_b[None, None, :]
    )
    data += rng.normal(scale=1e-3, size=data.shape)

    return SpectralImage(data, axis)


def test_variance_explained_increases_with_more_components():
    image = _two_material_image()

    result = variance_explained(
        image, lambda n: decompose.PCA(n_components=n, random_state=0), param_values=[1, 2, 3],
    )

    assert list(result.keys()) == [1, 2, 3]
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in result.values())
    # Two components should already explain (nearly) everything for two-material data.
    assert result[2] == pytest.approx(1.0, abs=1e-2)
    # More components can't explain less variance than fewer.
    assert result[1] <= result[2] + 1e-9
    assert result[2] <= result[3] + 1e-9


def test_variance_explained_works_with_kmeans():
    image = _two_material_image()

    result = variance_explained(
        image, lambda n: cluster.KMeans(n_clusters=n, random_state=0, n_init=10), param_values=[1, 2],
    )

    assert result[1] < result[2]
