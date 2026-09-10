import sys

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from _synthetic_data import additive_blob_image, three_component_image


def test_segmented_module_does_not_import_sklearn_at_load_time():
    # The default k-means classifier needs sklearn, but it is imported lazily
    # inside _default_classifier so the DHO fit's multiprocessing workers (which
    # re-import brillouinpy.analysis on spawn) never pay for it.
    sys.modules.pop("sklearn", None)
    sys.modules.pop("brillouinpy.analysis.fit.segmented", None)
    import brillouinpy.analysis.fit.segmented  # noqa: F401

    assert "sklearn" not in sys.modules


@pytest.mark.parametrize("n_components", [1, 4, 0, "2"])
def test_segmented_fit_rejects_unsupported_component_count(n_components):
    image, _, _ = additive_blob_image(nx=6, ny=6, n_channels=60)
    with pytest.raises(ValueError):
        import brillouinpy as bp

        bp.analysis.segmented_fit(image, n_components=n_components)


@pytest.mark.parametrize("model", ["dho", "lorentzian"])
def test_segmented_fit_two_components_recovers_labels_and_shift(model):
    import brillouinpy as bp

    image, label_map, _ = additive_blob_image(
        nx=14, ny=14, n_channels=150, background_shift=5.6, component_shift=7.0,
        noise_model="poisson", peak_photons=400, edge="sharp", seed=1,
    )

    result = bp.analysis.segmented_fit(image, n_components=2, model=model)

    assert result.labels.shape == image.shape
    assert set(np.unique(result.labels)) <= {0, 1}
    # k-means keys off the added signal, so the segmentation should be essentially perfect.
    assert adjusted_rand_score(label_map.ravel(), result.labels.ravel()) > 0.9

    bg = result.shift[result.labels == 0]
    comp = result.shift[result.labels == 1]
    assert np.nanmedian(bg) == pytest.approx(5.6, abs=0.3)
    assert np.nanmedian(comp) == pytest.approx(7.0, abs=0.3)

    # stage diagnostics: one entry per stage, ascending component order.
    assert len(result.stage_shift) == 2
    assert result.stage_shift[0] == pytest.approx(5.6, abs=0.3)
    assert result.stage_shift[1] == pytest.approx(7.0, abs=0.3)


def test_segmented_fit_three_components_recovers_nucleus():
    import brillouinpy as bp

    image, label_map, true_params = three_component_image(
        nx=16, ny=16, n_channels=150, peak_photons=400, seed=3,
    )

    result = bp.analysis.segmented_fit(image, n_components=3)

    assert set(np.unique(result.labels)) <= {0, 1, 2}
    assert adjusted_rand_score(label_map.ravel(), result.labels.ravel()) > 0.9

    nucleus_shift = true_params["nucleus"][0]
    fitted_nucleus = result.shift[result.labels == 2]
    assert np.nanmedian(fitted_nucleus) == pytest.approx(nucleus_shift, abs=0.4)
    assert len(result.stage_shift) == 3


def test_segmented_fit_honours_custom_classifier():
    import brillouinpy as bp

    image, _, _ = additive_blob_image(nx=8, ny=8, n_channels=80, edge="sharp", seed=2)

    forced = np.zeros(image.shape, dtype=int)
    forced[:, 4:] = 1  # right half = "has the added component"
    calls = []

    def classifier(img, n_components):
        calls.append(n_components)
        return forced

    result = bp.analysis.segmented_fit(image, n_components=2, classifier=classifier)

    assert calls == [2]
    np.testing.assert_array_equal(result.labels, forced)


def test_segmented_fit_rejects_bad_classifier_argument():
    import brillouinpy as bp

    image, _, _ = additive_blob_image(nx=6, ny=6, n_channels=60)
    with pytest.raises(ValueError):
        bp.analysis.segmented_fit(image, n_components=2, classifier="spectral-clustering")


def test_segmented_fit_rejects_bad_model_argument():
    import brillouinpy as bp

    image, _, _ = additive_blob_image(nx=6, ny=6, n_channels=60)
    with pytest.raises(ValueError, match="model"):
        bp.analysis.segmented_fit(image, n_components=2, model="voigt")
