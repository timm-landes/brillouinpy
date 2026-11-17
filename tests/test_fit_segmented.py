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


def test_segmented_fit_p0_shifts_seed_the_new_peak_of_each_stage(monkeypatch):
    import brillouinpy as bp
    from brillouinpy.analysis.fit import segmented

    image, _, _ = additive_blob_image(nx=8, ny=8, n_channels=80, edge="sharp", seed=2)
    forced = np.zeros(image.shape, dtype=int)
    forced[:, 4:] = 1
    seen_p0 = []

    class RecordingDHO:
        """Stands in for core.DHO: records p0 and returns it as every pixel's fit."""

        def __init__(self, expected_peaks, p0, bounds, irf=None):
            seen_p0.append(list(p0))
            self.p0 = np.asarray(p0, dtype=float)

        def apply(self, img):
            return np.broadcast_to(self.p0, tuple(img.shape) + self.p0.shape).copy(), None

    monkeypatch.setattr(segmented.core, "DHO", RecordingDHO)
    result = bp.analysis.segmented_fit(image, n_components=2, classifier=lambda img, n: forced,
                                       shift_bounds=(3.0, 18.0), p0_shifts=[7.3, 15.0])

    assert len(seen_p0) == 2
    assert seen_p0[0][1] == pytest.approx(7.3)    # stage 1: the only peak
    assert seen_p0[1][4] == pytest.approx(15.0)   # stage 2: the new (second) peak
    assert seen_p0[1][1] == pytest.approx(7.3)    # stage 2: the anchored first peak
    assert result.stage_shift == pytest.approx([7.3, 15.0])


def test_segmented_fit_p0_shifts_none_entry_keeps_default_guess(monkeypatch):
    import brillouinpy as bp
    from brillouinpy.analysis.fit import segmented

    image, _, _ = additive_blob_image(nx=8, ny=8, n_channels=80, edge="sharp", seed=2)
    forced = np.zeros(image.shape, dtype=int)
    forced[:, 4:] = 1
    seen_p0 = []

    class RecordingDHO:
        def __init__(self, expected_peaks, p0, bounds, irf=None):
            seen_p0.append(list(p0))
            self.p0 = np.asarray(p0, dtype=float)

        def apply(self, img):
            return np.broadcast_to(self.p0, tuple(img.shape) + self.p0.shape).copy(), None

    monkeypatch.setattr(segmented.core, "DHO", RecordingDHO)
    bp.analysis.segmented_fit(image, n_components=2, classifier=lambda img, n: forced,
                              p0_shifts=[6.0, None])

    assert seen_p0[0][1] == pytest.approx(6.0)
    assert seen_p0[1][4] == pytest.approx(6.0 + 1.5)   # default: 1.5 GHz above the previous peak


def test_segmented_fit_p0_shifts_recovers_far_apart_component():
    import brillouinpy as bp

    # background ~7.5 GHz, weak component ~15 GHz: far outside the default
    # "previous peak + 1.5 GHz" seed, so it is supplied explicitly.
    image, label_map, _ = additive_blob_image(
        nx=12, ny=12, n_channels=300, background_shift=7.5, component_shift=15.0,
        component_amplitude=1.5e-3, noise_model="poisson", peak_photons=400, edge="sharp", seed=5,
    )

    result = bp.analysis.segmented_fit(image, n_components=2, shift_bounds=(5.0, 18.0),
                                       p0_shifts=[7.5, 15.0])

    assert adjusted_rand_score(label_map.ravel(), result.labels.ravel()) > 0.9
    comp = result.shift[result.labels == 1]
    assert np.nanmedian(comp) == pytest.approx(15.0, abs=0.3)
    assert np.mean(np.abs(comp - 15.0) < 1.0) > 0.95


def test_segmented_fit_passes_irf_to_every_stage_and_keeps_the_image_irf(monkeypatch):
    import brillouinpy as bp
    from brillouinpy.analysis.fit import segmented

    image, _, _ = additive_blob_image(nx=8, ny=8, n_channels=80, edge="sharp", seed=2)
    irf = np.exp(-0.5 * (np.asarray(image.spectral_axis) / 0.2) ** 2)
    image = bp.SpectralImage(image.spectral_data, image.spectral_axis, instrument_response_function=irf)
    forced = np.zeros(image.shape, dtype=int)
    forced[:, 4:] = 1
    seen = []

    class RecordingDHO:
        def __init__(self, expected_peaks, p0, bounds, irf=None):
            self.irf = irf
            self.p0 = np.asarray(p0, dtype=float)

        def apply(self, img):
            seen.append((self.irf, img.instrument_response_function))
            return np.broadcast_to(self.p0, tuple(img.shape) + self.p0.shape).copy(), None

    monkeypatch.setattr(segmented.core, "DHO", RecordingDHO)
    bp.analysis.segmented_fit(image, n_components=2, classifier=lambda img, n: forced, irf="auto")

    assert len(seen) == 2
    for passed, carried in seen:
        assert passed == "auto"
        np.testing.assert_array_equal(carried, irf)   # the per-class image still carries the IRF


def test_segmented_fit_irf_auto_recovers_intrinsic_linewidth():
    import brillouinpy as bp
    from brillouinpy.analysis.fit import core

    # intrinsic HWHM 0.3 (background) / 0.5 GHz (component), broadened by a
    # Lorentzian IRF of 0.4 GHz FWHM (adds 0.2 GHz to each HWHM)
    clean, label_map, _ = additive_blob_image(
        nx=8, ny=8, n_channels=250, background_shift=5.6, component_shift=7.2,
        background_linewidth=0.3, component_linewidth=0.5, noise=1e-5, edge="sharp", seed=4,
    )
    axis = np.asarray(clean.spectral_axis)
    kernel = core.irf_kernel(("lorentzian", 0.4), axis)
    data = np.apply_along_axis(lambda s: np.convolve(s, kernel, mode="same"), -1,
                               np.ma.filled(clean.spectral_data, np.nan))
    irf_sample = 1.0 / (1.0 + (axis / 0.2) ** 2)            # the IRF as the instrument would record it
    image = bp.SpectralImage(data, axis, instrument_response_function=irf_sample)
    def truth(img, n):                                       # isolate the fit from the classifier
        return label_map

    plain = bp.analysis.segmented_fit(image, 2, classifier=truth, shift_bounds=(4.0, 10.0))
    conv = bp.analysis.segmented_fit(image, 2, classifier=truth, shift_bounds=(4.0, 10.0), irf="auto")

    comp = label_map == 1
    assert np.nanmedian(conv.linewidth[comp]) == pytest.approx(0.5, abs=0.05)
    assert np.nanmedian(conv.linewidth[~comp]) == pytest.approx(0.3, abs=0.05)
    assert np.nanmedian(plain.linewidth[comp]) > 0.6            # without the IRF: broadened
    assert np.nanmedian(conv.shift[comp]) == pytest.approx(7.2, abs=0.05)


@pytest.mark.parametrize("p0_shifts", [[7.5], [7.5, 15.0, 20.0]])
def test_segmented_fit_rejects_wrong_p0_shifts_length(p0_shifts):
    import brillouinpy as bp

    image, _, _ = additive_blob_image(nx=6, ny=6, n_channels=60)
    with pytest.raises(ValueError, match="p0_shifts"):
        bp.analysis.segmented_fit(image, n_components=2, p0_shifts=p0_shifts)


def test_segmented_fit_rejects_bad_model_argument():
    import brillouinpy as bp

    image, _, _ = additive_blob_image(nx=6, ny=6, n_channels=60)
    with pytest.raises(ValueError, match="model"):
        bp.analysis.segmented_fit(image, n_components=2, model="voigt")
