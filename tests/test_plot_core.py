"""
Unit tests for brillouinpy.plot._core - the internal helpers behind
brillouinpy.plot.spectra / mean_spectra (RamanSPy-derived). They take a "genus"
(list of species), where a species is an (intensity_2d, shift_axis) tuple or a
bare array, and draw onto a given Axes.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from brillouinpy.plot import _core


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def axis():
    return np.linspace(-15, 15, 40)


def _species(axis, n_rows=3, scale=1.0):
    """One species: (intensity stack of shape (n_rows, B), shift axis)."""
    rng = np.random.default_rng(0)
    return (scale * (1.0 + rng.random((n_rows, axis.size))), axis)


# --------------------------------------------------------------------------- #
# plot_spectral_stack
# --------------------------------------------------------------------------- #

def test_plot_spectral_stack_with_axis_uses_it_as_x(axis):
    fig, ax = plt.subplots()
    stack = np.ones((3, axis.size))

    _core.plot_spectral_stack(stack, axis, plot_axis=ax)

    assert len(ax.lines) == 3
    np.testing.assert_allclose(ax.lines[0].get_xdata(), axis)


def test_plot_spectral_stack_without_axis_uses_indices(axis):
    fig, ax = plt.subplots()
    stack = np.ones((2, axis.size))

    _core.plot_spectral_stack(stack, None, plot_axis=ax)

    assert len(ax.lines) == 2
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.arange(axis.size))


# --------------------------------------------------------------------------- #
# plot_species
# --------------------------------------------------------------------------- #

def test_plot_species_accepts_tuple_and_bare_array(axis):
    fig, ax = plt.subplots()
    _core.plot_species((np.ones((2, axis.size)), axis), plot_axis=ax)
    _core.plot_species(np.ones((3, axis.size)), plot_axis=ax)

    assert len(ax.lines) == 5


def test_plot_species_offset_shifts_curves_up(axis):
    fig, ax = plt.subplots()
    stack = np.zeros((1, axis.size))

    _core.plot_species((stack, axis), plot_axis=ax, offset=7.0)

    np.testing.assert_allclose(ax.lines[0].get_ydata(), 7.0)


# --------------------------------------------------------------------------- #
# plot_species_mean
# --------------------------------------------------------------------------- #

def test_plot_species_mean_draws_confidence_band_when_dist(axis):
    fig, ax = plt.subplots()

    _core.plot_species_mean(_species(axis, n_rows=5), plot_axis=ax, dist=True)

    assert len(ax.lines) == 1          # the mean
    assert len(ax.collections) == 1    # fill_between band


def test_plot_species_mean_draws_all_spectra_when_not_dist(axis):
    fig, ax = plt.subplots()

    _core.plot_species_mean(_species(axis, n_rows=4), plot_axis=ax, dist=False)

    assert len(ax.collections) == 0
    assert len(ax.lines) == 4 + 1      # shaded spectra + mean


def test_plot_species_mean_single_spectrum_draws_only_mean(axis):
    fig, ax = plt.subplots()

    _core.plot_species_mean((np.ones((1, axis.size)), axis), plot_axis=ax, dist=True)

    assert len(ax.lines) == 1
    assert len(ax.collections) == 0


def test_plot_species_mean_is_nan_robust(axis):
    fig, ax = plt.subplots()
    stack = np.ones((4, axis.size))
    stack[0, :5] = np.nan

    _core.plot_species_mean((stack, axis), plot_axis=ax, dist=True)

    assert np.isfinite(ax.lines[0].get_ydata()).all()


@pytest.mark.xfail(reason="np.unique(shift_axes, axis=0) chokes on differing-length "
                           "shift axes instead of raising the intended ValueError - "
                           "see issue #10", strict=False)
def test_plot_species_mean_raises_for_differing_length_shift_axes(axis):
    fig, ax = plt.subplots()
    short_axis = axis[:-1]
    species = [_species(axis), (np.ones((2, short_axis.size)), short_axis)]

    with pytest.raises(ValueError):
        _core.plot_species_mean(species, plot_axis=ax, dist=True)


# --------------------------------------------------------------------------- #
# single_plot / offset_plot / stacked_plots
# --------------------------------------------------------------------------- #

def test_single_plot_returns_axes_and_sets_labels(axis):
    genus = [_species(axis), _species(axis)]

    ax = _core.single_plot(_core.plot_species, genus,
                           color=["C0", "C1"], label=["A", "B"],
                           title="T", xlabel="x", ylabel="y")

    assert ax.get_title() == "T"
    assert ax.get_xlabel() == "x"
    assert ax.get_legend() is not None


def test_single_plot_no_legend_without_labels(axis):
    genus = [_species(axis)]

    ax = _core.single_plot(_core.plot_species, genus, color=["C0"], label=[None])

    assert ax.get_legend() is None


def test_offset_plot_stacks_species_and_hides_yticks(axis):
    genus = [_species(axis, scale=1.0), _species(axis, scale=1.0)]

    ax = _core.offset_plot(_core.plot_species, genus,
                           color=["C0", "C1"], label=[None, None])

    assert list(ax.get_yticks()) == []
    # second species is drawn at a negative offset below the first
    assert ax.lines[-1].get_ydata().mean() < ax.lines[0].get_ydata().mean()


def test_stacked_plots_returns_figure_with_one_row_per_species(axis):
    genus = [_species(axis), _species(axis), _species(axis)]

    fig = _core.stacked_plots(_core.plot_species, genus,
                              color=[None] * 3, label=[None] * 3, title="Top")

    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) == 3


# --------------------------------------------------------------------------- #
# spectra_plot_wrapper - dispatch by plot_type
# --------------------------------------------------------------------------- #

@pytest.fixture
def genus(axis):
    return [_species(axis), _species(axis)]


def test_wrapper_single_returns_one_axes(genus):
    ax = _core.spectra_plot_wrapper(_core.plot_species, genus, "single", label=None)
    assert isinstance(ax, plt.Axes)


def test_wrapper_separate_returns_list_of_axes(genus):
    axs = _core.spectra_plot_wrapper(_core.plot_species, genus, "separate", label=None)
    assert isinstance(axs, list) and len(axs) == 2
    assert all(isinstance(a, plt.Axes) for a in axs)


def test_wrapper_separate_single_species_returns_bare_axes(axis):
    ax = _core.spectra_plot_wrapper(_core.plot_species, [_species(axis)], "separate", label=None)
    assert isinstance(ax, plt.Axes)


def test_wrapper_stacked_returns_figure(genus):
    fig = _core.spectra_plot_wrapper(_core.plot_species, genus, "stacked", label=None, yscale="linear")
    assert isinstance(fig, plt.Figure)
    assert all(ax.get_yscale() == "linear" for ax in fig.axes)


def test_wrapper_single_stacked_returns_axes(genus):
    ax = _core.spectra_plot_wrapper(_core.plot_species, genus, "single stacked", label=None, yscale="linear")
    assert isinstance(ax, plt.Axes)


def test_wrapper_scalar_label_is_broadcast(genus):
    # kwargs['label'] is a bare string, not a list - must not raise
    axs = _core.spectra_plot_wrapper(_core.plot_species, genus, "separate", label="shared")
    assert len(axs) == 2


def test_wrapper_default_color_gives_one_color_per_species(genus):
    ax = _core.spectra_plot_wrapper(_core.plot_species, genus, "single", label=None)
    colours = {tuple(np.round(line.get_color(), 5)) if not isinstance(line.get_color(), str)
               else line.get_color() for line in ax.lines}
    assert len(colours) >= 2


def test_wrapper_applies_yscale(genus):
    ax = _core.spectra_plot_wrapper(_core.plot_species, genus, "single", label=None, yscale="log")
    assert ax.get_yscale() == "log"
