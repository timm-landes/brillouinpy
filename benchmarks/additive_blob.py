# -*- coding: utf-8 -*-
"""
Additive mixture: constant background + a shifted component in a blob
====================================================================

Study script, not a tutorial example - see ``benchmarks/README.md``.

``method_limitations.py`` and ``snr_requirement.py`` use two hard-separated pure
domains. Real samples are rarely like that: a hydrated specimen carries a water /
hydration Brillouin signal in *every* pixel, and an inclusion adds its own peak
on top. ``additive_blob_image`` models exactly that - one constant background
spectrum everywhere, plus a second DHO component added inside a blob, so **no
pixel is a pure component spectrum**.

**Primary question: how accurately does each method recover the added
component's true Brillouin shift *and linewidth*** (``shift_recovery=True``,
``true_width=COMPONENT_LINEWIDTH`` in :func:`_assay.run_sweep` - median
|fitted - true| over the true sample pixels, independent of segmentation
quality; the added component's linewidth is set distinct from the
background's so this is a real reconstruction question, not a giveaway).
Whether a method can also *find* the blob (adjusted Rand index, as in the
other two scripts) is the secondary question here - most of them can, for
different reasons, so ARI alone would flatten exactly the distinction this
script exists to show.

Three ways to get a component-shift estimate out of a field like this are
compared, motivated by a concrete question: **if this field genuinely needs two
DHO peaks somewhere, how do you actually fit that?**

- **Naive - one peak everywhere** (``DHO fit (1-peak)``): sees background +
  component blended into one asymmetric peak; its single fitted shift is a
  biased blend of the two, not the true component shift.
- **Wrong way - two peaks everywhere, raw** (``DHO fit (2-peak, raw)``): the
  right model for the blob pixels, run without first asking *where* they are.
  Background-only pixels make the second peak rank-deficient (its
  amplitude -> 0, so its shift/width stop mattering) - the fit fails outright on
  many of them and puts a tall narrow spike on a noise excursion on others. A
  common instinct is to then mask out pixels by fitted amplitude after the
  fact (``DHO fit (2-peak + amplitude mask)``); shown here too, and it does not
  really fix things - see its interpretation below.
- **Recommended - segment first, then fit** (``DHO fit (segmented)``): use a
  cheap, model-free classifier (k-means on the raw spectra - already
  near-perfect here) to split background-only from sample pixels first, *then*
  fit one DHO in the former and two in the latter. The second peak is only ever
  asked for where it is actually well-posed.

Two more variants avoid a multi-peak fit entirely: subtracting the estimated
common background and reading off a single residual peak (``DHO fit
(bg-subtracted)``, ``Spectral phasor (bg-subtracted)``), or a plain
segmentation method's continuous feature where it happens to correlate with
shift (``Spectral phasor``, ``PCA``'s PC1 - not really shift estimators, shown
for the segmentation figure only).

Note: several DHO fits now run per sweep point (each a process pool); shrink
``COMPONENT_SHIFTS`` or ``NX``/``NY`` to iterate faster.
"""
import matplotlib.pyplot as plt
import numpy as np

from _assay import (METHODS, dho_bgsub_method, phasor_bgsub_method, dho_fit_segmented,
                    make_dho_fit_2peak_method, make_dho_fit_2peak_masked_method,
                    run_sweep, make_figures, make_recovery_figure, save_figures)
from _synthetic_data import additive_blob_image

BACKGROUND_SHIFT = 5.6       # GHz - the everywhere "hydration" peak
BACKGROUND_LINEWIDTH = 0.8   # GHz
COMPONENT_LINEWIDTH = 0.5    # GHz - deliberately distinct from the background's, so
                             # linewidth (not just shift) recovery is a meaningful question

# Five DHO-family variants (see the module docstring for what each one is), plus
# the shared model-free methods (minus the base 'DHO fit (1-peak)', re-added
# explicitly first so it stays at the front of the dict/plots).
METHODS_ADDITIVE = {'DHO fit (1-peak)': METHODS['DHO fit (1-peak)'],
                    'DHO fit (2-peak, raw)': make_dho_fit_2peak_method(BACKGROUND_SHIFT),
                    'DHO fit (2-peak + amplitude mask)': make_dho_fit_2peak_masked_method(BACKGROUND_SHIFT),
                    'DHO fit (segmented)': dho_fit_segmented,
                    'DHO fit (bg-subtracted)': dho_bgsub_method,
                    'Spectral phasor (bg-subtracted)': phasor_bgsub_method,
                    **{k: v for k, v in METHODS.items() if k != 'DHO fit (1-peak)'}}

COMPONENT_SHIFTS = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.3, 6.5, 6.75, 7.0, 7.5, 8.0])  # GHz
COMPONENT_AMPLITUDE = 4e-3  # background peak amplitude is 5e-3
SNAPSHOT_SHIFT = 6.15       # GHz
NOISE_MODEL = 'poisson'
PEAK_PHOTONS = 200          # counts at the background peak
NX = NY = 24


def run(save_dir=None):
    cases = []
    for component_shift in COMPONENT_SHIFTS:
        image, label_map, _ = additive_blob_image(
            nx=NX, ny=NY, background_shift=BACKGROUND_SHIFT, component_shift=component_shift,
            background_linewidth=BACKGROUND_LINEWIDTH, component_linewidth=COMPONENT_LINEWIDTH,
            component_amplitude=COMPONENT_AMPLITUDE, noise_model=NOISE_MODEL,
            peak_photons=PEAK_PHOTONS, geometry='blob', edge='sharp',
        )
        cases.append((component_shift, image, label_map))

    result = run_sweep(cases, snapshot_value=SNAPSHOT_SHIFT, methods=METHODS_ADDITIVE,
                       shift_recovery=True, true_width=COMPONENT_LINEWIDTH)

    common_title_bits = (f'(bg {BACKGROUND_SHIFT} GHz/{BACKGROUND_LINEWIDTH} GHz everywhere; added amp '
                         f'{COMPONENT_AMPLITUDE:g} vs bg 5e-3, added linewidth {COMPONENT_LINEWIDTH} GHz; '
                         f'{NOISE_MODEL}, bg peak ≈ {PEAK_PHOTONS:g} photons)')
    shift_recovery_fig = make_recovery_figure(
        result, key='recovery',
        xlabel=f'True Brillouin shift of the added component (GHz)\n(constant background peak fixed at {BACKGROUND_SHIFT} GHz)',
        title=f'Component-shift recovery accuracy (primary metric)\n{common_title_bits}',
    )
    width_recovery_fig = make_recovery_figure(
        result, key='width_recovery',
        xlabel=f'True Brillouin shift of the added component (GHz)\n(constant background peak fixed at {BACKGROUND_SHIFT} GHz)',
        title=f'Component-linewidth recovery accuracy (primary metric)\n{common_title_bits}',
        ylabel=f'Median |fitted linewidth - true linewidth ({COMPONENT_LINEWIDTH} GHz)| (GHz)\n'
              'over true sample pixels, per sweep point',
    )
    figs = make_figures(
        result,
        xlabel=f'Brillouin shift of the added component (GHz)\n(constant background peak fixed at {BACKGROUND_SHIFT} GHz)',
        title=f'Segmentation of an additive blob (secondary metric)\n{common_title_bits}',
        snapshot_value=SNAPSHOT_SHIFT,
        snapshot_label=f'component at {SNAPSHOT_SHIFT} GHz on a {BACKGROUND_SHIFT} GHz background',
    )
    if save_dir:
        save_figures((shift_recovery_fig, width_recovery_fig), save_dir, 'additive_blob',
                     suffixes=['shift_recovery', 'width_recovery'])
        save_figures(figs, save_dir, 'additive_blob')
    return result


if __name__ == '__main__':
    run()
    plt.show()
