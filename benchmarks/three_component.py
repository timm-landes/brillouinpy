# -*- coding: utf-8 -*-
"""
Three additive components: medium, cytoplasm, nucleus (experiment D)
=====================================================================

Study script, not a tutorial example - see ``benchmarks/README.md``.

``additive_blob.py`` (experiment C) has one background component and one added
component. This is the natural next step toward a real measurement: a cell
sitting in a hydrated medium, imaged with a confocal volume that in practice
never isolates a single compartment cleanly. ``three_component_image`` models a
**constant medium (background) peak everywhere**, a **cytoplasm** peak added on
top inside a "cell" disc, and a **nucleus** peak added on top of *both* inside a
smaller, concentric disc - so nucleus pixels carry all three components
additively, cytoplasm pixels carry two, background pixels carry one. All three
components have distinct linewidths as well as shifts, so this experiment scores
**shift and linewidth recovery for all three components**, not just the swept
one.

The sweep varies the nucleus shift (its separation from the fixed cytoplasm
peak); background and cytoplasm shift/linewidth stay fixed throughout, so their
recovery curves are a *reference level* the nucleus curve can be read against.

Given what ``additive_blob.py`` already establishes at length (see its docstring:
a raw multi-peak DHO fit run everywhere is unstable, and no post-hoc filter - by
amplitude, by a robust threshold, or by fit covariance - rescues it), this
experiment does not repeat those failure modes for three peaks (which would only
be worse). It compares:

- **The recommended hierarchical workflow** (``DHO fit (segmented, 3-comp)``):
  classify background/cytoplasm/nucleus first with a cheap, model-free
  classifier (k-means on the raw spectra, ordered by total intensity - valid
  here because every extra component only adds signal), then fit exactly the
  right number of DHO peaks in each class (one/two/three), anchoring
  already-identified peaks so each fit only ever resolves the *one* new
  component its class actually contains.
- **Multi-class versions of the same model-free classifiers** as experiments A-C
  (k-means, PCA, NMF, VCA, spectral phasor, background-subtracted phasor), scored
  purely on 3-way segmentation ARI - they expose no physical shift/linewidth, so
  they do not appear in the recovery figures.

Note: three DHO fits (1-peak, 2-peak, 3-peak, each restricted to its own class)
run per sweep point; shrink ``NUCLEUS_SHIFTS`` or ``NX``/``NY`` to iterate faster.
"""
import matplotlib.pyplot as plt
import numpy as np

from _assay import METHODS_D, run_sweep_d, make_figures_d, make_recovery_figure_d, save_figures
from _synthetic_data import three_component_image

BACKGROUND_SHIFT, BACKGROUND_LINEWIDTH = 5.6, 0.8    # GHz - the everywhere medium peak
CYTOPLASM_SHIFT, CYTOPLASM_LINEWIDTH = 6.3, 0.6      # GHz - fixed throughout the sweep
NUCLEUS_LINEWIDTH = 0.4                              # GHz - fixed; only the nucleus shift is swept
NUCLEUS_SHIFTS = np.array([6.5, 6.7, 6.9, 7.1, 7.4, 7.7, 8.0, 8.5, 9.0])  # GHz
SNAPSHOT_SHIFT = 7.4  # GHz

BACKGROUND_AMPLITUDE, CYTOPLASM_AMPLITUDE, NUCLEUS_AMPLITUDE = 5e-3, 4e-3, 3.5e-3
NOISE_MODEL = 'poisson'
PEAK_PHOTONS = 200  # counts at the background peak
NX = NY = 24


def run(save_dir=None):
    cases = []
    for nucleus_shift in NUCLEUS_SHIFTS:
        image, label_map, _ = three_component_image(
            nx=NX, ny=NY,
            background_shift=BACKGROUND_SHIFT, cytoplasm_shift=CYTOPLASM_SHIFT, nucleus_shift=nucleus_shift,
            background_linewidth=BACKGROUND_LINEWIDTH, cytoplasm_linewidth=CYTOPLASM_LINEWIDTH,
            nucleus_linewidth=NUCLEUS_LINEWIDTH,
            background_amplitude=BACKGROUND_AMPLITUDE, cytoplasm_amplitude=CYTOPLASM_AMPLITUDE,
            nucleus_amplitude=NUCLEUS_AMPLITUDE, noise_model=NOISE_MODEL, peak_photons=PEAK_PHOTONS,
        )
        cases.append((nucleus_shift, image, label_map))

    result = run_sweep_d(
        cases, snapshot_value=SNAPSHOT_SHIFT, methods=METHODS_D,
        background_truth=(BACKGROUND_SHIFT, BACKGROUND_LINEWIDTH),
        cytoplasm_truth=(CYTOPLASM_SHIFT, CYTOPLASM_LINEWIDTH),
        nucleus_linewidth=NUCLEUS_LINEWIDTH,
    )

    common = (f'(medium {BACKGROUND_SHIFT}/{BACKGROUND_LINEWIDTH} GHz; cytoplasm '
             f'{CYTOPLASM_SHIFT}/{CYTOPLASM_LINEWIDTH} GHz; nucleus linewidth {NUCLEUS_LINEWIDTH} GHz; '
             f'{NOISE_MODEL}, medium peak ≈ {PEAK_PHOTONS:g} photons)')
    xlabel = 'True nucleus Brillouin shift (GHz)\n(cytoplasm shift fixed at {:g} GHz)'.format(CYTOPLASM_SHIFT)

    recovery_figs, recovery_suffixes = [], []
    for component in ('background', 'cytoplasm', 'nucleus'):
        for quantity in ('shift', 'width'):
            recovery_figs.append(make_recovery_figure_d(
                result, xlabel=xlabel, component=component, quantity=quantity,
                title=f'{component.capitalize()} {quantity} recovery accuracy (primary metric)\n{common}',
            ))
            recovery_suffixes.append(f'{component}_{quantity}_recovery')

    figs = make_figures_d(
        result, xlabel=xlabel,
        title=f'Segmentation of medium/cytoplasm/nucleus (secondary metric)\n{common}',
        snapshot_value=SNAPSHOT_SHIFT,
        snapshot_label=f'nucleus at {SNAPSHOT_SHIFT} GHz (cytoplasm fixed at {CYTOPLASM_SHIFT} GHz)',
    )
    if save_dir:
        save_figures(tuple(recovery_figs), save_dir, 'three_component', suffixes=recovery_suffixes)
        save_figures(figs, save_dir, 'three_component')
    return result


if __name__ == '__main__':
    run()
    plt.show()
