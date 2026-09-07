# -*- coding: utf-8 -*-
"""
How close in Brillouin shift can two regions be before each method fails?
=======================================================================

Study script, not a tutorial example - see ``benchmarks/README.md``.

Sweeps the Brillouin-shift separation between two hard-edged regions
(``two_region_image``: every pixel one pure spectrum) at a fixed noise level and
scores how well each method (see :mod:`_assay`) recovers the ground-truth
segmentation, with the adjusted Rand index (ARI: 1 = perfect, 0 = chance).

Companion scripts: ``snr_requirement.py`` (fixes the separation, sweeps SNR) and
``additive_blob.py`` (constant background everywhere + an added component in a
blob, instead of two pure domains).

What the default settings tend to show:

- Nobody separates the regions once the shift difference nears the per-pixel
  noise floor (leftmost point).
- The bounded DHO fit is the most shift-sensitive - the correct model wins when
  it is well constrained.
- The phasor tracks raw-spectrum k-means: a little less sensitive than the fit
  here, but parameter-free and (timing panel) far cheaper. Its documented edge
  over fitting is the shot-noise-limited, unknown-lineshape regime of Elsayad
  (2019) - push into it with ``snr_requirement.py``.

Note: the DHO fit runs a process pool per sweep point, so this takes a minute or
so. Shrink ``SHIFT_SEPARATIONS`` or ``NX``/``NY`` to iterate faster.
"""
import matplotlib.pyplot as plt
import numpy as np

from _assay import run_sweep, make_figures, save_figures
from _synthetic_data import two_region_image

FREQ_SHIFT_A = 5.6           # GHz, fixed
SHIFT_SEPARATIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5])  # GHz
SNAPSHOT_SEPARATION = 0.5    # GHz; the separation whose maps get drawn
NOISE_MODEL = 'poisson'
PEAK_PHOTONS = 60            # counts at the region peak -> shot-noise SNR ~ sqrt(60) ~ 8
NX = NY = 24


def run(save_dir=None):
    cases = []
    for separation in SHIFT_SEPARATIONS:
        image, label_map, _ = two_region_image(
            nx=NX, ny=NY, freq_shift_a=FREQ_SHIFT_A, freq_shift_b=FREQ_SHIFT_A + separation,
            noise_model=NOISE_MODEL, peak_photons=PEAK_PHOTONS, geometry='blob', edge='sharp',
        )
        cases.append((separation, image, label_map))

    result = run_sweep(cases, snapshot_value=SNAPSHOT_SEPARATION)
    figs = make_figures(
        result,
        xlabel='Brillouin-shift separation between the two regions (GHz)',
        title=f'Segmentation accuracy vs. shift separation\n'
              f'({NOISE_MODEL} noise, peak ≈ {PEAK_PHOTONS:g} photons, shot-noise SNR ≈ {PEAK_PHOTONS ** 0.5:.0f})',
        snapshot_value=SNAPSHOT_SEPARATION,
        snapshot_label=f'delta = {SNAPSHOT_SEPARATION} GHz '
                       f'(regions at {FREQ_SHIFT_A} and {FREQ_SHIFT_A + SNAPSHOT_SEPARATION} GHz)',
    )
    if save_dir:
        save_figures(figs, save_dir, 'method_limitations')
    return result


if __name__ == '__main__':
    run()
    plt.show()
