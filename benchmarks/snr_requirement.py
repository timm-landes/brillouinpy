# -*- coding: utf-8 -*-
"""
What SNR does each method need to resolve a fixed shift difference?
==================================================================

Study script, not a tutorial example - see ``benchmarks/README.md``.

The mirror image of ``method_limitations.py``: here the Brillouin-shift
separation between the two hard-edged regions (``two_region_image``) is *fixed*
and the per-pixel noise is swept, so the x-axis is the per-pixel signal-to-noise
ratio (peak amplitude divided by the noise standard deviation, dimensionless).
Each method is scored against the ground-truth segmentation with the adjusted
Rand index.

Read the curves as "minimum SNR for usable segmentation": the SNR at which each
method's ARI rises off the chance-level floor. The phasor and the other
model-free methods are expected to hold on to lower SNR than a per-pixel
non-linear fit - the regime Elsayad (2019) singles out.

Note: the DHO fit runs a process pool per sweep point, so this takes a few
minutes at the default resolution. Shrink ``NOISE_LEVELS`` or ``NX``/``NY`` to
iterate faster.
"""
import matplotlib.pyplot as plt
import numpy as np

from _assay import run_sweep, make_figures, save_figures
from _synthetic_data import two_region_image

FREQ_SHIFT_A = 5.6                       # GHz
SHIFT_SEPARATION = 0.8                   # GHz, fixed - comfortably resolvable at high SNR
NOISE_MODEL = 'poisson'
PHOTON_LEVELS = np.geomspace(3, 1500, 14)   # counts at the region peak; ascending -> ascending SNR
TARGET_SNAPSHOT_SNR = 6.0               # the sweep point nearest this shot-noise SNR gets its maps drawn
NX = NY = 24


def run(save_dir=None):
    # Shot-noise SNR at a peak of N photons (on a small background) is ~ sqrt(N).
    snr_values = np.sqrt(PHOTON_LEVELS)
    snapshot_snr = float(snr_values[np.argmin(np.abs(snr_values - TARGET_SNAPSHOT_SNR))])

    cases = []
    for photons, snr in zip(PHOTON_LEVELS, snr_values):
        image, label_map, _ = two_region_image(
            nx=NX, ny=NY, freq_shift_a=FREQ_SHIFT_A, freq_shift_b=FREQ_SHIFT_A + SHIFT_SEPARATION,
            noise_model=NOISE_MODEL, peak_photons=photons, geometry='blob', edge='sharp',
        )
        cases.append((snr, image, label_map))

    result = run_sweep(cases, snapshot_value=snapshot_snr)
    figs = make_figures(
        result,
        xlabel='Shot-noise SNR at the region peak (≈ sqrt of peak photon count)\n'
               'swept low → high; read the threshold as the minimum SNR for usable segmentation',
        title=f'Segmentation accuracy vs. shot-noise SNR\n(region shift separation fixed at {SHIFT_SEPARATION} GHz)',
        snapshot_value=snapshot_snr,
        snapshot_label=f'SNR ≈ {snapshot_snr:.1f}',
        xscale='log',
    )
    if save_dir:
        save_figures(figs, save_dir, 'snr_requirement')
    return result


if __name__ == '__main__':
    run()
    plt.show()
