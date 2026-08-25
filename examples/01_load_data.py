# -*- coding: utf-8 -*-
"""
Example 1 - Loading Brillouin data
===================================

Shows the two ways to get a :class:`brillouinpy.SpectralImage` (or
:class:`~brillouinpy.SpectralVolume`) to work with:

1. Loading a folder of measurement files from disk, with the interferometer scan
   parameters needed for the frequency axis read automatically from the
   measurement's ``META.json`` where available.
2. Loading a previously saved ``.pkl`` spectral object.

Run this script directly (``python 01_load_data.py``) - if no real project path is
set below, it falls back to a small synthetic dataset so the example still runs.
"""
import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import single_peak_image

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Option A: load real measurement data from disk
    # ------------------------------------------------------------------
    # Your data should live in a folder <project_path>/data, with filenames encoding
    # the (x, y, z, t) coordinates of each spectrum, e.g. "Bri_0_0_0_0.DAT".
    project_path = r'C:\Users\Timm\Desktop\Leipzig\Zygo_2_4'

    try:
        # 'load_spectral_image' returns a single (x, y, spectral) layer - use
        # 'prepare_brillouin_data' instead if you have multiple z-layers/timepoints.
        brillouin_data = bp.utils.load_spectral_image(project_path, 'Brillouin')

        # The Brillouin spectral (frequency-shift) axis is not stored in the raw data
        # files - it has to be computed from the interferometer's scan parameters.
        # If the measurement has a 'META.json' (directly in 'project_path' or in its
        # 'data' subfolder), those parameters can be read from it automatically:
        try:
            spectral_axis = bp.utils.brillouin_spectral_axis_from_meta(
                project_path, no_of_channels=brillouin_data.shape[-1],
            )
        except (FileNotFoundError, KeyError) as exc:
            # No (usable) META.json for this measurement - fall back to values
            # entered by hand. This is also how to proceed for measurements
            # taken before META.json was introduced.
            print(f"Could not read scan parameters from META.json ({exc}); using manual values.")
            spectral_axis = bp.utils.brillouin_spectral_axis(
                mirror_spacing=6e-3,     # [m]
                scan_amplitude=480e-9,   # [m]
                no_of_channels=brillouin_data.shape[-1],
            )

        brillouin_image = bp.SpectralImage(brillouin_data, spectral_axis)
        print(f"Loaded real data from '{project_path}': shape {brillouin_image.shape}")

        # 'META.json' can hold more than just the scan parameters used above - e.g.
        # the sample name, operator, and acquisition date. 'read_meta' gives you the
        # raw dict for anything else you want to pull out; attaching it to the
        # object keeps that information alongside the data for later reference
        # (e.g. when exporting via 'export.to_hdf5_bls'/'export.to_brim').
        try:
            brillouin_image.metadata = bp.utils.read_meta(project_path)
            # 'Sample' lives at the top level in the older flat META.json schema, or
            # under 'General' in the current nested one - check both.
            meta = brillouin_image.metadata
            sample = meta.get('Sample') or meta.get('General', {}).get('Sample', 'n/a')
            print(f"Sample: {sample}")
        except FileNotFoundError:
            pass

    except (FileNotFoundError, NotADirectoryError):
        # No data at 'project_path' on this machine - fall back to a synthetic
        # dataset so the rest of the example still has something to show.
        print(f"No data found at '{project_path}', using synthetic data instead.")
        brillouin_image = single_peak_image()

    # ------------------------------------------------------------------
    # Option B: load a spectral object saved earlier via '.save(...)'
    # ------------------------------------------------------------------
    # brillouin_image.save('brillouin_image.pkl', directory='pp_data')
    # brillouin_image = bp.SpectralImage.load('pp_data/brillouin_image.pkl')

    # A quick look at what was loaded
    bp.plot.mean_spectra(brillouin_image, title='Loaded Brillouin data', yscale='log')
    plt.show()
