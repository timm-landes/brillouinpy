# BrillouinPy

BrillouinPy is a Python package for analysing Brillouin light scattering (BLS)
microscopy and imaging data - from raw spectra through instrument-response
deconvolution and damped-harmonic-oscillator (DHO) fitting to multivariate
analysis.

Its data model and pipeline architecture originate as a fork of
[RamanSPy](https://github.com/barahona-research-group/RamanSPy). BrillouinPy keeps
the parts of that architecture that generalise well - spectral containers,
composable preprocessing pipelines, decomposition/clustering workflows - and
replaces or extends the rest with Brillouin-specific pieces: masked-array/NaN-aware
pipelines, elastic-peak (IRF/Rayleigh) removal, and Lorentzian/DHO peak fitting.
See [`NOTICE.md`](https://github.com/timm-landes/brillouinpy/blob/main/NOTICE.md)
for a file-by-file breakdown of what was adapted and what changed.

- **Source:** [github.com/timm-landes/brillouinpy](https://github.com/timm-landes/brillouinpy)
  (a mirror on the Leibniz University Hannover GitLab is kept for internal
  development, reachable only from within the LUH network).
- **License:** BSD 3-Clause.

## Quickstart

```python
import numpy as np
import matplotlib.pyplot as plt

import brillouinpy as bp

if __name__ == '__main__':
    project_path = r'complete_path_to_your_project'

    # Load every spectrum under <project_path>/data; .squeeze() drops the
    # singleton z / timepoint axes of a plain 2D scan.
    intensity = bp.io.prepare_brillouin_data(project_path, 'Brillouin').squeeze()

    # Frequency axis from the interferometer scan parameters (read from META.json
    # if present, otherwise supplied by hand - these must match your setup).
    try:
        spectral_axis = bp.io.brillouin_spectral_axis_from_meta(
            project_path, no_of_channels=intensity.shape[-1],
        )
    except (FileNotFoundError, KeyError):
        spectral_axis = bp.utils.brillouin_spectral_axis(
            mirror_spacing=3e-3, scan_amplitude=309e-9,
            no_of_channels=intensity.shape[-1],
        )

    brillouin_data = bp.SpectralImage(intensity, spectral_axis)

    pipeline = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
        bp.preprocessing.misc.Deconvoluter_IRF(offset=65, iterations=4, padding=None),
        bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
    ])
    pp_brillouin_data = pipeline.apply(brillouin_data)

    model = bp.analysis.fit.DHO(
        expected_peaks=1,
        p0=[20, 8, 0.5, 0, 0],  # [amplitude, shift (GHz), HWHM (GHz), background, axis_shift]
        bounds=([0, 5, 0.1, -5, -5], [100, 15, 10, 5, 5]),
    )
    fit_result, metrics = model.apply(pp_brillouin_data)
    print(f"mean Brillouin frequency shift: {np.nanmean(fit_result[..., 1]):.3f} GHz")
```

The [Tutorial](tutorial/index.md) walks through this end to end, one stage per
page, with figures. See [Installation](installation.md) to get set up.

```{toctree}
:maxdepth: 2
:hidden:

installation
tutorial/index
api/index
```
