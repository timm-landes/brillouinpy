# Writing your own loader

Everything in `brillouinpy.io.tfp`/`io.legacy`/`io.multimodal` (see
[Loading data](loading.md)) is specific to this group's raw-data layout -
filenames encoding `(x, y, z, t)` coordinates, a particular `META.json` schema,
`.DAT`/`.csv`/`.txt` file formats. If your own setup writes data differently, you
don't need to reshape it into that layout; you only need a function that produces
the two things every `brillouinpy` object needs:

1. an intensity array, shaped `(..., B)` - any number of leading spatial
   dimensions (`x`, `x, y`, `x, y, z`, ...), then a trailing spectral axis of
   length `B`; and
2. a matching 1D spectral axis of length `B` (frequency shift, wavelength, or
   whatever your spectral axis represents).

Wrap those two into a `Spectrum`/`SpectralImage`/`SpectralVolume` (`brillouinpy`
picks the right one for you via `core._create_data`, or you can construct one
directly, e.g. `bp.SpectralImage(intensity_array, spectral_axis)`) and every
preprocessing/analysis/export step in this tutorial works on it unchanged - none
of them care how the data was loaded.

A minimal skeleton, modelled on `io.tfp.prepare_brillouin_data`:

```python
import numpy as np
import brillouinpy as bp


def load_my_setup_data(project_path):
    """
    Replace the body below with whatever it takes to get from your raw files
    to (intensity_array, spectral_axis). The rest of this function is generic.
    """
    # ... read your files, build the (x, y, ..., B)-shaped array and the
    # length-B spectral axis however your setup requires ...
    intensity_array = ...   # numpy.ndarray, shape (..., B)
    spectral_axis = ...     # numpy.ndarray, shape (B,)

    return bp.core._create_data(intensity_array, spectral_axis)


my_data = load_my_setup_data(project_path)
```

A few things worth carrying over from `io.tfp.prepare_brillouin_data`/
`io.multimodal.prepare_raman_data`, even though they're not strictly required:

- **Missing points as a masked array, not a crash.** If some spectra fail to load
  or are missing entirely (a common occurrence in large scans), initialise the
  array with `numpy.ma.masked_all(shape)` and leave the corresponding entries
  masked instead of raising - `brillouinpy`'s preprocessing/analysis steps are
  masked-array/NaN-aware throughout (see [Preprocessing](preprocessing.md))
  specifically so a few missing points don't take down an entire pipeline run.
- **Warn, don't silently drop, on a mismatch.** `warnings.warn(...)` when the
  number of files found doesn't match the expected number of points - it's much
  easier to notice a printed warning while a `tqdm` progress bar is running than
  to later wonder why an image has an odd blank region.
- **Read your axis from metadata if you have it**, rather than hard-coding
  instrument parameters into the loader itself (see `io.tfp.read_meta`/
  `brillouin_spectral_axis_from_meta` for the pattern) - keeps the loader
  reusable across measurements taken with different settings.

Where you put the finished function is up to you: for a one-off analysis, a
local script is fine; if you expect to reuse it, a module of your own (e.g.
`my_lab_io.py`) that you `import` alongside `brillouinpy` works just as well as
contributing it back to `brillouinpy.io` - the package doesn't need to know
about your setup for any of this to work.
