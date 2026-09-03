# Loading data

*Script: `examples/01_load_data.py`*

Everything starts from a folder of raw measurement files (`.DAT` for Brillouin,
`.csv`/`.txt` for Raman) whose filenames encode the `(x, y, z, t)` coordinates of
each spectrum, e.g. `Bri_0_0_0_0.DAT`.

Loading itself lives under `brillouinpy.io`: `io.tfp` for this group's actively
maintained tandem Fabry-Perot (TFP) loading path, `io.legacy` for older, superseded
loaders (kept for reading old datasets), and `io.multimodal` for other modalities
(currently Raman). `io.export` (also reachable as `io.<function>` for the most
commonly used functions, as used below) covers the other direction: writing to and
reading from external formats. See [Writing your own loader](custom-loader.md) if
your own setup uses a different raw-data layout.

```python
import brillouinpy as bp

# A single (x, y, spectral) layer. Use io.tfp.prepare_brillouin_data(...) instead
# for a full (x, y, z, t, spectral) volume with multiple z-layers/timepoints.
brillouin_data = bp.io.legacy.load_spectral_image(project_path, 'Brillouin')

# The frequency-shift axis isn't stored in the raw files - it's derived from the
# tandem Fabry-Perot interferometer's scan parameters. If the measurement has a
# 'META.json' (directly in 'project_path' or in its 'data' subfolder), those
# parameters can be read from it automatically:
spectral_axis = bp.io.brillouin_spectral_axis_from_meta(
    project_path, no_of_channels=brillouin_data.shape[-1],
)

brillouin_image = bp.SpectralImage(brillouin_data, spectral_axis)
```

`brillouin_spectral_axis_from_meta` transparently handles the different
`META.json` schema versions found across existing datasets (a current, nested
`Brillouin`/`Laser` section, and an older flat one) - see
`io.tfp._brillouin_scan_parameters_from_meta` if you need to know exactly which
keys it looks for. It raises `FileNotFoundError` if no `META.json` exists for the
measurement, and `KeyError` if the one found doesn't contain a required field
(e.g. some older flat-schema files are missing the scan amplitude entirely) - in
either case, fall back to calling `brillouin_spectral_axis` directly with values
entered by hand:

```python
spectral_axis = bp.utils.brillouin_spectral_axis(
    mirror_spacing=6e-3,     # [m]
    scan_amplitude=480e-9,   # [m]
    no_of_channels=brillouin_data.shape[-1],
)
```

`META.json` usually holds more than just the scan parameters - sample name,
operator, acquisition date, and so on. `io.read_meta(project_path)` gives you
that raw dict directly, for anything beyond what
`brillouin_spectral_axis_from_meta` already extracts:

```python
brillouin_image.metadata = bp.io.read_meta(project_path)
```

Attaching it to `.metadata` keeps it alongside the data for later reference (e.g.
`io.to_hdf5_bls`/`io.to_brim`, see [Exporting results](exporting.md), both accept
a `sample` argument you could pull from here). Where the value actually lives
depends on the schema version - e.g. `meta['Sample']` (flat) vs.
`meta['General']['Sample']` (nested) - the same distinction
`brillouin_spectral_axis_from_meta` handles for the scan parameters.

`mirror_spacing` and `scan_amplitude` are properties of *your* interferometer setup,
not of the sample - get them from your instrument's calibration (via `META.json` or
by hand), not from the values shown here. Getting them wrong stretches or
compresses the whole frequency axis, so if your fitted frequency shifts look
consistently off by a constant factor, this is the first place to check.

The resulting `SpectralImage` (see also `Spectrum`, `SpectralVolume` and the common
base class `SpectralContainer` in the API reference) is what every other step in
this tutorial operates on:

```{image} /_static/tutorial/01_loaded_data.png
:alt: Mean spectrum (with 95% confidence band) of a loaded Brillouin image, on a log intensity axis, showing the symmetric Stokes/anti-Stokes doublet with a broad, noisy floor between the peaks.
:width: 480px
:align: center
```

This is `bp.plot.mean_spectra(brillouin_image, yscale='log')`: the solid line is the
mean spectrum across every pixel in the image, the shaded band its 95% confidence
interval. Plotting the *mean*, rather than a single pixel, is usually the fastest
way to sanity-check a newly loaded dataset - it averages out per-pixel noise while
still showing whether the peaks are where you expect them, and how noisy/flat the
background between them is.

`SpectralImage` can be persisted at any point with `.save(filename, directory=...)`
and read back with `SpectralImage.load(path)` - handy for checkpointing between the
preprocessing-heavy steps and the analysis steps, so you don't have to re-run
expensive steps (like fitting, see [Fitting spectra](fitting.md)) while iterating
on a plot.
