# Preprocessing

*Scripts: `examples/02_remove_irf.py`, `examples/03_preprocess_data.py`*

## Removing the Instrument Response Function

Real Brillouin spectra contain a strong, narrow Instrument Response Function
(IRF)/Rayleigh line - the elastically scattered light, sitting at zero frequency
shift - in addition to the much weaker Brillouin peak(s) of interest. This should
happen right after loading, before the rest of preprocessing: a
`MaxIntensity` normalisation run beforehand would just rescale everything relative
to the IRF's peak height rather than the signal you actually care about.

```{image} /_static/tutorial/02_irf_removal.png
:alt: Two single-spectrum plots side by side. Left, a raw spectrum dominated by a tall, narrow central IRF peak dwarfing the much smaller Brillouin doublet either side of it. Right, the same spectrum after IRF removal, showing a flat NaN gap where the IRF was and the Brillouin doublet now clearly visible.
:width: 720px
:align: center
```

`preprocessing.misc.Deconvoluter_IRF` finds the IRF by starting at the spectrum's
global maximum and scanning outwards for the first exact-zero-intensity channel on
each side (the real detector baseline around a saturated/blanked IRF region), uses
it as the point-spread function to sharpen the rest of the spectrum via
Richardson-Lucy deconvolution, and only then blanks that region out (set to `NaN`):

```python
deconvoluter = bp.preprocessing.misc.Deconvoluter_IRF(offset=3, iterations=4, padding=20)
cleaned_image = deconvoluter.apply(brillouin_image)
```

`offset` extends the blanked region by that many extra channels on each side, e.g.
to also catch stray Rayleigh-scattered light bleeding past the IRF's zero-baseline
edges - real data with more stray light may need a much larger value here (e.g.
`65`) than this synthetic example does. `iterations` controls how many
Richardson-Lucy iterations run; it isn't something to change without reason.
`padding` reflect-pads the spectrum by that many channels on each side before
deconvolution, then crops back to the original length afterwards - Richardson-Lucy
otherwise treats the array edges as a hard boundary, which can produce a spurious
up/down swing near the first/last few channels; a value comparable to a few times
the IRF's width is a reasonable start. `None`/`0` disables it.
Because this relies on finding genuine zero-intensity channels, always check a raw,
unprocessed spectrum first (as above) to confirm your data has a comparable zero
baseline around the IRF before relying on this step - if the scan never finds a
zero channel, it silently falls back to the start/end of the whole spectrum, which
would blank out everything.

The removed region is set to `NaN` rather than being cropped out of the spectral
axis, so `cleaned_image` keeps the exact same shape as `brillouin_image` -
downstream steps see a gap, not a shorter spectrum. Downstream steps must be
prepared to handle the `NaN`s it leaves behind - the built-in fitting and analysis
steps already do (they drop NaN-containing spectral channels automatically), but a
custom preprocessing step you write yourself would need to handle them explicitly.
It works on any spectral object - `Spectrum`, `SpectralImage`, `SpectralVolume`, or
the raw `(x, y, z, t, spectral)` shape returned by `io.tfp.prepare_brillouin_data` -
looping over whatever spatial dimensions are present.

For a simpler alternative that just crops the IRF away (blanked to `0` rather than
`NaN`) without the deconvolution step, see `preprocessing.misc.IRF_Remover` - same
interface, just `IRF_Remover(offset=3)`.

## Building a preprocessing pipeline

Preprocessing steps are small, composable objects (subclasses of
`preprocessing.Step.PreprocessingStep`) chained together in a
`preprocessing.Pipeline`, applied in the order given:

```python
pipeline = bp.preprocessing.Pipeline([
    bp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=8),
    bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
    bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
])

preprocessed_image = pipeline.apply(cleaned_image)
```

A pipeline can be applied as a whole, or any single step can be applied on its own
via `step.apply(spectral_object)` - useful while tuning parameters interactively,
e.g. to check how aggressive a given `threshold`/`window_length` is on a single
pixel before committing to it for the whole dataset.

```{image} /_static/tutorial/03_preprocessing.png
:alt: Two mean-spectrum plots side by side, raw data on a log axis on the left, despiked/denoised/normalised data on a linear axis on the right, both showing a symmetric Stokes/anti-Stokes doublet; the right plot is visibly smoother.
:width: 720px
:align: center
```

The building blocks available are:

| Module | Steps | Purpose |
|---|---|---|
| `preprocessing.despike` | `WhitakerHayes` | Cosmic-ray spike removal |
| `preprocessing.denoise` | `SavGol`, `Gaussian`, `Whittaker`, `Kernel` | Smoothing |
| `preprocessing.misc` | `Cropper`, `BackgroundSubtractor`, `IRF_Remover`, `Deconvoluter_IRF` | Spectral-region and IRF handling (IRF removal covered above) |
| `preprocessing.normalise` | `Vector`, `MinMax`, `MaxIntensity`, `AUC` | Intensity normalisation |

A few things worth knowing about the order steps run in:

- **Despike before denoise.** A smoothing filter (Savitzky-Golay, Gaussian, ...)
  averages neighbouring channels, so if a cosmic-ray spike is still present when
  denoising runs, it gets blurred into a wider, still-visible bump instead of being
  removed - the spike survives, it's just uglier afterwards. Despiking first
  removes it outright before it can contaminate its neighbours.
- **IRF removal before normalisation** (see above) - the IRF/Rayleigh line is
  typically far more intense than the Brillouin peaks, so a `MaxIntensity`
  normalisation run beforehand would just rescale everything relative to the IRF's
  peak height rather than the signal you actually care about.
- **`pixelwise=True` vs `False`** on the normalisation steps controls whether every
  spectrum is normalised individually (use this if pixel-to-pixel intensity
  differences are an acquisition artefact you want to remove) or all spectra are
  scaled together by a single global factor (use this if those intensity
  differences carry real information, e.g. local reflectivity, that you want to
  preserve for comparison across pixels).
