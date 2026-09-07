# Multi-component and mixed images

*Scripts: `examples/12_segmented_fit.py`, `examples/16_auto_peak_count.py`, `examples/11_phasor_analysis.py`*

The [fitting page](fitting.md) assumes every pixel holds the same, known number of
Brillouin modes. Real samples often break that assumption: a hydrated specimen has
a medium spectrum *everywhere* and a stiffer component (an inclusion, a cell
nucleus) only in part of the field of view. This page covers the tools for that
case and, just as importantly, when to reach for which.

## Three spatial scales

Whether a pixel's spectrum is *one* Brillouin mode or a *sum* of several is
decided by the relationship between three length scales, so it is worth being
explicit about them (Prevedel et al., ["Brillouin microscopy: an emerging tool
for mechanobiology"](https://doi.org/10.1038/s41592-019-0543-3),
*Nat. Methods* 16:969-977, 2019).

1. **The acoustic phonon wavelength Λ** - the length scale the measurement
   mechanically probes. It is fixed by the scattering geometry,
   `Λ = λ_laser / (2 n sin(θ/2))`, and the Brillouin shift is inversely the
   phonon wavelength, `ν_B = V / Λ`. For a 532 nm laser in backscattering
   (θ = 180°) through a cell/water refractive index n ≈ 1.33, **Λ ≈ 200 nm**
   and ν_B ≈ 7.5 GHz. Below Λ the notion of a single longitudinal modulus "at
   that point" is not defined.

2. **The optical voxel** - what one pixel integrates over spatially: roughly
   0.25-0.5 µm laterally and 1-several µm axially for a confocal Brillouin
   setup. This is larger than Λ in every direction (much larger axially), so a
   single voxel spans many phonon wavelengths and can straddle a boundary
   between materials.

3. **The structural scale** - cells (~10 µm), nucleus (~5 µm), membranes, ECM
   fibres, tissue domains: what you actually want to map, and what the `labels`
   output below recovers.

The mechanical contrast is generated at Λ but recorded at the voxel scale, and
the two differ. What a voxel that spans two mechanically distinct phases records
therefore depends on how large those phases are relative to Λ:

| Heterogeneity within the voxel | Measured spectrum | Handled by |
|---|---|---|
| Two or more phases, **each larger than Λ** | Incoherent sum of their doublets, weighted by volume fraction | multi-component fitting (this page) |
| Structure **finer than Λ** | A single effective-medium peak, possibly shifted / broadened | nothing here - it is one mode; fit it as one |

Multi-component fitting is the tool for the first row only. A broadened single
peak produced by sub-Λ heterogeneity is not a two-peak problem, and adding a
second peak to it just fits noise.

## Why "fit the maximum number of peaks everywhere, then filter" fails

The obvious approach - run `DHO(expected_peaks=2)` on the whole image and discard
the second peak where it looks spurious - does not work, and the reason is
structural rather than a tuning problem. Wherever the second component's amplitude
approaches zero, the two-peak model is **rank-deficient**: the data no longer
constrains that peak's three parameters, so `curve_fit` either fails outright or
plants a narrow spike on a noise excursion. That contamination is a persistent
fraction of pixels - every pixel that lies in a genuine component's feature range
but has no second component - not a handful of outliers a threshold can reject.
Post-hoc filtering (by amplitude, robust threshold, or fit covariance) can recover the
right answer in certain cases, but generally if fails because the bad fits are not
distinguishable from marginal good ones. The `benchmarks/` suite quantifies this
(adjusted Rand index of the two-peak fit sits *below chance* across a range of
component separations).

## `segmented_fit`: classify first, then fit

`analysis.segmented_fit` does the workflow that works - classify the image with a
cheap model-free classifier, then fit **exactly as many peaks as each class holds**,
carrying the peaks found in an earlier stage forward as anchors:

```python
result = bp.analysis.segmented_fit(image, n_components=2)

result.labels        # 0 = fewest components (e.g. background only), 1 = one more, ...
result.shift         # per pixel, the *newest* component's shift (GHz)
result.linewidth     # per pixel, the newest component's HWHM (GHz)
result.amplitude     # per pixel, the newest component's amplitude
result.stage_shift   # [median shift of stage 0's peak, stage 1's, ...] - diagnostics
```

Each pixel's `shift`/`linewidth`/`amplitude` describe the component its class
*introduces* on top of the previous class: the background shift where `labels == 0`,
the inclusion shift where `labels == 1`, and so on.

### The nested-additive assumption

This is the volume-fraction incoherent-sum picture from *Three spatial scales*
made concrete. `segmented_fit` assumes a **nested additive** geometry: class 0
pixels contain one peak, class 1 pixels contain that same peak plus one more, ...,
each class strictly contains more signal than the previous one, everywhere - i.e.
each component occupies its own sub-volume of the voxel, larger than Λ so it
carries its own doublet, and the class map tracks which components are present.
The default `'kmeans'` classifier exploits exactly this by ordering the k-means
clusters by ascending total intensity. If that ordering does not hold for your data but some other classifier
recovers the right nesting order, pass it as `classifier=callable` - it is called as
`classifier(image, n_components)` and must return an integer label map ordered from
fewest to most components. `n_components` is 2 or 3 (the fit engine has closed-form
models for one to three peaks).

### Lineshape choice

`segmented_fit(..., model='lorentzian')` fits a Lorentzian doublet at each stage
instead of the default DHO (`model='dho'`). The result object is identical - same
parameter layout, `linewidth` is the HWHM either way - so this only changes the
lineshape, not the interface. Use `'lorentzian'` for the same reasons as in a plain
fit (IRF-dominated width, robustness, cross-checking); use `'dho'` when the fitted
linewidths feed into mechanical properties (see the parameter conventions on the
[fitting page](fitting.md)).

## Choosing between the approaches

The methods answer genuinely different questions:

| Method | Returns | Answers | Fails / limits |
|---|---|---|---|
| `DHO(expected_peaks=N)` everywhere | shift, linewidth, amplitude per mode | "Fit N modes I already know are present" | Spurious peaks in pixels where a mode's amplitude → 0 (see above) |
| `fitmodel.estimate_peak_count(image)` | one integer for the whole image | "How many modes does this sample have?" | Biased upward by anything the DHO misses (elastic wing, non-Lorentzian lineshape, un-deconvolved IRF) |
| `DHO(expected_peaks='auto')` | NaN-padded params + per-pixel count (`fitted_peak_count`) | "How many modes at *each* pixel?" | Per-pixel model selection is noisy; same upward bias, per pixel |
| `segmented_fit(image, n_components)` | per-pixel shift/linewidth/amplitude of the *newest* component + class map | "Recover each component's parameters in a mixed, nested image" | Needs the nested-additive geometry; needs a classifier that recovers the nesting order |
| unmixing / NMF / k-means ([exploratory analysis](exploratory-analysis.md)) | endmember/component spectra + abundance/label maps | "Separate spatially mixed materials without a peak model" | Classifies and separates, but returns **no** shift or linewidth - not physical parameters |
| phasor (`analysis.phasor`, `examples/11`) | `(G, S)` point per pixel; `phase_to_shift` → shift | "Model-free shift map, fast, for noisy data or live preview" | Gives a shift but **no linewidth**; shift error grows as components overlap |

The practical decision:

- **Known, constant mode count** → plain `DHO(expected_peaks=N)`. This is the
  common case and everything else is overkill.
- **Unknown count, one number for the sample** → `estimate_peak_count`, then a
  plain fixed-count fit.
- **A background everywhere plus components added in sub-regions** →
  `segmented_fit`. This is the one case where the multivariate methods can match
  the *classification* but cannot give you the per-component shift and linewidth,
  and where a plain multi-peak fit produces the spurious-peak failure above.
- **Just need contrast / a shift map, fast** → phasor or a multivariate method.
  Fit only the regions that turn out to matter.

`segmented_fit`'s specific edge is **quantitative recovery of shift and linewidth
per component in spectra that are never pure** - the multivariate methods classify
but don't measure, phasor measures a shift but not a width, and a plain multi-peak
fit is unreliable there. Where you *can* measure and subtract a clean background
spectrum, plain background subtraction followed by a one-peak fit can beat it at
close component separations; `segmented_fit` earns its place when that background
spectrum is not separately available.
