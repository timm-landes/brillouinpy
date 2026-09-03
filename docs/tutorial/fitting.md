# Fitting spectra

*Script: `examples/09_fit_spectra.py`*

`analysis.fitmodel.DHO` fits a Damped Harmonic Oscillator lineshape - the
physically correct model for a Brillouin peak, describing the material's acoustic
phonon mode as a damped oscillator driven by thermal fluctuations - to every
spectrum in a spectral object, in parallel across a process pool. This step goes
back to the single-mode data from [loading](loading.md)/[preprocessing](preprocessing.md),
so `expected_peaks=1` here - for the two-doublet data explored in
[exploratory analysis](exploratory-analysis.md), `expected_peaks=2` would be the
right setting instead (see below):

```python
dho_fit = bp.analysis.fitmodel.DHO(
    expected_peaks=1,
    p0=[0.005, 8.5, 1, 0, 0],  # [Amplitude, FreqShift (GHz), FWHM (GHz), Background, Asymmetry]
    bounds=None,
)
fitted_parameters, covariances = dho_fit.apply(preprocessed_image)
# fitted_parameters has shape (x, y, 5)
```

```{image} /_static/tutorial/09_fit_overlay.png
:alt: The preprocessed mean spectrum (purple) with the mean DHO fit result overlaid as a red line, closely tracking both peaks of the Stokes/anti-Stokes doublet.
:width: 480px
:align: center
```

Note that `expected_peaks` counts *modes* (i.e. symmetric peak pairs), not
individual peaks in the plot - `expected_peaks=1` is correct for the single-mode
data used here, even though it visibly produces two peaks (see the note on the
[tutorial overview](index.md)). For `expected_peaks > 1` (e.g. two overlapping
materials each contributing their own mode), `p0`/`bounds` grow to length
`expected_peaks * 3 + 2` (one `[Amplitude, FreqShift, FWHM]` triplet per mode,
followed by the shared `[Background, Asymmetry]`) - one triplet per doublet found
in the variance spectrum, roughly centred on the peak positions read off the mean
spectrum.

A good `p0` (initial guess) matters: `scipy.optimize.curve_fit` (used internally)
is a local optimiser, so a wildly wrong starting frequency shift or linewidth can
make it converge on a spurious local minimum, or fail outright and get discarded as
a `RuntimeError` (`analysis.fitmodel` prints a warning per failed pixel and returns
`NaN` there). `bounds` can additionally constrain the search space (e.g. to keep the
frequency shift within a physically plausible range) once you have a rough idea of
where the fit should land.

`analysis.fitmodel.Lorentzian` follows the exact same interface if you need a
simpler (if less physically accurate) lineshape instead. Because fitting spawns a
process pool, fitting scripts must guard their entry point with `if __name__ ==
'__main__':` (as all the example scripts do) - required on Windows in particular,
where child processes re-import the launching script from scratch.

```{image} /_static/tutorial/09_fit_maps.png
:alt: Three side-by-side heatmaps of the fitted amplitude, frequency shift and linewidth per pixel. Amplitude and linewidth show mild pixel-to-pixel noise around a consistent value; frequency shift is mostly uniform around 8.5 GHz with a handful of clear outlier pixels where the fit converged on the wrong value.
:width: 900px
:align: center
```

The per-pixel parameter maps are the actual payoff of fitting an *imaging* dataset:
here, amplitude and linewidth vary mildly pixel-to-pixel around a consistent value
(as expected - the synthetic data has no real spatial structure in those
parameters), while the frequency-shift map is mostly uniform but shows a handful of
clear outlier pixels where noise made the fit converge on the wrong value entirely.
This is normal for real data too - a quick look at each parameter map (or the
returned `covariances`, an estimate of the fit's parameter uncertainty per pixel)
is worth doing before trusting a fit result at face value, e.g. to catch and mask
out such outlier pixels rather than propagating them into further analysis.
