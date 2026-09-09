# Removing the instrument response: plain fit vs. `irf=`

*Script: `examples/14_irf_fit_comparison.py`*

A measured Brillouin spectrum is the sample's *intrinsic* lineshape convolved
with the spectrometer's **instrument response function** (IRF) - the same
narrow line the elastically scattered light produces at zero shift. The
`LineWidth` a plain [`DHO` fit](fitting.md) returns is therefore the *observed*
width: intrinsic linewidth **plus** instrumental broadening. The Brillouin
*shift* is unaffected (the IRF is symmetric), but the linewidth - which carries
the sample's viscosity / acoustic loss - is not what you want as-is.

There are two ways to correct for it:

- **Deconvolve first, then fit** - `preprocessing.misc.Deconvoluter_IRF`
  (Richardson-Lucy), then a plain `DHO` fit. Covered in
  [preprocessing](preprocessing.md).
- **Fit the convolved model** - keep the data as measured and pass `irf=` to
  `DHO`, so the model it fits is `(intrinsic DHO) ⊛ IRF + background`. The fitted
  `LineWidth` is then the intrinsic linewidth directly, with no separate
  deconvolution step and no iteration count to tune. `benchmarks/irf_convolution.py`
  compares the two across noise levels and IRF widths; this page shows how to
  *use* the second one.

## Getting the IRF to the fit

`DHO`'s `irf=` argument accepts:

| `irf=` | when |
|---|---|
| `('gaussian', fwhm)` / `('lorentzian', fwhm)` / `('voigt', fwhm_l, fwhm_g)` | you have **calibrated** your spectrometer's IRF shape and width (GHz) |
| `'auto'` | the object carries an `instrument_response_function` - see below |
| a 1-D array | you want to pass an explicit measured IRF sample yourself |

For `'auto'`, something has to put the IRF on the object first:

- **Tandem Fabry-Perot** records the elastic peak on every scan.
  `IRF_Remover(offset=..., store_irf=True)` crops it away *and* keeps it as a
  per-pixel `instrument_response_function`:

  ```python
  prepared = bp.preprocessing.misc.IRF_Remover(offset=6, store_irf=True).apply(image)
  params, _ = bp.analysis.fit.DHO(expected_peaks=1, p0=p0, bounds=bounds,
                                       irf='auto').apply(prepared)
  ```

- **VIPA** and similar - the IRF is usually measured separately (once, or a few
  times over a long acquisition for drift). Attach it with
  `preprocessing.misc.assign_irf`:

  ```python
  # measured once:
  prepared = bp.preprocessing.misc.assign_irf(prepared, irf_spectrum)
  # measured before and after, drift-corrected across the scan:
  prepared = bp.preprocessing.misc.assign_irf(
      prepared, [irf_before, irf_after],
      at=[0, prepared.flat.shape[0] - 1], method='linear')
  ```

## On synthetic data (known ground truth)

`_synthetic_data.irf_broadened_image` builds a doublet with a **known** intrinsic
linewidth (0.90 GHz outside a blob, 0.45 GHz inside), convolves it with a
0.8 GHz Gaussian IRF, and adds that IRF back as a measurable elastic peak.

```{image} /_static/tutorial/14_irf_fit_synthetic.png
:alt: Four linewidth maps side by side. The true intrinsic map is flat (0.9 GHz background, 0.45 GHz blob). The plain-fit map sits noticeably higher everywhere (~1.0/0.65 GHz). The irf=('gaussian', 0.8) map and the irf='auto' map both closely reproduce the true map.
:width: 900px
:align: center
```

| method | background linewidth | blob linewidth |
|---|---|---|
| *true intrinsic* | 0.90 | 0.45 |
| plain fit | 1.04 | 0.65 |
| `irf=('gaussian', 0.8)` (calibrated) | 0.90 | 0.45 |
| `irf='auto'` (IRF detected from the elastic peak) | 0.92 | 0.47 |

The plain fit is biased high by roughly the instrumental contribution. A
calibrated `irf=` recovers the intrinsic linewidth exactly; `irf='auto'` - with
no calibration, just the elastic peak the data already contains - recovers it to
within a few hundredths of a GHz. How close `'auto'` gets depends on how
faithfully the detected elastic peak represents the true IRF (a clean, isolated
elastic line helps; wide Lorentzian-tailed IRFs whose wings the detector
truncates are the hardest case).

## On the real measurement from `01_load_data.py`

Running the same steps on the tandem-Fabry-Perot measurement from
[loading](loading.md) (no ground truth here - this is a real sample):

```{image} /_static/tutorial/14_irf_fit_real.png
:alt: Left, the detected per-pixel IRF - a compact peak at zero shift, essentially identical across all pixels, FWHM 0.26 GHz. Middle and right, the plain-fit and irf='auto' linewidth maps, which show the same spatial structure and nearly the same values.
:width: 900px
:align: center
```

- The **detected IRF** is compact and highly consistent pixel-to-pixel, with a
  FWHM of **0.26 GHz** - a plausible number for this spectrometer.
- The **fitted shift** is identical between the two methods (8.34 vs. 8.33 GHz) -
  as expected, a symmetric IRF does not move the peak.
- The **fitted linewidth** goes from 0.90 GHz (plain) to 0.88 GHz (`irf='auto'`) -
  a **~2 % correction**, because here the IRF (0.26 GHz) is narrow next to the
  ~0.9 GHz Brillouin linewidth. The correction is small on this dataset, but it
  is the right small correction and costs nothing extra.

## Which to use

- **IRF narrow compared to your linewidths** (as above) - the plain fit is fine;
  the correction is within the fit's own scatter.
- **You have calibrated the IRF** - `irf=('gaussian'|'lorentzian'|'voigt', ...)`
  is the most accurate.
- **No calibration, IRF recorded per scan (TFP)** - `IRF_Remover(store_irf=True)`
  + `irf='auto'`.
- **IRF measured separately (VIPA)** - `assign_irf(...)` + `irf='auto'`.
- **Very noisy data / uncertain IRF** - prefer the convolution fit over
  Richardson-Lucy deconvolution: it does not amplify noise and needs no
  iteration count (see `benchmarks/irf_convolution.py`).
