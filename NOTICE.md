# Third-party notices

BrillouinPy's own code is licensed under the BSD 3-Clause License (see
[`LICENSE`](LICENSE)). It also incorporates code from the following third-party
projects, under their own licenses (full texts in
[`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/)). Every file listed below also
carries its own inline attribution comment/docstring at the top (or just above
the relevant class/function, where only part of the file is derived) with the
same information, so the notice travels with the code itself, not just this
central file.

## RamanSPy (BSD 3-Clause)

[RamanSPy](https://github.com/barahona-research-group/RamanSPy), Copyright (c)
2023 Dimitar Georgiev, licensed under the [BSD 3-Clause
License](THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt). The following files
are adapted from it - verbatim or near-verbatim unless noted otherwise:

| File | Extent | Notable BrillouinPy-specific changes |
|---|---|---|
| `brillouinpy/core.py` | Whole file (`SpectralContainer`/`Spectrum`/`SpectralImage`/`SpectralVolume` hierarchy) | Masked-array/NaN support throughout |
| `brillouinpy/utils.py` | `is_aligned()` only; rest of the file is original | - |
| `brillouinpy/preprocessing/Step.py` | Whole file (`PreprocessingStep`) | Masked-array/NaN handling in `_process_object` |
| `brillouinpy/preprocessing/Pipeline.py` | Whole file (`Pipeline`) | - |
| `brillouinpy/preprocessing/despike.py` | Whole file (`WhitakerHayes`) | - |
| `brillouinpy/preprocessing/denoise.py` | Whole file (`SavGol`/`Gaussian`/`Whittaker`/`Kernel`) | - |
| `brillouinpy/preprocessing/normalise.py` | Whole file (`Vector`/`MinMax`/`MaxIntensity`/`AUC`) | NaN-aware pixelwise min/max |
| `brillouinpy/preprocessing/misc.py` | `BackgroundSubtractor`, `Cropper` only; `IRF_Remover`/`Deconvoluter_IRF` are original | - |
| `brillouinpy/analysis/Step.py` | Whole file (`AnalysisStep`) | Drops/re-embeds NaN-containing spectral channels around the analysis method call |
| `brillouinpy/analysis/decompose.py` | Whole file (`PCA`/`ICA`/`NMF` wrappers) | - |
| `brillouinpy/analysis/cluster.py` | Whole file (`KMeans` wrapper) | - |
| `brillouinpy/plot/_core.py` | Whole file | - |
| `brillouinpy/plot/plot.py` | Partial: the `spectra`/`mean_spectra` structure and the `@scalable` decorator pattern behind `image`/`volume` | `peaks`, `peak_dist` and the rest are original |

`brillouinpy/preprocessing/protocols.py`, the `brillouinpy/analysis/fit/`
subpackage (`core.py`, `lineshapes.py`, `segmented.py`, `step.py`),
`brillouinpy/export.py`, and the remainder of `brillouinpy/utils.py` are original
to this package (no RamanSPy content).

## Laadr/VCA (Apache License 2.0)

Vertex Component Analysis implementation (`_vca` in
`brillouinpy/analysis/unmix.py`), from [Laadr/VCA](https://github.com/Laadr/VCA),
Copyright 2018 Adrien Lagrange, licensed under the [Apache License
2.0](THIRD_PARTY_LICENSES/VCA-Apache-2.0.txt). Already carries its own inline
attribution comment above the `_vca` function listing the specific changes made
relative to the original.

## Optional runtime dependencies

These are not incorporated into brillouinpy's own source (no code was copied) -
they're only imported, at runtime, when the corresponding optional `export.*`
function is used. Listed here because their licenses are less permissive than
the BSD/MIT-family stack the rest of the package depends on:

| Package | Used by | License |
|---|---|---|
| [`brimfile`](https://github.com/brillouin-imaging/brimfile) (extra: `brim`) | `export.to_brim`/`export.from_brim` | LGPL-3.0-or-later |
