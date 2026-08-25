# Third-party notices

BrillouinAnalyzer's own code is licensed under the BSD 3-Clause License (see
[`LICENSE`](LICENSE)). It also incorporates code from the following third-party
projects, under their own licenses (full texts in
[`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/)):

| Component | Source | License |
|---|---|---|
| Overall package architecture: the `SpectralContainer`/`Spectrum`/`SpectralImage`/`SpectralVolume` class hierarchy, the `PreprocessingStep`/`Pipeline` and `AnalysisStep` patterns, and parts of the plotting API (`brillouinanalyzer/core.py`, `brillouinanalyzer/preprocessing/`, `brillouinanalyzer/analysis/Step.py`, `brillouinanalyzer/plot/`) | [RamanSPy](https://github.com/barahona-research-group/RamanSPy), Copyright (c) 2023 Dimitar Georgiev | [BSD 3-Clause](THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt) |
| Vertex Component Analysis implementation (`_vca` in `brillouinanalyzer/analysis/unmix.py`) | [Laadr/VCA](https://github.com/Laadr/VCA), Copyright 2018 Adrien Lagrange | [Apache License 2.0](THIRD_PARTY_LICENSES/VCA-Apache-2.0.txt) |

`brillouinanalyzer/analysis/unmix.py` already carries an inline attribution
comment above the `_vca` function listing the specific changes made relative to
the original.

**Note:** this table reflects the architectural components known to be adapted
from RamanSPy at the time of writing. Before making the repository public, do a
pass through the codebase (particularly `brillouinanalyzer/preprocessing/` and
`brillouinanalyzer/plot/`) to confirm no other file-level copy-paste from
RamanSPy is left unmarked, and extend this table/add inline comments accordingly.

## Optional runtime dependencies

These are not incorporated into brillouinanalyzer's own source (no code was
copied) - they're only imported, at runtime, when the corresponding optional
`export.*` function is used. Listed here because their licenses are less
permissive than the BSD/MIT-family stack the rest of the package depends on:

| Package | Used by | License |
|---|---|---|
| [`brimfile`](https://github.com/brillouin-imaging/brimfile) (extra: `brim`) | `export.to_brim`/`export.from_brim` | LGPL-3.0-or-later |
