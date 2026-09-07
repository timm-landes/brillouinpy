# Benchmarks / method studies

Study scripts that compare BrillouinPy's analysis methods against each other on
synthetic data with known ground truth. **These are deliberately not in
`examples/`** and are not part of the built documentation or the tutorial:

- the `examples/` scripts each teach *one* API in isolation, kept short;
- these scripts weigh several methods against one another and are only meaningful
  read as a whole, with their figures.

They exist so the trade-offs between the analysis methods are reproducible rather
than folklore, and double as a regression check on the analysis code. They draw
no conclusions for you: each script writes its figures (and, for some, a
`results.json`) to `report/` - look at those and decide what fits your data.

They reuse the synthetic-data generators from `examples/_synthetic_data.py`
(added to `sys.path` in `_assay.py`) so there is a single source of truth for the
test data. `_assay.py` holds the shared parts: the methods under test
(DHO fit — 1-peak / raw 2-peak / background-subtracted / segmented (2- and
3-component) — spectral phasor, raw k-means, PCA, NMF, VCA), the ground-truth
scoring (adjusted Rand index), the sweep loops and the standard figures. Each
sweep script exposes a `run(save_dir=None)` and is a thin driver that builds a
list of `(parameter, image, label_map)` cases and hands it over.

## Scripts

| Script | Question |
|---|---|
| `method_limitations.py` | Two hard-separated pure domains: how close in Brillouin shift can they be before each method stops segmenting them? Sweeps the shift separation at fixed noise. |
| `snr_requirement.py` | Same two domains, shift separation fixed: what minimum SNR does each method need? Sweeps the per-pixel noise (x-axis = peak SNR, log scale). |
| `additive_blob.py` | Not two pure domains: one constant background spectrum everywhere (a hydrated sample's water signal) plus a second component added inside a blob, so no pixel is pure. Sweeps the added component's shift - separates methods that need pure endmembers (VCA) from those keyed to the change in spectral shape. Scores both segmentation (ARI) and shift **and linewidth** recovery of the added component. |
| `three_component.py` | A cell-in-medium geometry: medium/background everywhere, cytoplasm added inside a disc, nucleus added inside a smaller concentric disc - three nested additive components. Sweeps the nucleus shift. Scores 3-way segmentation ARI for all classifiers, and per-component (background/cytoplasm/nucleus) shift **and linewidth** recovery for the hierarchical `DHO fit (segmented, 3-comp)` method. |
| `irf_convolution.py` | Recovering the *intrinsic* linewidth from an IRF-broadened doublet: plain fit vs. Richardson-Lucy deconvolution + fit vs. `DHO(irf=...)` convolution fit. Sweeps the IRF width and, separately, the SNR. |
| `phasor_paper_reproduction.py` | A Monte-Carlo look at Elsayad (2019)'s Fig. 3/5 claims (repeated noisy realisations of *pure* single-peak spectra - a different question from the spatial/additive experiments above). Standalone; parked, kept for reference. |

The four sweep scripts (`method_limitations`, `snr_requirement`, `additive_blob`,
`three_component`) each render three figures: ARI vs. the swept parameter
alongside the per-method wall time (mean ± standard error, log y-axis); the
predicted segmentation vs. ground truth at one snapshot value; and the continuous
parameter maps (DHO shift, phasor shift, PC1) at that snapshot. `additive_blob.py`
adds a shift- and a linewidth-recovery figure; `three_component.py` adds six
(background/cytoplasm/nucleus × shift/linewidth). `irf_convolution.py` writes one
figure and prints its tables.

## Running

```bash
python benchmarks/method_limitations.py   # runs one experiment, shows its figures
python benchmarks/three_component.py       # etc.
```

Each sweep is ~9-14 points on a 24×24-pixel image and the DHO fit spawns a
process pool per point, so a full run of one script is a few minutes. Drop
`NX`/`NY` or shorten the sweep array (module constants at the top of each script)
while iterating.

Needs `matplotlib` and `scikit-learn`. VCA's own algorithm
(`brillouinpy.analysis.unmix.VCA`) is plain NumPy/SciPy; only the unused
alternative endmember finders `PPI`/`FIPPI`/`NFINDR` would need the optional
`pysptools` dependency.

## Results in `report/`

`report/figures/` holds the rendered figures; `report/results.json` holds the raw
sweep numbers for the four segmentation experiments (ARI, recovery errors,
timings) and `report/results_e.json` the summary numbers for
`phasor_paper_reproduction.py`. Diff these across code changes. Regenerate a
script's outputs by running it with a `save_dir` (its `if __name__ == "__main__"`
block does this) after any change to the analysis code or the script.
