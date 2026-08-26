# -*- coding: utf-8 -*-
"""
Example 5 - Spectral unmixing with VCA
========================================

Shows how to use :class:`brillouinpy.analysis.unmix.VCA` (Vertex Component
Analysis) to automatically find the "purest" spectra (endmembers) in a spectral
image, and how much of each is present at every pixel (abundance maps) - useful to
separate spatially-mixed materials without having to fit individual peaks.

Other endmember-finding algorithms in :mod:`brillouinpy.analysis.unmix` follow
the exact same interface, e.g. :class:`~brillouinpy.analysis.unmix.NFINDR`
(N-FINDR). VCA is used here because, unlike NFINDR/PPI/FIPPI, it's implemented
natively in brillouinpy rather than delegating to the `pysptools
<https://pysptools.sourceforge.io/>`_ package - so it doesn't depend on pysptools
staying compatible with newer scipy versions (as of pysptools 0.15.0/scipy >= 1.13,
NFINDR/PPI/FIPPI raise ``AttributeError: module 'scipy.linalg' has no attribute
'_flinalg'`` because pysptools calls a private scipy API that has since been
removed).

This example uses synthetic data mixing two materials so the "correct" answer
(2 endmembers, a smooth left-to-right abundance gradient) is known in advance.

VCA assumes at least one (near-)pure pixel is present per endmember - i.e. that
somewhere in the image, each material dominates the pixel almost completely - and
needs that to reliably recover the correct endmembers. The gradient built into
``two_material_image()`` reaches abundance 0/1 at its edges, so that assumption
holds here; real data without any sufficiently pure pixels (e.g. every pixel a
genuine, inseparable mixture) can make VCA's endmembers inaccurate. NMF
(``06_decompose_nmf.py``) makes no such assumption - it optimises all components
simultaneously against the whole dataset - which is worth trying as a cross-check
if you suspect your data lacks pure pixels. See Prats-Mateu et al., "Multivariate
unmixing approaches on Raman images of plant cell walls: new insights or
overinterpretation of results?", Plant Methods 14:52 (2018),
https://doi.org/10.1186/s13007-018-0320-9.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import brillouinpy as bp
from _synthetic_data import two_material_image

if __name__ == '__main__':
    brillouin_image, true_abundances, true_spectra = two_material_image()

    preprocessed_image = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ]).apply(brillouin_image)

    # n_endmembers is the number of distinct materials you expect to find.
    # abundance_method picks the algorithm used to derive the abundance maps from the
    # endmembers found ('ucls', 'nnls' or 'fcls' - see the class docstring).
    unmixer = bp.analysis.unmix.VCA(n_endmembers=2, abundance_method='ucls')
    abundance_maps, endmembers = unmixer.apply(preprocessed_image)

    # Plot the found endmember spectra
    plt.figure(figsize=(10, 5), layout='tight')
    plt.subplot(121)
    bp.plot.spectra(
        endmembers, preprocessed_image.spectral_axis, plot_type='single',
        label=[f"Endmember {i + 1}" for i in range(len(endmembers))], yscale='linear',
    )

    # Overlay each endmember's abundance map, colour-coded
    ax = plt.subplot(122)
    cmap = plt.get_cmap()(np.linspace(0, 1, len(abundance_maps)))
    white = [1, 1, 1, 0]
    for i, abundance_map in enumerate(abundance_maps):
        ax.imshow(abundance_map, cmap=LinearSegmentedColormap.from_list('', [white, cmap[i]]))
    ax.set_title('Abundance maps')
    plt.show()
