# -*- coding: utf-8 -*-
"""
Example 6 - Non-negative Matrix Factorisation (NMF)
=====================================================

Shows how to use :class:`brillouinpy.analysis.decompose.NMF` to decompose a
spectral image into a small number of non-negative "source" spectra plus
per-pixel score maps of how much each source contributes - similar in spirit to
the unmixing examples, but based on scikit-learn's NMF rather than an endmember
finder.

Data must be non-negative for NMF; use
:class:`brillouinpy.preprocessing.normalise.MinMax` first if that's not
already the case for your data.

This example uses synthetic data mixing two materials so the "correct" answer
(2 components) is known in advance.
"""
import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import two_material_image

if __name__ == '__main__':
    brillouin_image, true_abundances, true_spectra = two_material_image()

    preprocessed_image = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MinMax(pixelwise=True),  # NMF needs non-negative data
    ]).apply(brillouin_image)

    # n_components is the number of source spectra to decompose into. Extra keyword
    # arguments are forwarded to sklearn.decomposition.NMF (e.g. init, max_iter, ...).
    nmf = bp.analysis.decompose.NMF(n_components=2, init='nndsvda', max_iter=500, random_state=0)
    scores, components = nmf.apply(preprocessed_image)
    # scores: list of (x, y) score maps, one per component
    # components: list of (spectral_length,) source spectra, one per component

    plt.figure(figsize=(10, 8), layout='constrained')
    for i in range(len(components)):
        plt.subplot(2, len(components), i + 1)
        plt.imshow(scores[i])
        plt.colorbar(label=f'Component {i + 1} score')

        plt.subplot(2, len(components), len(components) + i + 1)
        plt.plot(preprocessed_image.spectral_axis, components[i])
        plt.xlabel('Brillouin shift (GHz)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(f'Component {i + 1} spectrum')
    plt.show()
