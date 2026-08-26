# -*- coding: utf-8 -*-
"""
Example 7 - Principal Component Analysis (PCA)
=================================================

Shows how to use :class:`brillouinpy.analysis.decompose.PCA` to reduce a
spectral image to its dominant modes of variation - useful as a quick way to spot
spatial structure/heterogeneity in a dataset, or as a preprocessing step before
clustering.

Unlike NMF, PCA components can be negative and aren't easily interpreted as "pure"
material spectra - they describe variation around the mean spectrum instead.

This example uses synthetic data mixing two materials, so the first principal
component is expected to capture the left-to-right abundance gradient between them.
"""
import matplotlib.pyplot as plt

import brillouinpy as bp
from _synthetic_data import two_material_image

if __name__ == '__main__':
    brillouin_image, true_abundances, true_spectra = two_material_image()

    preprocessed_image = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ]).apply(brillouin_image)

    # n_components can be an int, a float in (0, 1) (fraction of variance to keep), or
    # 'mle'. Extra keyword arguments are forwarded to sklearn.decomposition.PCA.
    pca = bp.analysis.decompose.PCA(n_components=3, random_state=0)
    scores, components = pca.apply(preprocessed_image)
    # scores: list of (x, y) score maps, one per principal component
    # components: list of (spectral_length,) loading vectors, one per principal component

    plt.figure(figsize=(12, 8), layout='constrained')
    for i in range(len(components)):
        plt.subplot(2, len(components), i + 1)
        plt.imshow(scores[i], cmap='RdBu_r')
        plt.colorbar(label=f'PC{i + 1} score')

        plt.subplot(2, len(components), len(components) + i + 1)
        plt.plot(preprocessed_image.spectral_axis, components[i])
        plt.xlabel('Brillouin shift (GHz)')
        plt.ylabel('Loading (a.u.)')
        plt.title(f'PC{i + 1} loading')
    plt.show()
