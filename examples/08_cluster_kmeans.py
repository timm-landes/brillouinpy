# -*- coding: utf-8 -*-
"""
Example 8 - Clustering with k-means
======================================

Shows how to use `analysis.cluster.KMeans` to assign every pixel in a spectral
image to one of a fixed number of clusters based on spectral similarity - a hard
(one-cluster-per-pixel) alternative to the soft membership maps produced by
unmixing (`05_unmix_vca.py`) or decomposition (`06_decompose_nmf.py`,
`07_decompose_pca.py`). Useful as a quick, model-free way to segment a spectral
image into spatially/spectrally distinct regions, e.g. as a first pass before
deciding how many endmembers/components a more detailed unmixing/decomposition
analysis should look for.

This example uses synthetic data mixing two materials, so a 2-cluster k-means
result is expected to roughly recover the same top-to-bottom split as the
VCA/NMF/PCA examples.
"""
import matplotlib.pyplot as plt
import numpy as np

import brillouinpy as bp
from _synthetic_data import two_material_image

if __name__ == '__main__':
    brillouin_image, true_abundances, true_spectra = two_material_image()

    preprocessed_image = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ]).apply(brillouin_image)

    # n_clusters is the number of groups to partition the data into. Extra keyword
    # arguments are forwarded to sklearn.cluster.KMeans (e.g. n_init, max_iter, ...).
    kmeans = bp.analysis.cluster.KMeans(n_clusters=2, random_state=0)
    memberships, centers = kmeans.apply(preprocessed_image)
    # memberships: list of n_clusters (x, y) one-hot maps (1 where a pixel belongs
    #              to that cluster, 0 otherwise)
    # centers: list of n_clusters cluster-center spectra

    # Collapse the one-hot membership maps into a single (x, y) label map for display
    cluster_map = np.argmax(np.stack(memberships, axis=-1), axis=-1)

    plt.figure(figsize=(10, 4), layout='constrained')
    plt.subplot(121)
    im = plt.imshow(cluster_map, cmap='viridis')
    plt.colorbar(im, ticks=range(len(centers)), label='Cluster')
    plt.title('Cluster assignment')

    plt.subplot(122)
    bp.plot.spectra(
        centers, preprocessed_image.spectral_axis, plot_type='single',
        label=[f'Cluster {i + 1} centre' for i in range(len(centers))], yscale='linear',
    )
    plt.title('Cluster-centre spectra')
    plt.show()
