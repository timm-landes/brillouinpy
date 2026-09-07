# -*- coding: utf-8 -*-
"""
Regenerates the figures embedded in the docs/tutorial/ pages (docs/_static/tutorial/*.png).

Run with:

    python docs/generate_tutorial_images.py

from the repository root. Uses the same synthetic data as the examples in
examples/ (via examples/_synthetic_data.py) so the figures stay reproducible and
don't depend on access to real measurement data. Re-run this whenever the
underlying examples change in a way that would make the figures misleading.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, 'docs', '_static', 'tutorial')


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close('all')
    print(f'Wrote {path}')


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(REPO_ROOT, 'examples'))
    os.makedirs(OUT_DIR, exist_ok=True)

    import brillouinpy as bp
    from _synthetic_data import (single_peak_image, image_with_irf, two_material_image,
                                 irf_broadened_image)

    # -----------------------------------------------------------------------
    # 1. Loading data
    # -----------------------------------------------------------------------
    raw_image = single_peak_image()
    plt.figure(figsize=(5, 4))
    bp.plot.mean_spectra(raw_image, title='Loaded Brillouin data', yscale='log')
    savefig('01_loaded_data.png')

    # -----------------------------------------------------------------------
    # 2. IRF removal
    # -----------------------------------------------------------------------
    irf_image, irf_index = image_with_irf()
    cleaned_image = bp.preprocessing.misc.Deconvoluter_IRF(offset=3, iterations=4, padding=20).apply(irf_image)

    fig = plt.figure(figsize=(9, 4), layout='constrained')
    plt.subplot(121)
    bp.plot.spectra(irf_image[0, 0], title='Raw spectrum (with IRF)', yscale='linear')
    plt.subplot(122)
    bp.plot.spectra(cleaned_image[0, 0], title='IRF removed (deconvolved)', yscale='linear')
    savefig('02_irf_removal.png')

    # -----------------------------------------------------------------------
    # 3. Preprocessing pipeline
    # -----------------------------------------------------------------------
    pipeline = bp.preprocessing.Pipeline([
        bp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=8),
        bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ])
    preprocessed_image = pipeline.apply(raw_image)

    fig = plt.figure(figsize=(9, 4), layout='constrained')
    plt.subplot(121)
    bp.plot.mean_spectra(raw_image, title='Raw', yscale='log')
    plt.subplot(122)
    bp.plot.mean_spectra(preprocessed_image, title='Despiked + denoised + normalised', yscale='linear')
    savefig('03_preprocessing.png')

    # -----------------------------------------------------------------------
    # 4. Classical data analysis (mean & variance)
    # -----------------------------------------------------------------------
    # Frequency shifts much closer together than the default (6.0/11.0 GHz, used
    # for the unmixing/decomposition/clustering figures below) so the two doublets
    # fully merge in the mean spectrum, demonstrating what the variance spectrum adds.
    classical_image, _, _ = two_material_image(freq_shift_a=7.0, freq_shift_b=7.5)
    classical_preprocessed = bp.preprocessing.Pipeline([
        bp.preprocessing.denoise.SavGol(window_length=7, polyorder=2),
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ]).apply(classical_image)

    mean_spectrum = classical_preprocessed.mean
    variance_spectrum = classical_preprocessed.variance

    fig = plt.figure(figsize=(9, 4), layout='constrained')
    plt.subplot(121)
    bp.plot.spectra(mean_spectrum, title='Mean spectrum', yscale='linear')
    plt.subplot(122)
    bp.plot.peaks(
        variance_spectrum, title='Variance spectrum', yscale='linear',
        prominence=variance_spectrum.spectral_data.max() * 0.1,
    )
    savefig('04_classical_analysis.png')

    # -----------------------------------------------------------------------
    # 5. Unmixing (VCA)
    # -----------------------------------------------------------------------
    mix_image, true_abundances, true_spectra = two_material_image()
    mix_preprocessed = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True),
    ]).apply(mix_image)

    variances = bp.analysis.variance_explained(
        mix_preprocessed,
        lambda n: bp.analysis.unmix.VCA(n_endmembers=n, abundance_method='ucls'),
        param_values=range(1, 5),
    )
    plt.figure(figsize=(5, 4))
    plt.plot(list(variances.keys()), list(variances.values()), marker='o')
    plt.xlabel('n_endmembers')
    plt.ylabel('Variance explained')
    plt.title('Choosing n_endmembers')
    savefig('05a_variance_explained.png')

    unmixer = bp.analysis.unmix.VCA(n_endmembers=2, abundance_method='ucls')
    abundance_maps, endmembers = unmixer.apply(mix_preprocessed)

    plt.figure(figsize=(9, 4), layout='tight')
    plt.subplot(121)
    bp.plot.spectra(
        endmembers, mix_preprocessed.spectral_axis, plot_type='single',
        label=[f'Endmember {i + 1}' for i in range(len(endmembers))], yscale='linear',
    )
    ax = plt.subplot(122)
    cmap = plt.get_cmap()(np.linspace(0, 1, len(abundance_maps)))
    white = [1, 1, 1, 0]
    for i, abundance_map in enumerate(abundance_maps):
        ax.imshow(abundance_map, cmap=LinearSegmentedColormap.from_list('', [white, cmap[i]]))
    ax.set_title('Abundance maps')
    savefig('05_vca_unmixing.png')

    # -----------------------------------------------------------------------
    # 6. NMF
    # -----------------------------------------------------------------------
    nmf_preprocessed = bp.preprocessing.Pipeline([
        bp.preprocessing.normalise.MinMax(pixelwise=True),
    ]).apply(mix_image)
    nmf = bp.analysis.decompose.NMF(n_components=2, init='nndsvda', max_iter=500, random_state=0)
    scores, components = nmf.apply(nmf_preprocessed)

    plt.figure(figsize=(9, 7), layout='constrained')
    for i in range(len(components)):
        plt.subplot(2, len(components), i + 1)
        plt.imshow(scores[i])
        plt.colorbar(label=f'Component {i + 1} score')
        plt.subplot(2, len(components), len(components) + i + 1)
        plt.plot(nmf_preprocessed.spectral_axis, components[i])
        plt.xlabel('Brillouin shift (GHz)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(f'Component {i + 1} spectrum')
    savefig('06_nmf.png')

    # -----------------------------------------------------------------------
    # 7. PCA
    # -----------------------------------------------------------------------
    pca = bp.analysis.decompose.PCA(n_components=3, random_state=0)
    scores, components = pca.apply(mix_preprocessed)

    plt.figure(figsize=(11, 7), layout='constrained')
    for i in range(len(components)):
        plt.subplot(2, len(components), i + 1)
        plt.imshow(scores[i], cmap='RdBu_r')
        plt.colorbar(label=f'PC{i + 1} score')
        plt.subplot(2, len(components), len(components) + i + 1)
        plt.plot(mix_preprocessed.spectral_axis, components[i])
        plt.xlabel('Brillouin shift (GHz)')
        plt.ylabel('Loading (a.u.)')
        plt.title(f'PC{i + 1} loading')
    savefig('07_pca.png')

    # -----------------------------------------------------------------------
    # 8. Clustering (k-means)
    # -----------------------------------------------------------------------
    kmeans = bp.analysis.cluster.KMeans(n_clusters=2, random_state=0)
    memberships, centers = kmeans.apply(mix_preprocessed)
    cluster_map = np.argmax(np.stack(memberships, axis=-1), axis=-1)

    plt.figure(figsize=(10, 4), layout='constrained')
    plt.subplot(121)
    im = plt.imshow(cluster_map, cmap='viridis')
    plt.colorbar(im, ticks=range(len(centers)), label='Cluster')
    plt.title('Cluster assignment')
    plt.subplot(122)
    bp.plot.spectra(
        centers, mix_preprocessed.spectral_axis, plot_type='single',
        label=[f'Cluster {i + 1} centre' for i in range(len(centers))], yscale='linear',
    )
    plt.title('Cluster-centre spectra')
    savefig('08_kmeans.png')

    # -----------------------------------------------------------------------
    # 9. Fitting
    # -----------------------------------------------------------------------
    dho_fit = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=[0.005, 8.5, 1, 0, 0], bounds=None)
    fitted_parameters, _ = dho_fit.apply(preprocessed_image)

    mean_amplitude, mean_shift, mean_linewidth, mean_bg, mean_axis_shift = (
        fitted_parameters[..., i].mean() for i in range(5)
    )
    plt.figure(figsize=(5, 4))
    bp.plot.mean_spectra(preprocessed_image, title='Fit vs. data', yscale='linear')
    plt.plot(
        preprocessed_image.spectral_axis,
        bp.analysis.fitmodel._DHO_1(
            preprocessed_image.spectral_axis, mean_amplitude, mean_shift, mean_linewidth, mean_bg, mean_axis_shift
        ),
        label='Mean DHO fit', color='red',
    )
    plt.legend()
    savefig('09_fit_overlay.png')

    plt.figure(figsize=(12, 4), layout='constrained')
    plt.subplot(131)
    plt.imshow(fitted_parameters[:, :, 0])
    plt.colorbar(label='Amplitude (a.u.)')
    plt.title('Amplitude')
    plt.subplot(132)
    plt.imshow(fitted_parameters[:, :, 1])
    plt.colorbar(label='Frequency shift (GHz)')
    plt.title('Frequency shift')
    plt.subplot(133)
    plt.imshow(fitted_parameters[:, :, 2])
    plt.colorbar(label='Linewidth (GHz)')
    plt.title('Linewidth')
    savefig('09_fit_maps.png')

    # -----------------------------------------------------------------------
    # 14. Plain fit vs. IRF-convolution fit (synthetic panel only; the
    #     real-data panel needs a measurement and ships as a static asset)
    # -----------------------------------------------------------------------
    irf_image, true_lw_map, irf_info = irf_broadened_image(
        nx=40, ny=40, component_shift=6.5, irf_fwhm=0.8, irf_shape='gaussian',
        intrinsic_linewidth_background=0.9, intrinsic_linewidth_blob=0.45,
        elastic_amplitude=1.5, noise_model='poisson', peak_photons=500,
    )
    irf_prepared = bp.preprocessing.misc.IRF_Remover(offset=6, store_irf=True).apply(irf_image)
    irf_p0, irf_bounds = [3e-3, 6.5, 1.0, 1e-4, 0.0], ([0, 5, 0.05, -1e-2, -1], [1, 8, 5, 1e-1, 1])
    irf_plain, _ = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=irf_p0, bounds=irf_bounds).apply(irf_prepared)
    irf_param, _ = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=irf_p0, bounds=irf_bounds,
                                            irf=('gaussian', irf_info['irf_fwhm'])).apply(irf_prepared)
    irf_auto, _ = bp.analysis.fitmodel.DHO(expected_peaks=1, p0=irf_p0, bounds=irf_bounds,
                                           irf='auto').apply(irf_prepared)

    plt.figure(figsize=(15, 3.4), layout='constrained')
    for i, (m, t) in enumerate([
        (true_lw_map, 'True intrinsic'),
        (np.abs(irf_plain[..., 2]), 'Plain fit'),
        (np.abs(irf_param[..., 2]), "irf=('gaussian', 0.8)"),
        (np.abs(irf_auto[..., 2]), "irf='auto'"),
    ]):
        ax = plt.subplot(1, 4, i + 1)
        im = ax.imshow(m, vmin=0.3, vmax=1.4)
        ax.set_title(f'{t}\nlinewidth (GHz)', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    savefig('14_irf_fit_synthetic.png')

    print('Done.')
