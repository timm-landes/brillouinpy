"""
The AnalysisStep class below is adapted from RamanSPy
(https://github.com/barahona-research-group/RamanSPy), Copyright (c) 2023
Dimitar Georgiev, licensed under the BSD 3-Clause License (see LICENSE and
THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt in the repository root). Notable
change: spectral channels containing a NaN/masked value in any pixel are dropped
before the analysis method runs, and the resulting components are re-embedded
into the full channel range (RamanSPy's Raman data doesn't need this).
"""
from typing import List, Callable, Union, Tuple
import numpy as np
from numpy.typing import NDArray

from .. import utils
from .. import core


class AnalysisStep:
    """
    A class that defines analysis logic.

    Encapsulate projection-based analysis methods (e.g. decomposition, clustering, spectral unmixing).
    """

    def __init__(self, method: Callable, *args, **kwargs):
        # TODO: check if function of the exact type needed
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def apply(self, spectral_objects: Union[core.SpectralObject, List[core.SpectralObject]]) -> \
            Tuple[Union[List[NDArray], List[List[NDArray]]], List[NDArray]]:
        """
        Applies the defined analysis method on the Brillouin spectroscopic objects provided.

        The single point-of-contact method of analysis methods.

        Data is first flattened and stacked, then the method is applied, and projections are refolded into the shape of the original data.

        Parameters
        ----------
        spectral_objects : Union[core.SpectralObject, List[core.SpectralObject]]
            The data to analyse, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].

        Returns
        ----------
        List[numpy.array] or List[List[numpy.array]] :
            The projected data.
            For each object in ``spectral_objects``, a list of length equal to the dimensionality of the projective space is derived,
            containing the corresponding projection maps.
        List[numpy.array] :
            The components derived.


        Example
        ----------

        .. code::

            # once an analysis method is initialised, it can be applied to different Brillouin data
            projections, components = analysis_method.apply(brillouin_object)
            projections, components = analysis_method.apply([brillouin_object, brillouin_spectrum, brillouin_image])
        """

        if not isinstance(spectral_objects, list):
            spectral_objects = [spectral_objects]

        if not utils.is_aligned(spectral_objects):
            ValueError("Cannot perform analysis step on unaligned spectra. Spectral axis must match.")

        # Unfold data, tracking NaN/masked values per spectral channel
        spectral_data = []
        invalid_masks = []
        for spectral_object in spectral_objects:
            data = spectral_object.flat.spectral_data
            if isinstance(data, np.ma.MaskedArray):
                invalid = np.ma.getmaskarray(data) | np.isnan(np.ma.filled(data, np.nan))
                data = np.ma.filled(data, np.nan)
            else:
                data = np.asarray(data)
                invalid = np.isnan(data)
            spectral_data.append(data)
            invalid_masks.append(invalid)
        spectral_data = np.vstack(spectral_data)
        invalid_mask = np.vstack(invalid_masks)

        # Decomposition/unmixing methods need a value for every pixel in every channel, so
        # channels that are NaN/masked for at least one pixel (e.g. a removed IRF region) are
        # dropped entirely rather than filled in, to avoid biasing the result with fake signal.
        valid_channels = ~invalid_mask.any(axis=0)
        if not valid_channels.any():
            raise ValueError(
                "Every spectral channel contains a NaN/masked value in at least one of the spectra provided; "
                "nothing left to analyse.")
        spectral_data = spectral_data[:, valid_channels]

        # apply method
        projections, components = self.method(spectral_data, *self.args, **self.kwargs)

        # Re-embed components/endmembers into the full channel range (NaN for dropped
        # channels) so they keep aligning with the original (un-cropped) spectral_axis.
        full_components = np.full((components.shape[0], valid_channels.shape[0]), np.nan)
        full_components[:, valid_channels] = components
        components = [full_components[i, ...] for i in range(full_components.shape[0])]

        # Fold data
        projections_folded = []
        for spectral_object in spectral_objects:
            current_projection = projections[:spectral_object.flat.shape[0]]
            current_projection = current_projection.reshape(list(spectral_object.shape) + [projections.shape[-1]])
            current_projection = [current_projection[..., i] for i in range(current_projection.shape[-1])]

            projections_folded.append(current_projection)

            projections = projections[spectral_object.flat.shape[0]:]

        if len(projections_folded) == 1:
            return projections_folded[0], components

        return projections_folded, components


def variance_explained(spectral_object, step_factory: Callable[[int], AnalysisStep], param_values):
    """
    Computes the fraction of variance in the data explained by the reconstruction
    (``projections @ components``) of an :class:`AnalysisStep`, across a range of
    component/endmember/cluster counts - a model-agnostic way to decide how many are
    actually needed. This works for any decomposition/unmixing/clustering method
    (:mod:`~brillouinpy.analysis.decompose`, :mod:`~brillouinpy.analysis.unmix`,
    :class:`~brillouinpy.analysis.cluster.KMeans`) since they all share the same
    ``apply() -> (projections, components)`` interface.

    Parameters
    ----------
    spectral_object : core.SpectralObject
        The data to evaluate (a single object, not a list).
    step_factory : Callable[[int], AnalysisStep]
        Builds the analysis step for a given count, e.g.
        ``lambda n: brillouinpy.analysis.decompose.PCA(n_components=n)``,
        ``lambda n: brillouinpy.analysis.unmix.VCA(n_endmembers=n)`` or
        ``lambda n: brillouinpy.analysis.cluster.KMeans(n_clusters=n, random_state=0)``.
    param_values : Iterable[int]
        The component/endmember/cluster counts to evaluate.

    Returns
    -------
    dict[int, float]
        Fraction of variance explained (in ``[0, 1]``, higher is better) for each value
        in ``param_values``. Plot this against ``param_values`` and look for the "elbow"
        where additional components stop meaningfully improving the reconstruction,
        rather than guessing ``n_components``/``n_endmembers``/``n_clusters`` blind.

    Example
    ----------

    .. code::

        variances = bp.analysis.variance_explained(
            image,
            lambda n: bp.analysis.decompose.NMF(n_components=n, init='nndsvda'),
            param_values=range(1, 6),
        )
        plt.plot(list(variances.keys()), list(variances.values()), marker='o')
    """
    data = spectral_object.flat.spectral_data
    original = np.ma.filled(data, np.nan) if np.ma.is_masked(data) else np.asarray(data)
    original = original.astype(float)

    results = {}
    for n in param_values:
        projections, components = step_factory(n).apply(spectral_object)

        projections_matrix = np.stack([np.asarray(p).reshape(-1) for p in projections], axis=1)
        components_matrix = np.stack([np.asarray(c) for c in components], axis=0)
        reconstruction = projections_matrix @ components_matrix

        # Some methods (e.g. PCA) reconstruct data around its per-channel mean rather
        # than its absolute intensity; absorbing any constant per-channel offset here
        # (equivalent to fitting an intercept per channel) keeps the metric comparable
        # across methods without needing to know which ones centre their data.
        residual = original - reconstruction
        residual = residual - np.nanmean(residual, axis=0)
        ss_res = np.nansum(residual ** 2)
        ss_tot = np.nansum((original - np.nanmean(original, axis=0)) ** 2)
        results[n] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return results
