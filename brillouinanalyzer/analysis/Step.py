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
