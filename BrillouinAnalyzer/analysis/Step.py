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

        # Unfold data
        spectral_data = []
        for spectral_object in spectral_objects:
            data = spectral_object.flat.spectral_data
            # Only handle masked arrays, keep normal arrays unchanged
            if isinstance(data, np.ma.MaskedArray):
                # Handle masked arrays by filling with 0
                data = data.filled(0)
            spectral_data.append(data)
        spectral_data = np.vstack(spectral_data)

        # apply method
        projections, components = self.method(spectral_data, *self.args, *self.kwargs)

        components = [components[i, ...] for i in range(components.shape[0])]

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
