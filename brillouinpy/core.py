# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:19:25 2025

@author: Timm

The SpectralContainer/Spectrum/SpectralImage/SpectralVolume class hierarchy below
is adapted from RamanSPy (https://github.com/barahona-research-group/RamanSPy),
Copyright (c) 2023 Dimitar Georgiev, licensed under the BSD 3-Clause License (see
LICENSE and THIRD_PARTY_LICENSES/RamanSPy-BSD-3-Clause.txt in the repository
root). Notable changes: masked-array/NaN support throughout (RamanSPy's Raman
data doesn't need it), and Brillouin-specific docstring/unit updates.
"""

from __future__ import annotations  # default if Python >= 3.10
import copy
from numbers import Number
import os
import pickle
from typing import List, Union
import numpy as np
from scipy.signal import find_peaks

# from . import plot
from . import utils


#: Sentinel for "carry this forward from the source object" vs. an explicit
#: ``None`` (which means "drop it") in derivation helpers.
_UNSET = object()


def _empty_px_size() -> dict:
    return {"x": None, "y": None, "z": None}


def _create_data(spectral_data, spectral_axis, *, metadata=None, px_size_um=None, channels=None,
                 instrument_response_function=None):
    if len(spectral_data.shape) == 1:
        cls = Spectrum
    elif len(spectral_data.shape) == 3:
        cls = SpectralImage
    elif len(spectral_data.shape) == 4:
        cls = SpectralVolume
    else:
        cls = SpectralContainer
    return cls(spectral_data, spectral_axis, metadata=metadata, px_size_um=px_size_um, channels=channels,
              instrument_response_function=instrument_response_function)


def _stack_irf(sources):
    """Combine the per-source instrument response functions when stacking. Returns
    a stacked ``(N, B)`` array if every source carries a grid-conformant per-pixel
    IRF, the shared 1-D kernel if every source carries the same one, else ``None``."""
    irfs = [getattr(s, "instrument_response_function", None) for s in sources]
    if any(irf is None for irf in irfs):
        return None
    irfs = [np.asarray(irf) for irf in irfs]
    if all(irf.ndim == 1 for irf in irfs):
        return irfs[0].copy() if all(np.array_equal(irf, irfs[0]) for irf in irfs) else None
    try:
        return np.vstack([irf.reshape(-1, irf.shape[-1]) if irf.ndim > 1
                          else np.broadcast_to(irf, (s.flat.shape[0], irf.shape[-1]))
                          for irf, s in zip(irfs, sources)])
    except ValueError:
        return None


class SpectralContainer:
    """
    The base class that settles as the backbone of the package. It encapsulates a spectral data
    container of an arbitrary dimension and defines relevant behaviour and information.

    Parameters
    ----------
    spectral_data : array_like of shape (x, y, z, ..., B)
        The intensity values to store. Last dimension must be the spectral dimension.
    spectral_axis : array_like of shape (B, )
        The Brillouin spectral axis (typically the frequency shift, in GHz). Order and length must match the last dimension of ``spectral_data``.
    metadata : dict, optional
        Free-form acquisition metadata carried alongside the data. Populated e.g.
        by :func:`brillouinpy.io.from_brim` (as ``{category: {attribute: (value,
        units)}}``) and read back by :func:`brillouinpy.io.to_brim`. Always a dict
        on the instance (never ``None``); derived objects (slices, ``flat``,
        ``mean``, ...) inherit a copy of it.
    px_size_um : dict, optional
        Real-world pixel spacing as ``{"x"/"y"/"z": float_or_None}`` in
        micrometers. Always present on the instance (missing axes are ``None``).
    channels : dict, optional
        Additional co-acquired data as ``{name: SpectralObject}`` - e.g. a Raman
        or fluorescence channel taken on the same sample. The primary
        ``spectral_data`` stays the Brillouin data; channels are a generic
        side-car and brillouinpy makes no assumption about what they contain.
        Always a dict on the instance. A channel whose spatial ``shape`` matches
        this object's is "grid-conformant" and follows along through spatial
        operations (indexing, ``flat``, stacking); one that doesn't is passed
        through untouched. See :attr:`channels_grid_conformant`.
    instrument_response_function : array_like, optional
        The spectrometer's instrument response function, aligned to
        ``spectral_axis``: either a 1-D array of shape ``(B,)`` shared by every
        pixel (e.g. a VIPA IRF measured once), or a grid-conformant
        ``(*spatial, B)`` array giving a per-pixel IRF (e.g. a tandem
        Fabry-Perot elastic peak detected on every scan by
        :class:`brillouinpy.preprocessing.misc.Deconvoluter_IRF`). ``None`` if
        unknown. Carried forward through spatial operations (``flat``, indexing,
        stacking follow it; ``mean``/``variance`` collapse it to the mean IRF)
        and consumed by :class:`brillouinpy.analysis.fitmodel.DHO` with
        ``irf='auto'``. For an IRF measured a few times over the course of an
        acquisition (VIPA drift correction), expand it to a per-pixel array with
        :func:`brillouinpy.preprocessing.misc.assign_irf` first.

    Example
    ----------

    .. code::

        import numpy as np
        from brillouinpy import SpectralContainer

        spectral_data = np.random.rand(20, 1500)
        spectral_axis = np.linspace(-20, 20, 1500)  # frequency shift in GHz

        brillouin_object = SpectralContainer(spectral_data, spectral_axis)
    """
    #: Number of dimensions ``spectral_data`` must have for this class (spatial
    #: dimensions + the trailing spectral one). ``None`` on the base container,
    #: which accepts any dimensionality; set on the shape-specific subclasses.
    _required_ndim = None

    def __init__(self, spectral_data, spectral_axis, *, metadata=None, px_size_um=None, channels=None,
                 instrument_response_function=None):
        # Convert to masked array if it isn't already one
        if np.ma.is_masked(spectral_data):
            self.spectral_data = spectral_data
        else:
            self.spectral_data = np.ma.asarray(spectral_data)

        self.spectral_axis = np.asarray(spectral_axis)
        self.instrument_response_function = (
            None if instrument_response_function is None
            else np.asarray(instrument_response_function))
        self.metadata = {} if metadata is None else dict(metadata)
        self.channels = {} if channels is None else dict(channels)
        self.px_size_um = _empty_px_size()
        if px_size_um:
            self.px_size_um.update(px_size_um)

        if self._required_ndim is not None and self.spectral_data.ndim != self._required_ndim:
            raise ValueError(
                f"{type(self).__name__} requires {self._required_ndim}-dimensional data "
                f"({self._required_ndim - 1} spatial + 1 spectral); got {self.spectral_data.ndim} "
                f"dimensions (shape {self.spectral_data.shape}). Use a different container class "
                f"or SpectralContainer for this shape.")

        if self.spectral_data.shape[-1] != len(self.spectral_axis):
            raise ValueError(
                f"The last dimension of the data ({self.spectral_data.shape[-1]}) must match the axis provided ({len(self.spectral_axis)}).")

        irf = self.instrument_response_function
        if irf is not None:
            if irf.shape[-1] != len(self.spectral_axis):
                raise ValueError(
                    f"instrument_response_function's last dimension ({irf.shape[-1]}) must match the "
                    f"spectral axis ({len(self.spectral_axis)}).")
            if irf.ndim != 1 and irf.shape[:-1] != self.spectral_data.shape[:-1]:
                raise ValueError(
                    f"a per-pixel instrument_response_function must be 1-D (shape (B,), shared) or "
                    f"match the data's spatial shape; got {irf.shape} for data {self.spectral_data.shape}.")

        # Order data and axis by shift number values
        sorted_indices = self.spectral_axis.argsort()
        self.spectral_data = self.spectral_data[..., sorted_indices]
        self.spectral_axis = self.spectral_axis[sorted_indices]
        if self.instrument_response_function is not None:
            self.instrument_response_function = self.instrument_response_function[..., sorted_indices]

    def __setstate__(self, state):
        # Backwards compatibility: pickles written before metadata/px_size_um
        # existed as attributes.
        self.__dict__.update(state)
        self.__dict__.setdefault("instrument_response_function", None)
        self.__dict__.setdefault("metadata", {})
        self.__dict__.setdefault("channels", {})
        self.__dict__.setdefault("px_size_um", _empty_px_size())

    def __repr__(self):
        parts = [f"shape={self.shape}", f"spectral_length={self.spectral_length}"]
        if self.channels:
            chans = ", ".join(f"{k}: {tuple(getattr(v, 'shape', ()))}" for k, v in self.channels.items())
            parts.append(f"channels={{{chans}}}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @property
    def channels_grid_conformant(self) -> dict:
        """
        Maps each channel name to whether its spatial ``shape`` matches this
        object's - i.e. whether it follows along through spatial operations
        (indexing, ``flat``, stacking). Non-conformant channels are passed
        through untouched.
        """
        return {name: getattr(ch, "shape", None) == self.shape for name, ch in self.channels.items()}

    def _map_channels(self, op) -> dict:
        """Apply ``op`` to grid-conformant channels; pass the rest through unchanged."""
        conformant = self.channels_grid_conformant
        return {name: op(ch) if conformant[name] else ch for name, ch in self.channels.items()}

    def _child_irf(self, *, key=None, flatten=False, reduce=None):
        """Carry :attr:`instrument_response_function` through a spatial operation. A
        shared 1-D kernel passes through unchanged (it broadcasts); a per-pixel
        ``(*spatial, B)`` array is sliced (``key``), flattened to ``(-1, B)``
        (``flatten``) or collapsed to a single 1-D kernel (``reduce='mean'``)."""
        irf = self.instrument_response_function
        if irf is None:
            return None
        irf = np.asarray(irf)
        if irf.ndim <= 1:
            return irf.copy()
        if reduce == 'mean':
            return np.nanmean(irf.reshape(-1, irf.shape[-1]), axis=0)
        if flatten:
            return irf.reshape(-1, irf.shape[-1]).copy()
        if key is not None:
            return np.asarray(irf[key])
        return irf.copy()

    def _derive(self, spectral_data, spectral_axis=None, *, keep_px_size=True, channels=None,
                instrument_response_function=_UNSET):
        """
        Build a derived object of the appropriate type for ``spectral_data``,
        carrying this object's metadata (and, by default, pixel size and IRF)
        forward. ``channels`` defaults to a deep copy of this object's channels;
        pass an explicit dict when a spatial operation has transformed them.
        ``instrument_response_function`` defaults to this object's, carried as-is;
        pass an explicit value (or ``None``) when a spatial operation transforms
        or invalidates it.
        """
        return _create_data(
            spectral_data,
            self.spectral_axis if spectral_axis is None else spectral_axis,
            metadata=copy.deepcopy(self.metadata),
            px_size_um=dict(self.px_size_um) if keep_px_size else None,
            channels=copy.deepcopy(self.channels) if channels is None else channels,
            instrument_response_function=(self._child_irf() if instrument_response_function is _UNSET
                                          else instrument_response_function),
        )

    @staticmethod
    def _inherited_kwargs(source: SpectralContainer, *, keep_px_size: bool = True) -> dict:
        """Constructor kwargs that carry ``source``'s metadata/pixel size forward."""
        return {
            "metadata": copy.deepcopy(source.metadata),
            "px_size_um": dict(source.px_size_um) if keep_px_size else None,
        }

    @staticmethod
    def _stack_channels(sources: List[SpectralContainer], stack_fn) -> dict:
        """
        Merge channels across a list of objects being stacked, using ``stack_fn``
        (a ``from_stack``/``from_image_stack``-style classmethod). Channels are
        kept only when every source carries the same channel names and each of
        those channels is grid-conformant to its source; otherwise the stacked
        result carries no channels.
        """
        if not sources or not sources[0].channels:
            return {}
        names = set(sources[0].channels)
        if any(set(s.channels) != names for s in sources):
            return {}
        merged = {}
        for name in names:
            group = [s.channels[name] for s in sources]
            if any(getattr(ch, "shape", None) != s.shape for ch, s in zip(group, sources)):
                return {}
            merged[name] = stack_fn(group)
        return merged

    def peaks(self, *, height=None, threshold=None, distance=None, prominence=None,
              width=None, wlen=None, rel_height=0.5, plateau_size=None):
        """
        Finds peaks in the spectrum's intensity data.

        A thin wrapper around :func:`scipy.signal.find_peaks`; all keyword arguments are
        passed through unchanged, see its documentation for details. For an object with
        spatial dimensions the peaks are found in its :attr:`mean` spectrum.

        Returns
        -------
        peaks : numpy.ndarray
            Indices of the detected peaks (into ``spectral_axis``).
        properties : dict
            The peak properties computed by :func:`scipy.signal.find_peaks`.
        """
        intensity = self.spectral_data if self.spectral_data.ndim == 1 else self.mean.spectral_data
        return find_peaks(intensity, height=height, threshold=threshold, distance=distance,
                          prominence=prominence, width=width, wlen=wlen, rel_height=rel_height,
                          plateau_size=plateau_size)

    def save(self, filename: str, directory: str = None):
        """
        Save the spectral object to a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to save the spectral object to.
        directory : str, optional
            The name of the directory to save the file in. Must be the full path to the directory or the path relative
            to the working directory. If not provided (default), the file will be saved in the working directory.
        """
        full_filename = os.path.join(directory, filename) if directory is not None else filename
        with open(full_filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        """
        Load a spectral object from a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to load a spectral object from. Must be the full path or the path relative to the working directory.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_stack(cls, stack: List[Spectrum]) -> SpectralContainer:
        """
        Returns the combined Brillouin object defined by stacking the collection of individual spectra given.

        The spectral axes of the spectra provided must match.
        """
        if not utils.is_aligned(stack):
            raise ValueError("Cannot stack unaligned spectral objects. Spectral axes must match.")

        return cls(np.vstack([obj.flat.spectral_data for obj in stack]), stack[0].spectral_axis,
                   channels=cls._stack_channels(stack, SpectralContainer.from_stack),
                   instrument_response_function=_stack_irf(stack),
                   **cls._inherited_kwargs(stack[0]))

    @property
    def flat(self) -> SpectralContainer:
        """
        Flatten all spatial dimensions of the spectral object into a single one.

        Returns
        -------
        numpy.ndarray of shape (dim_1*dim_2*...*dim_n, B)
        """
        return SpectralContainer(self.spectral_data.reshape(-1, self.spectral_length), self.spectral_axis,
                                 metadata=copy.deepcopy(self.metadata), px_size_um=dict(self.px_size_um),
                                 channels=self._map_channels(lambda ch: ch.flat),
                                 instrument_response_function=self._child_irf(flatten=True))

    @property
    def shape(self) -> tuple[int]:
        """
        Returns the (spatial) shape of the spectral object (i.e. without the last dimension).
        """
        return self.spectral_data.shape[:-1] if len(self.spectral_data.shape) >= 2 else (1,)

    @property
    def spectral_length(self) -> int:
        """
        Returns the spectral length B of the spectral object.
        """
        return len(self.spectral_axis)

    @property
    def mean(self) -> Spectrum:
        """
        Returns the mean spectrum in the spectral object.
        """
        return Spectrum(np.nanmean(self.flat.spectral_data, axis=0), self.spectral_axis,
                        metadata=copy.deepcopy(self.metadata),
                        instrument_response_function=self._child_irf(reduce='mean'))

    @property
    def variance(self) -> Spectrum:
        """
        Returns the mean spectrum in the spectral object.
        """
        return Spectrum(np.nanvar(self.flat.spectral_data, axis=0), self.spectral_axis,
                        metadata=copy.deepcopy(self.metadata),
                        instrument_response_function=self._child_irf(reduce='mean'))

    # TODO: spatial vs spectral indexing
    def __getitem__(self, key):
        """
        Indexes the object along its spatial dimension(s).

        Only spatial indexing is supported (e.g. ``image[0]`` or ``volume[0, :, 2]``); a
        :class:`SpectralContainer` holding a single spectrum has no spatial dimension to
        index into and raises :class:`ValueError`. To select a slice of the spectral axis
        instead, use the :class:`~brillouinpy.preprocessing.misc.Cropper` preprocessing
        step.

        Parameters
        ----------
        key
            A standard numpy index/slice applied to the spatial dimension(s).

        Returns
        -------
        SpectralObject or numpy.ndarray
            A new spectral object of the appropriate type for the resulting shape (see
            :func:`_create_data`), or a bare intensity value if the index selects a single
            spatial point and spectral channel.
        """
        if self.shape == (1,):
            raise ValueError(
                "Only spatial indexing is supported. To index spectrally, use the brillouinpy.preprocessing.misc.Cropper class.")

        spectral_data_slice = self.spectral_data[key]

        if len(spectral_data_slice.shape) == 0:
            return spectral_data_slice
        else:
            return self._derive(spectral_data_slice,
                                channels=self._map_channels(lambda ch: ch[key]),
                                instrument_response_function=self._child_irf(key=key))

    def band(self, spectral_band: Number) -> np.ndarray:
        """Returns a spectral slice across the closest spectral band in the axis to the one given."""

        # Check if band given is within the spectral axis
        min_band = self.spectral_axis.min()
        max_band = self.spectral_axis.max()
        if not (min_band <= spectral_band <= max_band):
            raise ValueError(
                f"Band ({spectral_band}) must be within the bounds of the spectral axis ([{min_band}, {max_band}])")

        # Find the closest band in the axis to the one provided
        closest_band_index = np.argmin(np.abs(self.spectral_axis - spectral_band))

        # Return the slice across that band
        return self.spectral_data[..., closest_band_index]

    def tolist(self) -> list[Spectrum]:
        """
        Returns the spectral object as a list of Spectrum objects.
        """
        unfolded_spectral_data = self.spectral_data.reshape(-1, self.spectral_length)
        irf = self._child_irf(flatten=True)
        irf_rows = irf if (irf is not None and irf.ndim == 2) else [irf] * len(unfolded_spectral_data)

        return [Spectrum(spectral_data, self.spectral_axis, metadata=copy.deepcopy(self.metadata),
                         instrument_response_function=irf_row)
                for spectral_data, irf_row in zip(unfolded_spectral_data, irf_rows)]


class Spectrum(SpectralContainer):
    """
    The :class:`Spectrum` class defines a 1D spectroscopic signal of an arbitrary spectral length.

    Example
    ----------

    .. code::

        import numpy as np
        from brillouinpy import Spectrum

        spectral_data = np.random.rand(1500)
        spectral_axis = np.linspace(-20, 20, 1500)  # frequency shift in GHz

        brillouin_spectrum = Spectrum(spectral_data, spectral_axis)
    """

    _required_ndim = 1


class SpectralImage(SpectralContainer):
    """
    The :class:`SpectralImage` class defines a 2D spectroscopic image. Dimensions must be in the order of
    (x, y, B).

    Example
    ----------

    .. code::

        import numpy as np
        from brillouinpy import SpectralImage

        spectral_data = np.random.rand(50, 50, 1500)
        spectral_axis = np.linspace(-20, 20, 1500)  # frequency shift in GHz

        brillouin_image = SpectralImage(spectral_data, spectral_axis)
    """

    _required_ndim = 3

    # def plot(self, bands: Union[Number, List[Number]], **kwargs):
    #     """
    #     Plots the spectral image slice(s) across the spectral image, defined by the band(s) provided (using the closest
    #     band(s) in the spectral axis of the image to the one(s) given).

    #     Parameters
    #     ----------
    #     bands : Number or List[Number]
    #         The spectral bands to plot across.
    #     **kwargs : keyword arguments, optional,
    #         Check the :meth:`brillouinpy.plot.image' method for a list of keyword parameters.
    #     """
    #     if isinstance(bands, Number):
    #         bands = [bands]

    #     spectral_slices = [self.band(band) for band in bands]
    #     kwargs['cbar_label'] = [f"{cbar_label} ({band} GHz)" for cbar_label, band in
    #                             zip(itertools.repeat(kwargs.pop('cbar_label', 'Peak intensity'), len(spectral_slices)), bands)]

    #     return plot.image(spectral_slices, **kwargs)


class SpectralVolume(SpectralContainer):
    """
    The :class:`SpectralVolume` class defines a 3D spectroscopic volume. Dimensions must be in the order of
    (x, y, z, B).

    Example
    ----------

    .. code::

        import numpy as np
        from brillouinpy import SpectralVolume

        spectral_data = np.random.rand(50, 50, 10, 1500)
        spectral_axis = np.linspace(-20, 20, 1500)  # frequency shift in GHz

        brillouin_volume = SpectralVolume(spectral_data, spectral_axis)
    """

    _required_ndim = 4

    @classmethod
    def from_image_stack(cls, image_stack: List[SpectralImage]) -> SpectralVolume:
        """
        Returns the volumetric Brillouin object defined by z-stacking the collection of spectral images given.

        All dimensions of the spectral images must match, as well as their spectral axes.
        """
        if not utils.is_aligned(image_stack):
            raise ValueError("Cannot create a spectral volume out of unaligned spectral images. Spectral axes must match.")

        irfs = [im.instrument_response_function for im in image_stack]
        if any(x is None for x in irfs):
            stacked_irf = None
        elif all(np.asarray(x).ndim == 1 for x in irfs):
            stacked_irf = irfs[0] if all(np.array_equal(x, irfs[0]) for x in irfs) else None
        elif all(np.asarray(x).ndim == 3 and np.asarray(x).shape[:2] == im.shape
                 for x, im in zip(irfs, image_stack)):
            stacked_irf = np.stack([np.asarray(x) for x in irfs], axis=2)  # (nx, ny, nz, B)
        else:
            stacked_irf = None

        return cls(np.dstack([image.spectral_data[..., np.newaxis, :] for image in image_stack]),
                   image_stack[0].spectral_axis,
                   channels=cls._stack_channels(image_stack, SpectralVolume.from_image_stack),
                   instrument_response_function=stacked_irf,
                   **cls._inherited_kwargs(image_stack[0]))

    # def plot(self, bands, **kwargs):
    #     """
    #     Plots the spectral volume slice(s) across the spectral volume, defined by the band(s) provided (using the closest
    #     band(s) in the spectral axis of the image to the one(s) given).

    #     Parameters
    #     ----------
    #     bands : Number or List[Number]
    #         The spectral bands to plot across.
    #     **kwargs : keyword arguments, optional,
    #         Check the :meth:`brillouinpy.plot.volume' method for a list of keyword parameters.
    #     """
    #     if isinstance(bands, Number):
    #         bands = [bands]

    #     spectral_slices = [self.band(band) for band in bands]
    #     kwargs['cbar_label'] = [f"{cbar_label} ({band} GHz)" for cbar_label, band in
    #                             zip(itertools.repeat(kwargs.pop('cbar_label', 'Peak intensity'), len(spectral_slices)), bands)]

    #     return plot.volume(spectral_slices, **kwargs)

    def layer(self, layer_index: int) -> SpectralImage:
        """Returns the :class:`SpectralImage` layer specified by the given index as a SpectralImage. Index must be between 0 and z dimension - 1."""
        if not (0 <= layer_index <= self.shape[-1] - 1):
            raise ValueError(
                f"The layer index must be between 0 and {self.shape[-1] - 1} inclusively. Got {layer_index} instead.")

        irf = self.instrument_response_function
        layer_irf = None if irf is None else (irf if irf.ndim == 1 else np.asarray(irf)[..., layer_index, :])
        return SpectralImage(self.spectral_data[..., layer_index, :], self.spectral_axis,
                             metadata=copy.deepcopy(self.metadata),
                             px_size_um={"x": self.px_size_um.get("x"), "y": self.px_size_um.get("y")},
                             instrument_response_function=layer_irf)


# for typing
SpectralObject = Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume]
