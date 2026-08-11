from typing import Union, Callable
import copy
from typing import final, List
import numpy as np

from ..core import SpectralObject


class PreprocessingStep:
    """
    A class that defines preprocessing logic.

    Encapsulate preprocessing methods that transform the intensity values and spectral axis of Brillouin data.

    To define a preprocessing procedure that can be applied to any Brillouin spectroscopic data, you must wrap a predefined
    preprocessing method using this class, which in turn streamlines any consecutive operations.

    Parameters
    ----------
    method : Callable
        A Callable object (e.g. a method) which defines how the preprocessing step alters spectral objects. Its ``__call__`` method
        must have signature of the form: ``__call__(intensity_data, spectral_axis, *args, **kwargs)``, where ``intensity_data``
        is an ndarray of arbitrary shape defining the intensity values to process, whose last axis is the spectral axis,
        ``spectral_axis`` - a 1D ndarray defining the Brillouin spectral axis to process (typically the frequency shift, in GHz),
        ``*args`` - other positional arguments, and ``**kwargs`` - other keyword arguments.
    **kwargs :
        Any keyword arguments the Callable needs in its ``__call__`` method.


    .. note:: One has to use the :class:`PreprocessingStep` class only when devising and integrating custom preprocessing methods (check :ref:`Custom algorithms`).

              All preprocessing methods built into `brillouinanalyzer` can be directly accessed and used as indicated in :ref:`Built-in preprocessing methods`.

    Example
    ----------

    .. code::

        from brillouinanalyzer import preprocessing

        # Defining some preprocessing function of the correct type
        def preprocessing_func(intensity_data, spectral_axis, **kwargs):
            # Preprocess intensity_data and spectral axis
            ...

            return updated_intensity_data, updated_spectral_axis

        # wrapping the function into a PreprocessingStep object together with the relevant *args and **kwargs
        preprocessing_method = preprocessing.PreprocessingStep(preprocessing_func, **kwargs)
    """

    def __init__(self, method: Callable, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, spectral_data, spectral_axis, *args, **kwargs):
        return self.method(spectral_data, spectral_axis, *args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(kwargs:{self.kwargs}"

    @final
    def _process_object(self, spectral_object: SpectralObject) -> SpectralObject:
        new_spectral_object = copy.deepcopy(spectral_object)

        if np.ma.is_masked(new_spectral_object.spectral_data):
            # Keep track of the original mask
            original_mask = new_spectral_object.spectral_data.mask
            # Fill masked values with NaN for processing
            spectral_data = new_spectral_object.spectral_data.filled(np.nan)
        else:
            original_mask = None
            spectral_data = new_spectral_object.spectral_data

        # Process the data
        preprocessed_spectral_data, preprocessed_spectral_axis = self(
            spectral_data, new_spectral_object.spectral_axis, **self.kwargs)

        # Restore masked array properties
        if original_mask is not None:
            preprocessed_spectral_data = np.ma.array(
                preprocessed_spectral_data,
                mask=original_mask,
                fill_value=np.nan
            )

        new_spectral_object.spectral_data = preprocessed_spectral_data
        new_spectral_object.spectral_axis = preprocessed_spectral_axis

        return new_spectral_object

    @final
    def apply(self, spectral_objects: Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]) -> \
            Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]:
        """
        Applies the defined preprocessing method on the Brillouin spectroscopic objects provided.

        The single point-of-contact method of :class:`brillouinanalyzer.preprocessing.PreprocessingStep` instances.

        Method is applied on each data container instance provided individually.


        Parameters
        ----------
        spectral_objects : Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]
            The objects to preprocess, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].


        Returns
        -------
        Union[SpectralObject, List[Union[SpectralObject, List[SpectralObject]]]]
            The preprocessed objects, where SpectralObject := Union[SpectralContainer, Spectrum, SpectralImage, SpectralVolume].


        .. note:: When more than one class:`brillouinanalyzer.SpectralContainer` is passed, preprocessing methods are applied individually for each instance passed.


        Example
        ----------

        .. code::

            # once a preprocessing method is initialised, it can be applied to different Brillouin data
            preprocessed_data = preprocessing_method.apply(brillouin_object)
            preprocessed_data = preprocessing_method.apply([brillouin_object, brillouin_spectrum, brillouin_image])
            preprocessed_data = preprocessing_method.apply([brillouin_object, brillouin_spectrum], brillouin_object, [brillouin_spectrum, brillouin_image])
        """
        if isinstance(spectral_objects, list):
            return [self.apply(spectral_object) for spectral_object in spectral_objects]
        else:
            return self._process_object(spectral_objects)
