from . import Pipeline
from . import denoise, despike, normalise


def template(normalisation_pixelwise: bool = True) -> Pipeline:
    """
    Template for a standard Brillouin preprocessing recipe.

    Swap or reorder the steps below to build a project-specific pipeline. Check
    :mod:`brillouinanalyzer.preprocessing.denoise`, :mod:`~.despike`, :mod:`~.normalise`
    and :mod:`~.misc` for the available building blocks.

    Parameters
    ----------
    normalisation_pixelwise: bool, optional
        Whether to apply normalisation for each pixel individually or not. Default is ``True``.

    Example
    ----------

    .. code::

        pipeline = preprocessing.protocols.template()
        preprocessed_data = pipeline.apply(data)
    """
    return Pipeline([
        despike.WhitakerHayes(),
        denoise.Gaussian(),
        normalise.AUC(pixelwise=normalisation_pixelwise),
    ])
