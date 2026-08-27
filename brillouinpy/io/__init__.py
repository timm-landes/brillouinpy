# -*- coding: utf-8 -*-
"""
Reading/writing spectral data: acquisition-side loaders (this group's raw-data
layout and older/other formats) and interoperability with external file formats
and packages.

Submodules
----------
tfp
    Loading data from this group's tandem Fabry-Perot (TFP) interferometer setup -
    the actively used/maintained loading path. Also a worked example to model a
    custom loader on, if your own setup uses a different raw-data layout.
legacy
    Superseded loaders, kept for reading old datasets.
multimodal
    Loading data for modalities other than the group's primary TFP-Brillouin setup
    (currently just Raman).
export
    Interoperability with external formats/packages (``brim``, ``HDF5_BLS``, ...).
    The most commonly used functions are re-exported directly on ``brillouinpy.io``
    for convenience (e.g. ``brillouinpy.io.to_brim(...)``).
"""
from . import tfp
from . import legacy
from . import multimodal
from . import export

from .tfp import (  # noqa: F401
    read_meta,
    brillouin_spectral_axis_from_meta,
    prepare_brillouin_data,
)
from .export import (  # noqa: F401
    to_brim,
    from_brim,
    list_brim_measurements,
    to_hdf5_bls,
    from_hdf5_bls,
    list_measurements,
)

__all__ = [
    "tfp",
    "legacy",
    "multimodal",
    "export",
    "read_meta",
    "brillouin_spectral_axis_from_meta",
    "prepare_brillouin_data",
    "to_brim",
    "from_brim",
    "list_brim_measurements",
    "to_hdf5_bls",
    "from_hdf5_bls",
    "list_measurements",
]
