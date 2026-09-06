import numpy as np
import scipy.linalg as splin
from scipy.optimize import nnls as _scipy_nnls
import functools
from typing import Literal

from .Step import AnalysisStep


def _ucls(spectral_data, endmembers):
    """Unconstrained least squares abundances: solve ``endmembers.T @ a = pixel``
    for every pixel at once. Pure NumPy - no pysptools dependency."""
    solution, *_ = np.linalg.lstsq(endmembers.T, spectral_data.T, rcond=None)
    return solution.T


def _nnls(spectral_data, endmembers):
    """Non-negative least squares abundances (``scipy.optimize.nnls``, one pixel
    at a time - matches pysptools' per-pixel NNLS, no extra dependency)."""
    M = endmembers.T  # (bands, n_endmembers)
    return np.array([_scipy_nnls(M, pixel)[0] for pixel in spectral_data])


def _fcls(spectral_data, endmembers, delta=1 / 1000):
    """
    Fully-constrained least squares abundances (non-negative, sum-to-one), via
    the standard NNLS-with-an-augmented-row trick (Heinz & Chang, 2001): appending
    a heavily-weighted "abundances must sum to 1" equation to the NNLS system
    enforces both constraints using only ``scipy.optimize.nnls``.
    """
    n_endmembers = endmembers.shape[0]
    M = np.vstack([delta * endmembers.T, np.ones((1, n_endmembers))])  # (bands+1, n_endmembers)
    ones_row = np.ones((spectral_data.shape[0], 1))
    augmented_data = np.hstack([delta * spectral_data, ones_row])
    return np.array([_scipy_nnls(M, pixel)[0] for pixel in augmented_data])


"""
List of the available methods for calculating fractional abundances.
"""
abundance_methods = {
    'ucls': _ucls,
    'nnls': _nnls,
    'fcls': _fcls,
}


def unmixer(endmember_func):
    @functools.wraps(endmember_func)
    def wrap(spectral_data, n_endmembers, abundance_method):
        endmembers = endmember_func(spectral_data, n_endmembers)

        abundance_method_ = abundance_methods.get(abundance_method, None)
        if abundance_method_ is None:
            raise ValueError(
                f"{abundance_method} is not a valid abundance method. Possible methods are {abundance_methods.keys()}")

        abundances = abundance_method_(spectral_data, endmembers)

        return abundances, endmembers

    return wrap


def _import_eea():
    """Lazily import pysptools' endmember-extraction algorithms (``eea``) - only
    PPI/FIPPI/NFINDR need them; VCA's own endmember search (:func:`_vca`) and all
    three abundance methods above are pure NumPy/SciPy and need no such import."""
    try:
        import pysptools.eea as eea
    except ImportError as exc:
        raise ImportError(
            "PPI/FIPPI/NFINDR need the optional 'pysptools' dependency "
            "(pip install pysptools); VCA does not.") from exc
    return eea


class PPI(AnalysisStep):
    """
    Pixel Purity Index (PPI).

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'`` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `pysptools <https://pysptools.sourceforge.io/>`_.


    References
    ----------
    Boardman, J.W., Kruse, F.A. and Green, R.O., 1995. Mapping target signatures via partial unmixing of AVIRIS data.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(_import_eea().PPI().extract), n_endmembers, abundance_method)


class FIPPI(AnalysisStep):
    """
    Fast Iterative Pixel Purity Index (FIPPI).

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'`` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `pysptools <https://pysptools.sourceforge.io/>`_.


    References
    ----------
    Chang, C.I. and Plaza, A., 2006. A fast iterative algorithm for implementation of pixel purity index. IEEE Geoscience and Remote Sensing Letters, 3(1), pp.63-67.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(_import_eea().FIPPI().extract), n_endmembers, abundance_method)


class NFINDR(AnalysisStep):
    """
    N-FINDR.

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'`` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `pysptools <https://pysptools.sourceforge.io/>`_.


    References
    ----------
    Winter, M.E., 1999, October. N-FINDR: An algorithm for fast autonomous spectral end-member determination in hyperspectral data. In Imaging Spectrometry V (Vol. 3753, pp. 266-275). SPIE.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(_nfindr), n_endmembers, abundance_method)


class VCA(AnalysisStep):
    """
    Vertex Component Analysis (VCA).

    Parameters
    ----------
    n_endmembers : int
        The number of endmembers.
    abundance_method: {'ucls', 'nnls', 'fcls'}, optional
        The abundance finder method to use. Default is ``'fcls'``.

        - ``'ucls'`` - Unconstrained Least Squares;
        - ``'nnls'`` - Non-negative Least Squares;
        - ``'fcls'`` - Fully-constrained Least Squares.


    .. note :: Implementation based on `the MATLAB code provided by the authors <http://www.lx.it.pt/~bioucas/code.htm>`_,
               and `Adrien Lagrange's translation to Python <https://github.com/Laadr/VCA>`_.


    References
    ----------
    Nascimento, J.M. and Dias, J.M., 2005. Vertex component analysis: A fast algorithm to unmix hyperspectral data. IEEE transactions on Geoscience and Remote Sensing, 43(4), pp.898-910.
    """

    def __init__(self, *, n_endmembers: int, abundance_method: Literal['ucls', 'nnls', 'fcls'] = 'fcls'):
        super().__init__(unmixer(_vca), n_endmembers, abundance_method)


def _vca(data, n_endmembers, *, snr_input=0):
    """
    Copyright 2018 Adrien Lagrange

    Licensed under the Apache License, Version 2.0.

    Source code available at: https://github.com/Laadr/VCA

    Changes:
    - sp -> np;
    - removed comments;
    - removed semicolons at the end of lines;
    - removed verbose option and print statements;
    - renamed some variables;
    - only return endmembers;
    """
    # Transpose data to comply with code
    data = data.T

    N = data.shape[1]

    # Estimate SNR
    # ------------
    if snr_input == 0:
        data_mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - data_mean  # data with zero-mean
        Ud = splin.svd(np.dot(data_centered, data_centered.T) / float(N))[0][:, :n_endmembers]
        x_p = np.dot(Ud.T, data_centered)

        P_y = np.sum(data ** 2) / float(N)
        P_x = np.sum(x_p ** 2) / float(N) + np.sum(data_mean ** 2)
        SNR = 10 * np.log10((P_x - n_endmembers / data.shape[0] * P_y) / (P_y - P_x))
    else:
        SNR = snr_input

    SNR_th = 15 + 10 * np.log10(n_endmembers)

    # Projection to n_endmembers-1 subspace if SNR < SNR_th; else, no projective projection
    # -------------------------------------------------------------------------------------
    if SNR < SNR_th:
        d = n_endmembers - 1
        if snr_input == 0:
            Ud = Ud[:, :d]
        else:
            data_mean = np.mean(data, axis=1, keepdims=True)
            data_centered = data - data_mean

            Ud = splin.svd(np.dot(data_centered, data_centered.T) / float(N))[0][:, :d]  # computes the p-projection matrix
            x_p = np.dot(Ud.T, data_centered)

        Yp = np.dot(Ud, x_p[:d, :]) + data_mean

        x = x_p[:d, :]
        c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
        y = np.vstack((x, c * np.ones((1, N))))
    else:
        d = n_endmembers
        Ud = splin.svd(np.dot(data, data.T) / float(N))[0][:, :d]

        x_p = np.dot(Ud.T, data)
        Yp = np.dot(Ud, x_p[:d, :])

        x = np.dot(Ud.T, data)
        u = np.mean(x, axis=1, keepdims=True)
        y = x / np.dot(u.T, x)

    # VCA algorithm
    # -------------
    indice = np.zeros((n_endmembers), dtype=int)
    A = np.zeros((n_endmembers, n_endmembers))
    A[-1, 0] = 1

    for i in range(n_endmembers):
        w = np.random.rand(n_endmembers, 1)
        f = w - np.dot(A, np.dot(splin.pinv(A), w))
        f = f / splin.norm(f)

        v = np.dot(f.T, y)

        indice[i] = np.argmax(np.absolute(v))
        A[:, i] = y[:, indice[i]]

    Ae = Yp[:, indice]

    return Ae.T


def _nfindr(spectral_data, num_of_endmembers):
    # pysptools expects a (h, w, bands) HSI cube; 'spectral_data' arrives already
    # flattened to (n_pixels, bands), so present it as a cube with a dummy width of 1
    # (pysptools reshapes it straight back to (n_pixels, bands) internally anyway).
    cube = spectral_data[:, np.newaxis, :]
    return _import_eea().NFINDR().extract(cube, num_of_endmembers)
