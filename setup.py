# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:08:15 2025

@author: Timm
"""

from setuptools import setup, find_packages

setup(
    name='BrillouinPy',
    version='0.1.1',
    # Finds both 'brillouinpy' (the real package) and 'brillouinanalyzer' (a thin
    # backwards-compatibility shim for the package's former name) automatically.
    packages=find_packages(),
    install_requires=[  # Liste von Abhängigkeiten Ihres Pakets
         'numpy',
         'scipy',
         'tqdm',
         'matplotlib',
         'scikit-image',
         'scikit-learn',
         'pysptools',
         'tifffile',
    ],
    extras_require={
        'hdf5_bls': ['HDF5_BLS'],
        # 'brim' export/import (export.to_brim/from_brim) needs Python >= 3.11
        'brim': ['brimfile'],
    },
    author='Timm Landes',
    author_email='timm.landes@hot.uni-hannover.de',
    description='Brillouin light scattering data analysis (formerly BrillouinAnalyzer), architecturally inspired by RamanSPy (see NOTICE.md)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # Format der Langbeschreibung
    url='https://gitlab.uni-hannover.de/phytophotonics/brillouinpy',  # URL zu Ihrem Projekt
    license='BSD-3-Clause',
    classifiers=[  # Optional: Klassifizierung Ihres Pakets
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Minimale Python-Version, die unterstützt wird
    )
