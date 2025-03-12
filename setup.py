# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:08:15 2025

@author: Timm
"""

from setuptools import setup, find_packages

setup(
    name='BrillouinAnalyzer',  
    version='0.1.1',  
    packages=find_packages(),  # Automatisches Finden von Paketen im Projektverzeichnis
    install_requires=[  # Liste von Abhängigkeiten Ihres Pakets
         'numpy',
         'scipy',
         'tqdm',
         'matplotlib',
         'scikit-image',
         'scikit-learn',
         'pysptools',
    ],
    author='Timm Landes',
    author_email='timm.landes@hot.uni-hannover.de',
    description='Geänderte Version von RamanSPy um die Brillouin-Daten damit zu analysieren',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # Format der Langbeschreibung
    url='https://gitlab.uni-hannover.de/phytophotonics/brillouinanalyzer',  # URL zu Ihrem Projekt
    classifiers=[  # Optional: Klassifizierung Ihres Pakets
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Beispiel für eine Lizenz
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Minimale Python-Version, die unterstützt wird
    )