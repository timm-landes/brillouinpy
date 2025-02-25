# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:31:00 2025

@author: Timm
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from .spectralcontainer import SpectralImageContainer

class SpectralImageProcessor:
    def __init__(self):
        self.processing_steps = []
        self.results = {}

    def add_preprocessing_step(self, step_name, **kwargs):
        # Fügt einen Verarbeitungsschritt zur Pipeline hinzu
        self.processing_steps.append((step_name, kwargs))

    def apply(self, spectral_container):
        pipeline_length = len(self.processing_steps)
        for i, (step_name, kwargs) in enumerate(self.processing_steps):
            if hasattr(self, step_name):
                print(f"Executing step: {i}/{pipeline_length}")
                method = getattr(self, step_name)
                data = method(spectral_container, **kwargs)
            else:
                raise ValueError(f"Method {step_name} not found in processor.")

        # Erzeugen eines neuen Containers für die bearbeiteten Daten
        processed_container = SpectralImageContainer()
        processed_container.initialize_from_processed_data(data, spectral_container.get_frequency_axis())

        # Rückgabe des neuen Containers mit den bearbeiteten Daten
        return processed_container

    def detect_and_remove_spectral_response(self, spectral_container):
        # Entfernt die spektrale Antwort aus den Spektraldaten
        raw_hyperstack = spectral_container.get_hyperstack()
        corrected_hyperstack = np.copy(raw_hyperstack)
        spectral_responses = np.zeros_like(corrected_hyperstack)
        
        # Frequenzachse bleibt unverändert im Container erhalten
        frequency_axis = spectral_container.get_frequency_axis()

        for x in range(raw_hyperstack.shape[0]):
            for y in range(raw_hyperstack.shape[1]):
                pp_data = np.vstack([frequency_axis, corrected_hyperstack[x, y, :]])
                center, fitness, start, end, response = self.find_instrumental_response(pp_data)
                
                spectral_responses[x, y, start:end] = response[1, start:end]
                corrected_hyperstack[x, y, start:end] -= response[1, start:end]

        return corrected_hyperstack
    
    def apply_smoothing(self, spectral_container, sigma=1):
        hyperstack = spectral_container.get_hyperstack()
        corrected_hyperstack = np.copy(hyperstack)
        
        # Glättet die Daten mit einem Gaußschen Filter
        corrected_hyperstack = gaussian_filter(hyperstack, sigma=sigma)
        return corrected_hyperstack
    
    @staticmethod
    def find_instrumental_response(array):
        # Findet die spektrale Antwort in einem Array
        center_response_index = np.argmax(array[1, :])
        laser_fitness = np.max(array[1, :])
        inst_response = np.zeros(array.shape)
        inst_response[0, :] = array[0, :]
        
        antistokes_end_index = next(
            (i + 1 + 50 for i in range(center_response_index, center_response_index + 50) if array[1, i] <= 1), len(array[1]))
        stokes_end_index = next(
            (i - 50 for i in range(center_response_index, center_response_index - 50, -1) if array[1, i] <= 1), 0)
        
        inst_response[1, stokes_end_index:antistokes_end_index] = array[1, stokes_end_index:antistokes_end_index]
        
        return center_response_index, laser_fitness, stokes_end_index, antistokes_end_index, inst_response