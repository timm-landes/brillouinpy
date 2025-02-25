# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:29:21 2025

@author: Timm
"""
import os, re
import numpy as np
import matplotlib.pyplot as plt


class SpectralImageContainer:
    def __init__(self):
        self.project_path = None
        self.data_path = None
        self.pp_data_path = None
        self.mirror_spacing = None
        self.scan_amplitude = None
        self.laser_wavelength = 532.1e-9
        self.file_list = []
        self.no_of_channels = 0
        self.frequency_axis = None
        self.hyperstack = None

    def initialize_from_raw_data(self, project_path, mirror_spacing, scan_amplitude, laser_wavelength=532.1e-9):
        self.project_path = project_path
        self.data_path = os.path.join(project_path, 'data')
        self.pp_data_path = os.path.join(project_path, 'pp_data')

        if not os.path.exists(self.pp_data_path):
            os.makedirs(self.pp_data_path)

        self.mirror_spacing = mirror_spacing
        self.scan_amplitude = scan_amplitude
        self.laser_wavelength = laser_wavelength

        self.file_list = self.list_of_DAT_files(self.data_path)
        if not self.file_list:
            raise ValueError("No .DAT files found in the specified data path.")
        
        self.first_data, self.no_of_channels = self.import_DAT_File(self.file_list[0])
        self.frequency_axis = self.freq_scale()
        self.hyperstack = self.load_data()

    def initialize_from_processed_data(self, corrected_hyperstack, frequency_axis):
        self.hyperstack = corrected_hyperstack
        self.frequency_axis = frequency_axis

    def freq_scale(self):
        fsr = self.fsr(self.mirror_spacing)
        freq_limits = fsr * self.scan_amplitude / self.laser_wavelength
        freq_limits_GHz = freq_limits * 1e-9
        return np.linspace(-freq_limits_GHz, freq_limits_GHz, self.no_of_channels)

    @staticmethod
    def fsr(mirror_spacing):
        speed_of_light = 299792458  # m/s
        return speed_of_light / (2 * mirror_spacing)

    def import_DAT_File(self, file):
        data = np.loadtxt(file, skiprows=11)
        return data, data.shape[0]

    def list_of_DAT_files(self, filepath):
        return [os.path.join(filepath, f) for f in sorted(os.listdir(filepath)) if f.endswith(".DAT")]

    def get_data_shape(self):
        max_values = [0, 0, 0, 0]
        for filename in self.file_list:
            filepath = os.path.join(self.data_path, filename)
            coordinates = self.extract_coordinates(filepath)
            if coordinates:
                max_values = [max(m, c) for m, c in zip(max_values, coordinates)]
        return [m + 1 for m in max_values if m > 0]

    @staticmethod
    def extract_coordinates(filepath):
        filename = os.path.basename(filepath)
        pattern = r'Bri_(\d+)_(\d+)_(\d+)_(\d+)'
        match = re.match(pattern, filename)
        return tuple(map(int, match.groups())) if match else None

    def load_data(self):
        data_shape = self.get_data_shape()
        hyperstack = np.zeros((data_shape[0], data_shape[1], self.no_of_channels))

        for file in self.file_list:
            coordinates = self.extract_coordinates(file)
            x, y, z, t = coordinates
            data, _ = self.import_DAT_File(file)
            hyperstack[x, y, :] = data

        return hyperstack

    def set_corrected_hyperstack(self, corrected_hyperstack):
        self.hyperstack = corrected_hyperstack

    def get_hyperstack(self):
        return self.hyperstack

    def get_frequency_axis(self):
        return self.frequency_axis

    @property
    def shape(self):
        return self.hyperstack.shape

    @property
    def mean(self):
        return np.mean(self.hyperstack, axis=(0, 1))

    @property
    def variance(self):
        return np.var(self.hyperstack, axis=(0, 1))
    
    
    def plot_mean_spectrum(self, scale = 'log'):
        frequency_axis = self.frequency_axis
        mean = self.mean
        variance = self.variance

        plt.figure(figsize=(10, 6))
        plt.plot(frequency_axis, mean, label='Mean Spectrum')
        plt.fill_between(frequency_axis, mean - np.sqrt(variance), mean + np.sqrt(variance),
                         color='lightgray', alpha=0.5, label='Standard Deviation')
        plt.title('Mean Spectrum with Variance')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Intensity (a.u.)')
        plt.yscale(scale)
        plt.legend()
        plt.grid(True)
        plt.show()

        return "Plot generated"