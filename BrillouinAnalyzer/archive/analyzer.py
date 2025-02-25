# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:35:43 2025

@author: Timm
"""

import numpy as np
from .spectralcontainer import SpectralImageContainer

class SpectralAnalyzer:
    def __init__(self, spectral_container: SpectralImageContainer):
        self.spectral_container = spectral_container

    def perform_vca(self, num_endmembers, snr_input=0):
        X = self.spectral_container.get_corrected_hyperstack()
        num_pixels_x, num_pixels_y, spectral_length = X.shape
        Y = np.reshape(X, (num_pixels_x * num_pixels_y, spectral_length), order='A').T

        L, N = Y.shape
        R = int(num_endmembers)
        
        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m
            Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :R]
            x_p = np.dot(Ud.T, Y_o)
            SNR = self.estimate_snr(Y, y_m, x_p)
            print(f"SNR estimated = {SNR}[dB]")
        else:
            SNR = snr_input
            print(f"Input SNR = {SNR}[dB]")

        SNR_th = 15 + 10 * np.log10(R)

        if SNR < SNR_th:
            print("... Select proj. to R-1")
            
            d = R - 1
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m
            Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :d]
            x_p = np.dot(Ud.T, Y_o)
            
            Yp = np.dot(Ud, x_p) + y_m
            x = x_p
            c = np.amax(np.sum(x**2, axis=0))**0.5
            y = np.vstack((x, c * np.ones((1, N))))
        else:
            print("... Select the projective proj.")
            
            d = R
            Ud = np.linalg.svd(np.dot(Y, Y.T) / float(N))[0][:, :d]
            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(Ud, x_p)
            x = x_p
            u = np.mean(x, axis=1, keepdims=True)
            y = x / (np.dot(u.T, x) + 1e-7)

        indices = self.vca_algorithm(y, R)
        Ae = Yp[:, indices]
        return Ae, indices, Yp

    @staticmethod
    def estimate_snr(Y, y_m, x_p):
        L, N = Y.shape
        p, _ = x_p.shape
        P_y = np.sum(Y**2) / float(N)
        P_x = np.sum(x_p**2) / float(N) + np.sum(y_m**2)
        snr_est = 10 * np.log10((P_x - p/L * P_y) / (P_y - P_x))
        return snr_est

    @staticmethod
    def vca_algorithm(y, R):
        indices = np.zeros((R), dtype=int)
        A = np.zeros((R, R))
        A[-1, 0] = 1

        for i in range(R):
            w = np.random.rand(R, 1)
            f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
            f /= np.linalg.norm(f)
            v = np.dot(f.T, y)
            indices[i] = np.argmax(np.abs(v))
            A[:, i] = y[:, indices[i]]

        return indices