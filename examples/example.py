# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:37:29 2025

@author: Timm
"""
import BrillouinAnalyzer as bp
import ramanspy as rp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#%% Simple Analysis
if __name__ == '__main__':
    # Set the project path. Here, your data should be located in a folder data.
    # All output will be stored in the folder pp_data. If nonexistent, it will be generated. 
    # project_path = r'C:/Users/Timm/Desktop/Leipzig/Zygo_1'
    project_path = r'C:\Users\Timm\Desktop\Victor\Idared_95_05'
    
    # Load of the Brillouin spectral data
    brillouin_data = bp.utils.load_spectral_image(project_path, 'Brillouin')
    # Calculate the Frequency axis of the Brillouin data. The sett
    brillouin_frequency_scale = bp.utils.brillouin_spectral_axis(
        mirror_spacing = 6e-3, # [m]
        scan_amplitude = 480e-9, # [m] 
        no_of_channels = brillouin_data.shape[-1] # get spectral dimension from data
    )
    
    # Generate the Brillouin object
    brillouin_data = bp.SpectralImage(brillouin_data, brillouin_frequency_scale)
    
    # Plot the mean spectrum of the samples data
    bp.plot.mean_spectra(brillouin_data, title='Raw Brillouin Data')
    # plt.ylim(.1,100)
    bp.plot.show()
    
    # Define the processing pipeline. This gives you an example on what data manipulation/processing is possible
    pipeline = bp.preprocessing.Pipeline([
        # # Normalization regarding the spectrometer performance
        # bp.preprocessing.normalise.MaxIntensity(pixelwise=True # When pixelwise normalization is wanted
        #                                         ), 
        
        # # Removal of the Intrument Response Function (IRF) and intensity signal deconvolution
        bp.preprocessing.misc.Deconvoluter_IRF(offset=65, # additional offset from IRF. You can use this parameter to remove unwanted Rayleigh scattered light
                                               iterations = 4, # This is the number of iteration cycles. This is not something that should be changed unintentionally
                                               padding = None # currently not in use
                                               ), 
        
        # # Smooting of the intensity
        # bp.preprocessing.denoise.SavGol(window_length= 7, polyorder= 2) 
        
        # # Normalization regarding signal intensity
        # bp.preprocessing.normalise.MaxIntensity(pixelwise=True), 
        ])
    
    
    # Apply the pipeline to the data
    pp_brillouin_data = pipeline.apply(brillouin_data)
    
    # Plot the mean spectrum of the preprocessed samples data
    bp.plot.mean_spectra(pp_brillouin_data, title='Processed Brillouin Data', yscale='linear')
    plt.ylim(.0,30)
    bp.plot.show()
    
    # Perform the data fitting using the DHO
    DHO_fit = bp.analysis.fitmodel.DHO(expected_peaks=2,        # How much peaks do you expect
                                       p0= [10,7.5,1,5,15,1,0,0],    # Give the algorithm a good starting point, must match the lenght (expected_peaks*3)+2
                                       bounds= None,            # Give some bounds if you get unreasonable results or have strongly overlapping features
                                       padding = None              # Gurrently not in use
    )
    fitted_parameters, metric = DHO_fit.apply(pp_brillouin_data)

    # Plot all fitted variables    
    for parameter in fitted_parameters[:-2]:
        plt.imshow(parameter)
        plt.colorbar()
        plt.show()
        
    # Basic data analysis part done!
#%% Advanced Analysis

    # Here we do a Vertex Component Analysis (VCA)
    # Definition of the VCA parameters. Important for you is here the number of endmembers n_endmembers. Abundance_method is less relevant to you and defines the algorithm to determine the maps of the members.
    unmixer = bp.analysis.unmix.VCA(n_endmembers=2, abundance_method='ucls') 
    
    # Applying the VCA onto our Brillouin object. That is already all that's to it.
    abundance_maps, endmembers = unmixer.apply(pp_brillouin_data)
    
    # Plot the endmembers spectra
    bp.plot.spectra(endmembers, pp_brillouin_data.spectral_axis, plot_type="single", label=[f"Endmember {i + 1}" for i in range(len(endmembers))], yscale = 'linear')
    
    
    # Let's also make an overlay plot of where each endmember is most present using matplotlib
    fig, ax = plt.subplots()
    # Define coloring of the plot
    cmap = plt.get_cmap()(np.linspace(0, 1, len(abundance_maps)))
    white = [1, 1, 1, 0]
    
    for i in range(len(endmembers)):
        ax.imshow(abundance_maps[i], cmap=LinearSegmentedColormap.from_list('', [white, cmap[i]]))
    plt.show()
    
    
#%% Raman Analysis
    raman_data = bp.utils.load_spectral_image(project_path, 'Raman')
    raman_spectral_axis = rp.utils.wavelength_to_wavenumber(bp.utils.raman_spectral_axis(600,605),532.1)
    raman_data = rp.SpectralImage(raman_data, raman_spectral_axis)
    
    raman_pipeline = rp.preprocessing.Pipeline([
        rp.preprocessing.misc.Cropper(region=(500, 1800)),
        rp.preprocessing.despike.WhitakerHayes(kernel_size=3, threshold=1),
        rp.preprocessing.denoise.SavGol(window_length=21, polyorder=2),
        # rp.preprocessing.denoise.Gaussian(),
        rp.preprocessing.baseline.AIRPLS(),
        rp.preprocessing.normalise.MaxIntensity()
    ])
    
    pp_raman_data = raman_pipeline.apply(raman_data)
    
    
    peak_prominence = 0.3
    
    rp.plot.peaks(pp_raman_data.mean, prominence=peak_prominence)
    list_of_peaks = pp_raman_data.spectral_axis[pp_raman_data.mean.peaks(prominence=peak_prominence)[0]]
    rp.plot.show()
    
    for peak in list_of_peaks:
        ax = rp.plot.image(pp_raman_data.band(peak), title = f'Peak at {peak}')
        plt.tight_layout()
        # ax.figure.savefig(f'Raman_{np.round(peak, 3)}.png', dpi = 600)
    rp.plot.show()