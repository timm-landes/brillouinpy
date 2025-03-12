#%% Package import
import brillouinanalyzer as bp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#%% Simple data import
if __name__ == '__main__':
    # Set the project path. Here, your data should be located in a folder data.
    # All output will be stored in the folder pp_data. If nonexistent, it will be generated. 
    project_path = r'C:\\Users\\Timm\\Desktop\\Leipzig\\Zygo_2_4'
    # project_path = r'C:\Users\Timm\Desktop\Victor\Idared_95_05'
    
    # Load of the Brillouin spectral data
    brillouin_data = bp.utils.load_spectral_image(project_path, 'Brillouin')
    # Calculate the Frequency axis of the Brillouin data.
    brillouin_frequency_scale = bp.utils.brillouin_spectral_axis(
        mirror_spacing = 6e-3, # [m]
        scan_amplitude = 480e-9, # [m] 
        no_of_channels = brillouin_data.shape[-1] # get spectral dimension from data
    )

    # Generate the Brillouin object
    brillouin_data = bp.SpectralImage(brillouin_data, brillouin_frequency_scale)

    #%%% Basic Analysis
    #     
    # Define the processing pipeline. This gives you an example on what data manipulation/processing is possible
    pipeline = bp.preprocessing.Pipeline([
        # # Normalization regarding the spectrometer performance
        bp.preprocessing.normalise.MaxIntensity(pixelwise=True # When pixelwise normalization is wanted
                                                ), 
        
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
    
    # Plot the mean spectrum of the raw and preprocessed samples data
    fig = plt.figure(figsize=(10, 5), layout='constrained')
    plt.subplot(121)
    bp.plot.mean_spectra(brillouin_data, title='Raw Brillouin Data', yscale='log')  
    plt.subplot(122)
    bp.plot.mean_spectra(pp_brillouin_data, title='Processed Brillouin Data', yscale='linear')
    plt.show()

    # Perform the data fitting using the DHO
    DHO_fit = bp.analysis.fitmodel.DHO(expected_peaks=1,        # How much peaks do you expect
                                       p0= [        # Give the algorithm a good starting point, must match the lenght (expected_peaks*3)+2
                                           0.005,   # Amplitude
                                           8.5,     # Frequency Shift in GHz
                                           1,       # FWHM in GHz
                                           0,       # Background
                                           0        # Symmetry
                                           ],    
                                       bounds= None,            # Give some bounds if you get unreasonable results or have strongly overlapping features
    )
    fitted_parameters, metric = DHO_fit.apply(pp_brillouin_data)
    
    for parameter in fitted_parameters:
        print('DHO:', np.nanmean(parameter))
    
    # Plot all fitted variables
    plt.figure(figsize=(15, 5), layout='constrained')
    plt.subplot(131)
    plt.imshow(fitted_parameters[0])    
    plt.colorbar(label='Amplitude (a.u.)')
    plt.subplot(132)
    plt.imshow(fitted_parameters[1])
    plt.colorbar(label='Frequency (GHz)')
    plt.subplot(133)
    plt.imshow(fitted_parameters[2])
    plt.colorbar(label='Linewidth (GHz)')
    plt.title('Deconvolution then fit')
    plt.show()

    # # Perform the data fitting using the DHO
    # LOR_fit = bp.analysis.fitmodel.Lorentzian(expected_peaks=1,        # How much peaks do you expect
    #                                    p0= [1500,9,1,0,0],    # Give the algorithm a good starting point, must match the lenght (expected_peaks*3)+2
    #                                    bounds= None,            # Give some bounds if you get unreasonable results or have strongly overlapping features
    # )
    # fitted_parameters, metric = LOR_fit.apply(pp_brillouin_data)
    
    # for parameter in fitted_parameters:
    #     print('Lorentz:', np.nanmean(parameter))
        
    # Plot all fitted variables    
    # for parameter in fitted_parameters[:-2]:
    #     vmax = np.nanmean(parameter) + 1 * np.nanstd(parameter)
    #     vmin = np.nanmean(parameter) - 1 * np.nanstd(parameter)
    #     plt.imshow(parameter)
        
    #     plt.colorbar()
    #     plt.show()
        
    # Basic data analysis part done!
#%% Advanced Analysis

    # Here we do a Vertex Component Analysis (VCA)
    # Definition of the VCA parameters. Important for you is here the number of endmembers n_endmembers. Abundance_method is less relevant to you and defines the algorithm to determine the maps of the members.
    unmixer = bp.analysis.unmix.VCA(n_endmembers=2, abundance_method='ucls') 
    
    # Applying the VCA onto our Brillouin object. That is already all that's to it.
    abundance_maps, endmembers = unmixer.apply(pp_brillouin_data)
    
    # Plot the endmembers spectra
    plt.figure(figsize=(10, 5), layout='tight')
    plt.subplot(121)
    bp.plot.spectra(endmembers, pp_brillouin_data.spectral_axis, plot_type="single", label=[f"Endmember {i + 1}" for i in range(len(endmembers))], yscale = 'linear')
    
    
    # Let's also make an overlay plot of where each endmember is most present using matplotlib
    ax = plt.subplot(122)
    # Define coloring of the plot
    cmap = plt.get_cmap()(np.linspace(0, 1, len(abundance_maps)))
    white = [1, 1, 1, 0]
    
    for i in range(len(endmembers)):
        ax.imshow(abundance_maps[i], cmap=LinearSegmentedColormap.from_list('', [white, cmap[i]]))
    plt.show()