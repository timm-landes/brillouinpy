#  Brillouin Analyzer

This modul should help you to analyze Brillouin light scattering data efficiently. This is heavily, let's say "inspired" by the package RamanSPy. For the most steps it uses the same syntax, but I started removing unnecessary functions and adding meaningfull ones for Brillouin imaging.

## Installation
This gives a short manual for installation of Anaconda and BrillouinAnalyzer. I highly recommend the use of Anaconda as a Python distribution if you are new to Python Scripting and Programming.

### Prerequisites
 1. Install Anaconda: [Link to Anaconda download website](https://www.anaconda.com/download/success).
 2. Create a new conda environment `conda create --name <mynewenv>`. Make sure to replace `<mynewenv>` with a correct an perceptible name. You llater need to recall it.
 3. Activate the environment via `conda activate <mynewenv>`.
 4. Install pip and git via `conda install pip git`.

Now you can choose one of the two next sections to install Brillouin Analyzer.

### Install BrillouinAnalyzer from the Git-Repository
For this you'll need git and need to setup a local SSH key in Gitlab. However, this installation has the big advantage that its easier to keep your installation up-to-date. 

```bash
pip install git+ssh://git@gitlab.uni-hannover.de/phytophotonics/brillouinanalyzer.git
```
					
### Install BrillouinAnalyzer from a local folder


1. Generate a folder where you want to save the module.
2. Move to the folder in the Anaconda promt: 
   ```bash
   cd <your folder path>
   ```
4. Clone the Repository into this folder:
   ```bash
   git clone https://gitlab.uni-hannover.de/phytophotonics/brillouinanalyzer.git
   ```
6. Installation of BrillouinAnalyzer:
   ```bash
   pip install .
   ```

7. You can install the package editable for development purposes:

   ```bash
   pip install -e .
   ```
   
## Example
```python
import BrillouinAnalyzer as bp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#%% Simple Analysis
if __name__ == '__main__':
    # Set the project path. Here, your raw data needs to be in a folder 'data'.
    # All output will be stored in the folder pp_data. If nonexistent, it will be generated. 
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
```

## Usage in Spyder
When you want to use the package in spyder you have to install the spyder kernels:
```bash
   conda install spyder-kernels
   ```
You then can open a new console and select the environment.
