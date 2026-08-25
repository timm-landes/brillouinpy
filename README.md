
#  Brillouin Analyzer

This module should help you to analyze Brillouin light scattering data efficiently. This is strongly, let's say "inspired" by the package [RamanSPy](https://github.com/barahona-research-group/RamanSPy). For the most steps it uses the same syntax, but I started to remove unnecessary functions and added useful ones for Brillouin imaging. For instance, Raman scattering data often requires a baseline correction. This however, is something we don't do in Brillouin. Usually the background in Brillouin light scattering experiments is very flat and only requires the removal of a constant background. But analysis of Brillouin data requires removal of the Rayleigh scattered light, or in our case removal of the reference beam.



**Currently only for internal use within the Phytophotonics Division, as it contains many RamanSPy references that are not clearly marked! Please do not share this package with others without asking for permission first.**.

## Documentation
A step-by-step [Tutorial](docs/tutorial.md) walking through the `examples/` folder, and the full API reference (built from the docstrings in this package), are published via GitLab Pages. Find the link under **Deploy → Pages** in this project on GitLab once the `pages` pipeline job has run on `main`.

To build it locally instead:
```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
Then open `docs/_build/html/index.html` in your browser.

## Installation
This gives a short manual for installation of Anaconda and BrillouinAnalyzer. I highly recommend the use of Anaconda as a Python distribution if you are new to Python Scripting and Programming. After installing BrillouinAnalyzer, you will be able to use BrillouinAnalyzer like any other Python package.

### Prerequisites
 1. Install Anaconda: [Link to Anaconda download website](https://www.anaconda.com/download/success).
 2. Create a new conda environment `conda create --name <mynewenv>`. Make sure to replace `<mynewenv>` with a correct an perceptible name. You will later need to recall it.
 3. Activate the environment via `conda activate <mynewenv>`.
 4. Install pip and git via `conda install pip`.

Now you can choose one of the two next sections to install Brillouin Analyzer depending on your needs. 
 1. Installation using the **Git-Repository**. This method is recommended if you do not work on the package itself. Here you will always find the latest version of BrillouinAnalyzer.
 2. Installation from a **local directory**. This method is recommended if you want to work on the package code istelf. It will be possible to make changes to the code that can be applied directly.
  
Now just one remark: **Keep the installation of Conda as it is, unless you definitly need newer packages! :)** I bricked my conda several times updating conda.

### Install BrillouinAnalyzer from the Git-Repository
For this you'll need git. Intall it via `conda install pip` and need to set up a local SSH key in GitLab. A nice tutorial you can find [here (Link to Youtube)](https://www.youtube.com/watch?v=Vmt0V6a3ppE).
You have already installed git. So you can directly type in the Anaconda prompt.
However, this installation has the big advantage that it's easier to keep your installation up-to-date. 

```bash
pip install git+ssh://git@gitlab.uni-hannover.de/phytophotonics/brillouinanalyzer.git
```
					
### Install BrillouinAnalyzer from a local folder


1. Generate a folder where you want to save the module.
2. Move to the folder in the Anaconda prompt: 
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
## Updates
1. To update the installation of BrillouinAnalyzer open the anaconda prompt and start the conda environment `conda activate <mynewenv>`.
2. When installed from the git repository type
   ```bash
   pip install --upgrade git+ssh://git@gitlab.uni-hannover.de/phytophotonics/brillouinanalyzer.git.
   ```
3. When installed from a local folder navigate to the folder
   ```bash
   cd <your folder path>
   ```
   and then for the locked installation
   ```bash
   pip install --upgrade .
   ```
   For the editable mode:
   ```bash
   pip install -e .
   ```

## Example
I attached some examples in the [`example`-folder](examples/) of this project. You can access those, copy them, alter them, or clone them. This shows how capable BrillouinAnalyzer is regarding ML approaches. For a guided, illustrated walkthrough of those examples, see the [Tutorial](docs/tutorial.md).

Subsequently, you'll find a short example code. Example data can be found in [here](https://seafile.projekt.uni-hannover.de/d/ae7aff2e3bf14f119ed5/). The necessary password is `brillouin_test_data`.
```python
import brillouinanalyzer as bp
import matplotlib.pyplot as plt

#%% Simple Analysis
if __name__ == '__main__':
    # Set the project path. Here, your raw data needs to be in a folder 'data'.
    # All output will be stored in the folder pp_data. If nonexistent, it will be generated. 
    project_path = r'complete_path_to_your_project'
    
    # Load of the Brillouin spectral data
    brillouin_data = bp.utils.load_spectral_image(project_path, 'Brillouin')
    # Calculate the Frequency axis of the Brillouin data. Values needs to fit to your data!
    brillouin_frequency_scale = bp.utils.brillouin_spectral_axis(
        mirror_spacing = 3e-3, # [m]
        scan_amplitude = 309e-9, # [m] 
        no_of_channels = brillouin_data.shape[-1] # get spectral dimension from data
    )
    
    # Generate the Brillouin object
    brillouin_data = bp.SpectralImage(brillouin_data, brillouin_frequency_scale)
    
    # Plot the mean spectrum of the samples data
    bp.plot.mean_spectra(brillouin_data, title='Raw Brillouin Data')
    plt.ylim(.1, None)
    bp.plot.show()
    
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
        
        # Smooting of the intensity
        bp.preprocessing.denoise.SavGol(window_length= 7, polyorder= 2) 
        ])
    
    
    # Apply the pipeline to the data
    pp_brillouin_data = pipeline.apply(brillouin_data)
    
    # Plot the mean spectrum of the preprocessed samples data
    bp.plot.mean_spectra(pp_brillouin_data, title='Processed Brillouin Data', yscale='linear')
    plt.ylim(.0, None)
    bp.plot.show()

```

## Usage in Spyder
When you want to use the package in Spyder you have to install the Spyder kernels in your environment:
```bash
   conda install spyder-kernels
   ```
If you are using Spyder 5.XX you'll need to install spyder-kernels in a specific version. 
```bash
   conda install spyder-kernels=2.5
   ```
Do **NOT** use pip here!
You then can open a new console (Spyder 6) and select the environment or change the default environment (Spyder 5) with right-click on the bottom status bar right.


## Changelog
- 0.1: Initial upload
- 0.1.1: Added support for missing data points. Now the pipeline gives the user a warning, when points are missing but doesn't stop executing. The missing values are treated as None.