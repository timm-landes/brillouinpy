#  Brillouin Analyzer

This modul should help you to analyze Brillouin light scattering data efficiently. This is heavily, let's say "inspired" by the package RamanSPy. For the most steps it uses the same syntax, but I started removing unnecessary functions and adding meaningfull ones.

## Installation

### From Git-Repository
I use Anaconda as a python developement environment but installing anythin from scratch should also work. However, 

It lets you create environments in which certain packages can be installed. Start conda by opening the Anaconda Promt via the windows application launcher.

The creation of a new environment is done by: 
```bash
conda create --name <mynewenv>
```
Change `<mynewenv>` to something that you can remember. Activation of this environment is done by `conda create --name <mynewenv>`. In this environment there are certain packages we need to install prior to the Brillouin Analyzer.

First of all we install pip:
```bash
conda install pysptools
```
To install 

```bash
pip install git+ssh://git@gitlab.uni-hannover.de/phytophotonics/brillouinanalyzer.git
```
					
### Lokale Installation


1. Klonen Sie das Repository:

   ```bash
   git clone https://gitlab.uni-hannover.de/phytophotonics/brillouinanalyzer.git
   ```

2. Wechseln Sie in das Verzeichnis:

   ```bash
   cd my_module
   ```

3. Installieren Sie das Paket:

   ```bash
   pip install .
   ```

4. Für Entwicklungszwecke können Sie das Paket editierbar installieren:

   ```bash
   pip install -e .
   ```
## Nutzung


Beschreiben Sie hier, wie man Ihr Modul verwendet, vielleicht mit ein paar Codebeispielen.
``
