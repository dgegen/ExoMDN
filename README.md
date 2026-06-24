![ExoMDN](banner.png "Rapid characterization of exoplanet interiors")
# Rapid characterization of exoplanet interiors with Mixture Density Networks
![MIT License](https://img.shields.io/github/license/philippbaumeister/MDN_exoplanets.svg?style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7670706-blue?style=flat-square)](https://doi.org/10.5281/zenodo.7670706)
[![Paper](https://img.shields.io/badge/Paper-10.1051%2F0004--6361%2F202346216-red?style=flat-square)](https://doi.org/10.1051/0004-6361/202346216)

ExoMDN is a machine-learning-based exoplanet interior inference model using Mixture Density Networks. The model is 
trained on more than 5.6 million synthetic planet interior structures. Given mass, radius, and equilibrium 
temperature, ExoMDN is capable of providing a full inference of the interior structure of low-mass exoplanets in 
under a second without the need for a dedicated interior model.

This repository contains the trained models shown [Baumeister & Tosi 2023](https://doi.org/10.1051/0004-6361/202346216), as well as Python 
notebooks to load the models and run interior predictions of exoplanets. Interactive widgets are included 
to simplify loading an MDN model and running a prediction. 
We also make available the training routines in `more_examples/model_training_demo.ipynb`.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment and dependencies.
Install uv if you haven't already (see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/)), then
from the repository root run:
```
uv sync
```
This creates a virtual environment in `.venv` and installs the *exomdn* package along with all required dependencies.
This project requires Python 3.9 or higher, and currently does not work well with Python versions above 3.10 due to
incompatibility with higher tensorflow versions; `uv sync` will automatically install a compatible Python version if needed.

To run commands within the environment, prefix them with `uv run`, e.g.:
```
uv run jupyter notebook
```

### Required packages

- python>=3.9,<3.11
- tensorflow==2.15.*
- tensorflow-probability==0.23.*
- scikit-learn==1.1.1
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- joblib
- ipywidgets
- jupyter

## Getting started

To get started check out `introduction.ipynb`. More in-depth examples can be found in the *more_examples* directory and 
more will be added over time. This directory also contains the training routines used to train the ExoMDN models, with some example trianing data to be found in the *data* directory.

## Acknowledgements
We are using the MDN layer for Keras by https://github.com/cpmpercussion/keras-mdn-layer 
