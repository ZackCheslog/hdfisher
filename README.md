# Fisher Forecasting for CMB-HD

This repository contains software to produce Fisher forecasts for CMB-HD. It also contains the necessary files (along with functions to access them) that were used to produce the results in the main text of MacInnis, Sehgal, and Rothermel (2023). 

If you use this software, please cite MacInnis, Sehgal, and Rothermel (2023).


# Installation

## Installation requirements

To use this software, you must have Python (version >=3) installed, along with the following Python packages:
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [CAMB](https://camb.readthedocs.io/en/latest/)
- [matplotlib](https://matplotlib.org/) (this is only required to run the Jupyter notebooks)
- [getdist](https://getdist.readthedocs.io/en/latest/intro.html) (this is only required to run the Jupyter notebooks)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (__optional__: the calculation of the derivatives used in the Fisher matrices can be parallelized, but this is not required.) 


## Installation instructions

Simply close this repository and install with `pip`:

```
git clone https://github.com/CMB-HD/hdfisher.git
cd hdfisher
pip install . --user
```

To uninstall the code, use the command `pip uninstall hdfisher`. (Note that you may have to navigate away from the `hdfisher` directory, i.e. the directory containing this README, before running this command).


# CMB and BAO mock signal and covariance matrices

We provide mock signal and covariance matrices for SO-like, CMB-S4-like, and CMB-HD-like experiments, and for mock DESI BAO. These files are in the sub-directories of the `hdlike/data` directory, and can be accessed by using the `Data` class in `hdfisher/dataconfig.py`; see the provided notebook `forecast_plots.ipynb` (described in the "Reproducing plots and tables" section below) for an example.

For the three CMB experiments, the mock signal consists of the delensed CMB power spectra ($C_\ell^{XY}$ for $XY \in [TT,~TE,~EE,~BB]$ in units of $\mu$K$^2$) and the CMB lensing power spectrum ($C_L^{\kappa\kappa} = [L(L+1)^2] C_L^{\phi\phi} / 4$). We also provide the corresponding noise spectra. For CMB-HD, we include additional mock signal and covariance matrices that were used to obtain the results in MacInnis et. al. (2023). The mock DESI BAO signal consists of values of $r_s / d_V(z)$ for the redshifts $z$ ranging from $0.05$ to $1.85$. We also provide covariance matrices for the mock signal in each case.


# Pre-computed Fisher matrices

We provide pre-computed Fisher matrices in the `hdfisher/data/fisher_matrices` directory, which were used to obtain the results in MacInnis et. al. (2023). These can also be accessed with the `Data` class in `hdfisher/dataconfig.py`. __Note__ that we have already applied a prior on $\tau$ of $\sigma(\tau) = 0.007$ to all pre-computed Fisher matrices that were calculated from CMB power spectra.


# Reproducing plots and tables in MacInnis et. al. 2023

We have provided a Jupyter notebook, `forecast_plots.ipynb`, which will reproduce the plots and tables in the main text of MacInnis et. al. (2023). This can also be used as an example of how to access the files that are provided with `hdlike`, and how to obtain parameter uncertainties from a Fisher matrix.


# Calculating new Fisher matrices

The Fisher matrix for a given set of parameters is calculated from a set of  derivatives of the mock signal with respect to each parameter, and a covariance matrix for the mock signal (described above). The calculations are done using the `Fisher` class of `hdlike/fisher.py`. We include additional functions in `hdlike/fisher.py` to add Fisher matrices, apply a Gaussian prior on a parameter, and remove (a) parameter(s) from a Fisher matrix. 

We provide a Jupyter notebook, `example_calculate_fisher_matrices.ipynb`, as an example of the Fisher matrix calculation (including the calculation of the derivates with respect to parameters) for the baseline CMB-HD forecasts. We also provide an example script, `example_calculate_fisher_derivatives.py`, that can be used to calculate the derivatives in parallel with MPI. In the notebook, we compare the results to the corresponding pre-computed Fisher matrix provided with `hdfisher`.

In the notebook, we use the fiducial cosmological parameters, parameter step sizes, and CAMB accuracy settings described in MacInnis et. al. (2023); this is the default behavior. You may also provide your own set of fiducial parameter values by saving them in a YAML file, and passing the file name as the `param_file` to the `Fisher` class. You may also provide your own step sizes in another YAML file, and pass its name as the `fisher_steps_file` to the `Fisher` class. See `example_calculate_fisher_matrices.ipynb` for more details and additional options. The parameter names must be names that can be passed to the CAMB function `camb.set_params()`.
