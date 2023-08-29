"""functions used to calculate a Fisher matrix"""
import os
from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import yaml
from . import utils, theory, config, dataconfig, mpi

# functions for derivatives of theory spectra with respect to params:

def get_param_info(param_file, fisher_steps_file):
    """Returns a dict of the fiducial cosmological parameters and a dict
    of their step sizes, which are used to calculate the derivatives of 
    the theory with respect to each varied parameter. 
    
    Parameters
    ----------
    param_file : str
        The file name, including the absolute path, of a YAML file that
        contains the cosmological parameter names and values.
    fisher_steps_file : str
        The absolute path to the YAML file holding the parameter step sizes.

    Returns
    -------
    fids : dict of float
        A dictionary with the parameter names as the keys holding their 
        fiducial value.
    step_sizes : dict of float
        A dictionary containing a subset of the parameters in `fids`,
        holding their absolute step size used to calculate the derivatives.

    Raises
    ------
    ValueError
        If there is a step size provided for a parameter without a fiducial 
        value.

    Note
    ----
    The fiducial parameters are loaded from the YAML `param_file`, with entries 
    in the format `param_name: fiducial_value`. The step sizes for a subset of 
    the fiducial parameters are loaded from the YAML `fisher_steps_file`. 
    Each parameter (with a  `param_name` corresponding to a fiducial 
    `param_name`) gets its own block, with two entries: `step_size`, with 
    a (float) value, and `step_type`, with a (str) value of either `'abs'` 
    for 'absolute' or `'rel'` for 'relative'. If the step type is relative, 
    the step size is assumed to be a fraction of the fiducial value; e.g., 
    a relative step size of 0.01 means increase/decrease the parameter 
    value by 1% of its fiducial value.
    """
    # load fiducial params
    fids = theory.get_params(param_file=param_file)
    # load step sizes for some or all of them
    with open(fisher_steps_file, 'r') as f:
        step_info = yaml.safe_load(f)
    # create a dict of absolute step sizes
    step_sizes = {}
    for param in step_info.keys():
        # make sure we have a fiducial value for that param
        if param not in fids.keys():
            raise ValueError(f"The parameter '{param}' is listed in the `fisher_steps_file` '{fisher_steps_file}', but you must also provide a fiducial value in the `param_file` '{param_file}'.")
        step = step_info[param]['step_size']
        if 'rel' in step_info[param]['step_type'].lower():
            step *= fids[param]
        step_sizes[param] = step
    return fids, step_sizes


def get_varied_param_values(fids, step_sizes, include_fid=True):
    """Returns a list of tuples containing the names of parameters that are 
    varied when calculating a Fisher matrix, and their values when they are 
    varied away from their fiducial value. 

    Parameters
    ----------
    fids : dict of float
        A dictionary with the parameter names and their fiducial values.
    step_sizes : dict of float
        A dictionary with a subset of the parameters in `fids` and the
        absolute step sizes to take when varying the parameter values
        up or down.
    include_fid : bool, default=True
        If `True`, include a tuple `(None, None)` in the list, corresponding
        to the fiducial case.

    Returns
    -------
    param_values : list of tuple of str, float
        A list containing two tuples for each varied parameter (i.e. for each
        key in `step_sizes`). Each tuple has the form 
        `(param_name, param_value)`, where 
        `param_value = fids[param_name] +/- step_sizes[param_name]`. 
        If `include_fid = True`, there is also a tuple `(None, None)` 
        corresponding to the fiducial set of parameters.
    """
    param_values = []
    if include_fid:
        param_values.append((None, None))
    for param in step_sizes.keys():
        param_values.append((param, fids[param] - step_sizes[param]))
        param_values.append((param, fids[param] + step_sizes[param]))
    return param_values


# functions to calculate the fisher matrix:

def get_available_cmb_fisher_derivs(derivs_dir, use_H0=False):
    """Returns the available `cmb_types` and `params` for the derivatives 
    of the lensed, unlensed, or delensed theory spectra with respect to
    cosmological parameters.

    Parameters
    ----------
    derivs_dir : str
        The directory containing the derivatives.
    use_H0: str, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.

    Returns
    -------
    cmb_types : list of str
        The available types of spectra: may include `'lensed'`, `'unlensed'`, 
        and `'delensed'`, if there are files containing those names.
    params : list of str
        The available parameters, set in the Fisher parameteer YAML file
        used to calculate the derivatives.

    Note
    ----
    This function assumes that the file names of the derivatives are in the 
    format returned by `hdfisher.config.fisher_cmb_deriv_fname`, and that the 
    list of parameters will be the same for each CMB type.
    """
    # get the file name template 
    full_derivs_fname = config.fisher_cmb_deriv_fname(derivs_dir, 'cmbtype', 'param', use_H0=use_H0)
    derivs_fname = os.path.basename(full_derivs_fname) # remove abs path
    derivs_fname_root, derivs_fname_ext = os.path.splitext(derivs_fname)   # remove extension
    # filename has info about cmb_type and param; find where each is located
    derivs_info = derivs_fname_root.split('_')
    cidx = derivs_info.index('cmbtype')
    pidx = derivs_info.index('param')
    # loop through the files in `derivs_dir` and find the ones that match the
    # format above; keep track of the different CMB types and parameters.
    cmb_types = []
    params = []
    for fname in os.listdir(derivs_dir):
        if ((not use_H0) and ('useH0' in fname)) or ((use_H0) and ('useH0' not in fname)):
            pass
        else:
            root, ext = os.path.splitext(fname)
            info = root.split('_')
            if (ext == derivs_fname_ext) and ('cls_deriv' in root):
                cmb_type = info[cidx]
                if cmb_type not in cmb_types:
                    cmb_types.append(cmb_type)
                if use_H0:
                    param = '_'.join(info[pidx:-1])
                else:
                    param = '_'.join(info[pidx:])
                if param not in params:
                    params.append(param)
    return cmb_types, params


def get_available_bao_fisher_derivs(derivs_dir, use_H0=False):
    """Returns the available `params` for the derivatives of the BAO theory 
    with respect to cosmological parameters.

    Parameters
    ----------
    derivs_dir : str
        The directory containing the derivatives.
    use_H0: str, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.

    Returns
    -------
    params : list of str
        The available parameters, set in the Fisher parameter YAML file
        used to calculate the derivatives.

    Note
    ----
    This function assumes that the file names of the derivatives are in the
    format returned by `hdfisher.config.fisher_bao_deriv_fname`.
    """
    # get the file name template 
    full_derivs_fname = config.fisher_bao_deriv_fname(derivs_dir, 'param', use_H0=use_H0)
    derivs_fname = os.path.basename(full_derivs_fname) # remove abs path
    derivs_fname_root, derivs_fname_ext = os.path.splitext(derivs_fname)   # remove extension
    # filename has info about cmb_type and param; find where each is located
    derivs_info = derivs_fname_root.split('_')
    pidx = derivs_info.index('param')
    # loop through the files in `derivs_dir` and find the ones that match the
    # format above; keep track of the different parameters.
    params = []
    for fname in os.listdir(derivs_dir):
        if ((not use_H0) and ('useH0' in fname)) or ((use_H0) and ('useH0' not in fname)):
            pass
        else:
            root, ext = os.path.splitext(fname)
            info = root.split('_')
            if (ext == derivs_fname_ext) and ('bao' in root):
                if use_H0:
                    param = '_'.join(info[pidx:-1])
                else:
                    param = '_'.join(info[pidx:])
                if param not in params:
                    params.append(param)
    return params


def load_cmb_fisher_derivs(derivs_dir, cmb_types=None, params=None,
                           spectra=['tt', 'te', 'ee', 'bb', 'kk'], use_H0=False,
                           lmin=None, lmax=None, bin_edges=None, ell_ranges=None):
    """Returns a dict containing the derivatives of the theory `spectra`
    for each CMB type in `cmb_types` with respect to each parameter in `params`.

    Parameters
    ----------
    derivs_dir : str
        The directory to load the files from.
    cmb_types : list of str, default=None
        A list of types of theory spectra: can include `'lensed'`, `'unlensed'`,
        `'delensed'`. If `None`, uses all available CMB types.
    params : list of str, default=None
        A list of parameter names. If `None`, uses all available parameters.
    spectra : list of str, default=['tt', 'te', 'ee', 'bb', 'kk']
        A list of spectra to return for each combination of CMB type and parameter.
    use_H0: str, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.
    lmin, lmax : int, default=None
        If not `None`, return the derivatives in the multipole range from `lmin`
        to `lmax`. Otherwise use the full multipole range.
    bin_edges : array_like of float or int, default=None
        If not `None`, use the bin edges to bin the derivatives between `lmin`
        and `lmax`. Otherwise the unbinned derivatives are returned.
    ell_ranges : dict of list or tuple of int, default=None
        A dictionary with keys `'tt'`, `'kk'`, etc., whose values are a list or
        tuple giving the minimum and maximum multipole values for each spectrum
        in `spectra`.


    Returns
    -------
    ells : array_like of float
        The multipoles at which the derivatives are calculated.
    derivs : nested dict of array_like of float
        A nested dictionary of the form `derivs[cmb_type][param][spec]` for each
        `cmb_type` in `cmb_types`, `param` in `params`, and `spec` in `spectra`,
        which holds the derivative of that spectrum with respect to the
        parameter, d(C_ell^XY) / d(param).

    Raises
    ------
    ValueError
        If any `cmb_type` in `cmb_types` is not `'lensed'`, `'unlensed'`, or
        `'delensed'`.
    
    Note
    ----
    This function assumes that the file names of the derivatives are in the 
    format returned by `hdfisher.config.fisher_cmb_deriv_fname`, and that the 
    list of parameters will be the same for each CMB type.
    """
    valid_cmb_types = ['lensed', 'unlensed', 'delensed']
    cols = theory.Theory.theo_cols # all columns in derivs file
    all_cmb_types, all_params = get_available_cmb_fisher_derivs(derivs_dir, use_H0=use_H0)
    if cmb_types is None:
        cmb_types = all_cmb_types
    if params is None:
        params = all_params
    derivs = {}
    for cmb_type in cmb_types:
        cmb_type = cmb_type.lower()
        if cmb_type not in valid_cmb_types:
            raise ValueError(f"You passed an unknown CMB type `'{cmb_type}'` in `cmb_types`. The options are {valid_cmb_types}.")
        derivs[cmb_type] = {}
        for param in params:
            derivs[cmb_type][param] = {}
            derivs_fname = config.fisher_cmb_deriv_fname(derivs_dir, cmb_type, param, use_H0=use_H0)
            derivs_dict = utils.load_from_file(derivs_fname, cols)
            ells = derivs_dict['ells']
            # trim the multipole range
            ell_min = ells[0] if (lmin is None) else lmin
            ell_max = ells[-1] if (lmax is None) else lmax
            loc = np.where((ells >= ell_min) & (ells <= ell_max))
            ells = ells[loc]
            for s in spectra:
                derivs[cmb_type][param][s] = derivs_dict[s.lower()][loc].copy()
            if bin_edges is not None:
                ells, derivs[cmb_type][param] = utils.bin_theo_dict(ells, derivs[cmb_type][param], bin_edges, lmin=lmin, lmax=lmax, ell_ranges=ell_ranges)
    return ells, derivs


def load_bao_fisher_derivs(derivs_dir, params=None, use_H0=False):
    """Returns a dict containing the derivatives of the BAO theory with
    respect to each parameter in `params`.

    Parameters
    ----------
    derivs_dir : str
        The directory to load the files from.
    params : list of str, default=None
        A list of parameter names. If `None`, uses all available parameters.
    use_H0: str, default=False
        If `True`, look for derivatives that were calculated by passing `'H0'`
        to CAMB when varying the other parameters, as opposed to passing
        `'cosmomc_theta'`.


    Returns
    -------
    z : array_like of float
        The redshifts at which the derivatives were calculated.
    derivs : dict of array_like of float
        A dictionary whose keys are the parameter names (same as the elements
        of `params`, if it was provided), holding the derivative of the BAO
        theory with respect to the each parameter.

    Note
    ----
    This function assumes that the file names of the derivatives are in the
    format returned by `hdfisher.config.fisher_bao_deriv_fname`, and that all
    derivatives were calculated for the same redshifts, returned in `z`.
    """
    if params is None:
        params = get_available_bao_fisher_derivs(derivs_dir, use_H0=use_H0)
    derivs = {}
    for param in params:
        derivs_fname = config.fisher_bao_deriv_fname(derivs_dir, param, use_H0=use_H0)
        z, derivs[param] = np.loadtxt(derivs_fname, unpack=True)
    return z, derivs


def calc_cmb_fisher(cmb_cov, derivs, params, spectra=['tt', 'te', 'ee', 'bb', 'kk'], priors=None, fname=None):
    """Calculate the Fisher matrix for the given set of `params` using the
    covariance matrix `cmb_cov`, and the derivatives (`derivs`) of the theory
    `spectra` with respect to each parameter.

    Parameters
    ----------
    cmb_cov : array_like of float
        The two-dimensional covariance matrix of the CMB and CMB lensing
        power spectra, with blocks in the order given by `spectra`.
    derivs: dict of dict of array_like of float
        The derivatives of the power spectra with respect to each parameter.
        The first set of keys should be the same as the parameter names in
        `params`, and the second set should be the same as the names in
        `spectra`. Each array should be the same length as one block of 
        `cmb_cov`, i.e. the derivatives should be binned in the same way 
        as the covariance matrix.
    params : list of str
        The names of the parameters to use.
    spectra : list of str
        The names of the spectra to use.
    priors : dict of float, default=None
        If not `None`, provide any Gaussian priors with the parameter name
        as the key and the prior as the value, e.g. `{'tau': 0.007}`.
    fname : str, default=None
        If not `None`, save the Fisher matrix to the file name (including the
        absolute path) given by `fname`, which is assumed to be a `.txt` file.

    Returns
    -------
    fisher_matrix : array_like of float
        The Fisher matrix, with elements in the order given by `params`.
    """
    invcov = np.linalg.inv(cmb_cov)
    priors = {} if (priors is None) else priors
    # put derivatives into a single vector for each parameter
    dcls = {}
    for param in params:
        dcls[param] = np.hstack([derivs[param][s] for s in spectra])
    # calculate the fisher matrix
    nparams = len(params)
    fisher_matrix = np.zeros((nparams, nparams))
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            if j >= i:
                tmp = invcov @ dcls[p2]
                fisher_matrix[i, j] = dcls[p1] @ tmp
                if j > i:
                    fisher_matrix[j, i] = fisher_matrix[i, j]
                if (i == j) and (p1 in priors.keys()):
                    fisher_matrix[i, i] += (1. / priors[p1]**2)
    if fname is not None:
        save_fisher_matrix(fname, fisher_matrix, params)
    return fisher_matrix


def calc_bao_fisher(bao_cov, derivs, params, priors=None, fname=None):
    """Calculate the Fisher matrix for the given set of `params` using the
    covariance matrix `bao_cov`, and the derivatives (`derivs`) of the BAO
    theory with respect to each parameter.

    Parameters
    ----------
    bao_cov : array_like of float
        The two-dimensional covariance matrix for the mock BAO data.
    derivs: dict of array_like of float
        The derivatives of the BAO theory with respect to each parameter.
        The keys are the parameter names, and each name in `params` should 
        appear as a key in `derivs`. Each array should be the same length 
        as the `bao_cov`, i.e. they must be calculated at the same redshifts.
    params : list of str
        The names of the parameters to use.
    priors : dict of float, default=None
        If not `None`, provide any Gaussian priors with the parameter name
        as the key and the prior as the value, e.g. `{'tau': 0.007}`.
    fname : str, default=None
        If not `None`, save the Fisher matrix to the file name (including the
        absolute path) given by `fname`, which is assumed to be a `.txt` file.

    Returns
    -------
    fisher_matrix : array_like of float
        The Fisher matrix, with elements in the order given by `params`.
    """
    invcov = np.linalg.inv(bao_cov)
    priors = {} if (priors is None) else priors
    # calculate the fisher matrix
    nparams = len(params)
    fisher_matrix = np.zeros((nparams, nparams))
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            if j >= i:
                tmp = invcov @ derivs[p2].copy()
                fisher_matrix[i, j] = derivs[p1].copy() @ tmp
                if j > i:
                    fisher_matrix[j, i] = fisher_matrix[i, j]
                if (i == j) and (p1 in priors.keys()):
                    fisher_matrix[i, i] += (1. / priors[p1]**2)
    if fname is not None:
        save_fisher_matrix(fname, fisher_matrix, params)
    return fisher_matrix



# other useful functions:

def save_fisher_matrix(fname, fisher_matrix, params):
    """Save the Fisher matrix to the given file, with a header giving the 
    parameter names in the correct order.

    Parameters
    ----------
    fname : str
        The file name (including absolute path) to save the Fisher matrix to.
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    params : list of str
        A list of parameter names, in the same order as in the Fisher matrix.
    """
    header = ' '.join(params)
    np.savetxt(fname, fisher_matrix, header=header)


def load_fisher_matrix(fname):
    """Returns the Fisher matrix loaded from the given file, with a list of 
    parameter names in the correct order.

    Parameters
    ----------
    fname : str
        The file name (including absolute path) to save the Fisher matrix to.

    Returns
    -------
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    params : list of str
        A list of parameter names, in the same order as in the Fisher matrix.
    """
    fisher_matrix = np.loadtxt(fname)
    # read the header to get the params
    with open(fname, 'r') as f:
        header = f.readline()
    header = header.strip('# \n')
    params = header.split(' ')
    return fisher_matrix, params


def get_fisher_errors(fisher_matrix, params, fname=None):
    """Returns the 1-sigma error bars on the parameters from the Fisher matrix.
    
    Parameters
    ----------
    fisher_matrix : array_like of float
        The two-dimensional Fisher matrix.
    params : list of str
        A list of parameter names, in the same order as in the Fisher matrix.
    fname : str, default=None
        If not `None`, save the Fisher errors to the file name given by `fname`,
        assumed to be a YAML file.

    Returns
    -------
    errors : dict of float
        A dictionary with the keys given by the names in `params` and values
        holding the error bar for that parameter.
    """
    cov = np.linalg.inv(fisher_matrix.copy())
    errs = np.sqrt(np.diag(cov))
    errors = {}
    for i, param in enumerate(params):
        errors[param] = float(errs[i]) # convert from numpy dtype, for yaml saving
    if fname is not None:
        with open(fname, 'w') as f:
            yaml.dump(errors, f)
    return errors


def add_fishers(fisher_matrix1, fisher_params1, 
                fisher_matrix2, fisher_params2, 
                priors=None):
    # TODO: docstring
    # convert each Fisher matrix to a `pandas.DataFrame` instance to
    # deal with missing params, different ordering, etc.
    df1 = pd.DataFrame(fisher_matrix1.copy(), columns=fisher_params1.copy(), index=fisher_params1.copy(), copy=True)
    df2 = pd.DataFrame(fisher_matrix2.copy(), columns=fisher_params2.copy(), index=fisher_params2.copy(), copy=True)
    df_sum = df1.add(df2, fill_value=0)
    if priors is None:
        fisher_matrix = df_sum.values.copy()
        fisher_params = df_sum.columns.tolist().copy()
    else:
        params_with_prior = list(priors.keys())
        n_priors = len(params_with_prior)
        prior_matrix = np.zeros((n_priors, n_priors))
        for i, p in enumerate(params_with_prior):
            prior_matrix[i, i] += 1 / priors[p]**2
        df_prior = pd.DataFrame(prior_matrix.copy(), columns=params_with_prior.copy(), index=params_with_prior.copy())
        df_final = df_sum.add(df_prior, fill_value=0)
        fisher_matrix = df_final.values.copy()
        fisher_params = df_final.columns.tolist().copy()
    return fisher_matrix, fisher_params

def add_priors(fisher_matrix, fisher_params, priors):
    # TODO: docstring
    fisher_with_prior = fisher_matrix.copy()
    for i, param in enumerate(fisher_params):
        if param in priors.keys():
            fisher_with_prior[i, i] += 1 / priors[param]**2
    return fisher_with_prior

def remove_params(fisher_matrix, fisher_params, params_to_remove):
    # TODO: docstring
    df = pd.DataFrame(fisher_matrix.copy(), columns=fisher_params.copy(), index=fisher_params.copy(), copy=True)
    df.drop(index=params_to_remove, columns=params_to_remove, inplace=True)
    new_fisher_matrix = df.values.copy()
    new_fisher_params = df.columns.tolist().copy()
    return new_fisher_matrix, new_fisher_params

def remove_zeros(fisher_matrix, fisher_params):
    # TODO: docstring
    nparams = len(fisher_params)
    params_to_remove = []
    for i, param in enumerate(fisher_params):
        if all(np.isclose(np.zeros(nparams), fisher_matrix[i])):
            params_to_remove.append(param)
    new_fisher_matrix, new_fisher_params = remove_params(fisher_matrix, fisher_params, params_to_remove)
    return new_fisher_matrix, new_fisher_params
    


    
def get_common_params(list_of_dicts):
    """Returns a list of parameter names that appear as keys for each 
    dictionary in the `list_of_dicts`.
    
    Parameters
    ----------
    list_of_dicts : list of dict
        A list of dictionaries whose keys are parameter names.

    Returns
    -------
    params : list of str
        A list of parameter names that the dictionaries share in common.
    """
    all_params = []
    for d in list_of_dicts:
        all_params += list(d.keys())
    params = []
    for param in set(all_params):
        if all([param in d.keys() for d in list_of_dicts]):
            params.append(param)
    return params



class Fisher:
    def __init__(self, exp, fisher_dir, overwrite=False, fisher_params=None, 
            param_file=None, fisher_steps_file=None, feedback=False, 
            use_H0=False, hd_lmax=None, include_fg=True):
        # TODO: docstring
        self.datalib = dataconfig.Data()
        self.fisher_dir = fisher_dir
        self.overwrite = overwrite
        self.use_H0 = use_H0
        # check if we have a covariance matrix for this experiment
        self.exp = exp.lower()
        if exp not in self.datalib.cmb_exps:
            raise ValueError(f"Invalid `exp`: '{exp}'. Valid options are: {datalib.cmb_exps}.")
        # warn about ignored arguments, and which covmats are available
        if exp != 'hd':
            warnings.warn(f"Ignoring the `hd_lmax` and `include_fg` arguments for `exp = '{exp}'`.")
            warnings.warn(f"NOTE that we only have mock covariance matrices for *delensed* (as opposed to lensed) CMB spectra for `exp = '{exp}'` (i.e., we can only calculate a Fisher matrix from delensed power spectra in this case).")
        else:
            if not include_fg:
                warnings.warn("NOTE that for CMB-HD, we only have mock covariance matrix that excludes the effects of foregrounds for mock lensed power spectra  (i.e., we can only calculate a Fisher matrix from lensed power spectra in this case).")
            if hd_lmax is not None:
                warnings.warn("NOTE that for CMB-HD, we only have mock covariance matrix calculated with a lower `hd_lmax` for *delensed* (as opposed to lensed) CMB spectra (i.e., we can only calculate a Fisher matrix from delensed power spectra in this case).")
        # set the multipole limits for the CMB theory calculation
        self.ell_ranges = deepcopy(self.datalib.ell_ranges[self.exp])
        self.Lmin = self.datalib.lmins[exp]
        if (self.exp == 'hd') and (hd_lmax is not None):
            valid_lmax_vals = self.datalib.hd_lmaxs + [self.datalib.lmaxs['hd']]
            if int(hd_lmax) in valid_lmax_vals:
                self.lmax = int(hd_lmax)
                self.Lmax = int(hd_lmax)
                # update maximum multipole for each spectrum type (TT, TE, ...)
                for spec_type in self.ell_ranges.keys():
                    self.ell_ranges[spec_type][1] = int(hd_lmax)
            else:
                raise ValueError(f"Invalid `hd_lmax`: {hd_lmax}. Valid options are: {valid_lmax_vals}")
        else:
            self.lmax = self.datalib.lmaxs[self.exp]
            self.Lmax = self.datalib.Lmaxs[self.exp]
        self.z = self.datalib.desi_redshifts() # for BAO theory calculation
        # get the lensing noise, needed to calculate delensed CMB theory:
        self.include_fg = include_fg
        hd_Lmax = self.Lmax if (self.exp == 'hd') else None
        _, self.nlkk = self.datalib.load_cmb_lensing_noise_spectrum(self.exp, include_fg=self.include_fg, hd_Lmax=hd_Lmax)
        # get the fiducial parameter values and step sizes:
        self.param_file = param_file
        if self.param_file is None:
            self.param_file = self.get_param_file(feedback=feedback)
        if fisher_steps_file is None:
            fisher_steps_file = self.get_fisher_steps_file(feedback=feedback)

        self.fid_params, self.step_sizes = get_param_info(self.param_file, fisher_steps_file)
        mpi.comm.barrier()
        # create the output directories, and save a copy of the input param 
        #  and step sizes files:
        self.theo_dir = os.path.join(self.fisher_dir, 'theory')
        self.derivs_dir = os.path.join(self.fisher_dir, 'derivs')
        self.fmat_dir = os.path.join(self.fisher_dir, 'fisher_matrices')
        if mpi.rank == 0: 
            utils.set_dir(self.fisher_dir)
            utils.set_dir(self.theo_dir)
            utils.set_dir(self.derivs_dir)
            utils.set_dir(self.fmat_dir)
            self.save_param_steps_values()
        mpi.comm.barrier()
        # list of parameter names for varied parameters:
        self.all_varied_params = list(self.step_sizes.keys())
        if fisher_params is None:
            self.fisher_params = list(self.step_sizes.keys())
        else:
            self.fisher_params = fisher_params.copy()
        # make sure we have a step size for that parameter:
        for param in self.fisher_params:
            if param not in self.step_sizes.keys():
                raise ValueError(f"Invalid parameter `'{param}'` in `fisher_params`: you must provide a fiducial value and step size for `'{param}'` in the `param_file` and `fisher_steps_file`, respectively.")
        # remove any fixed params (i.e., not included in `fisher_params`) 
        #  from the `step_sizes` dict:
        step_size_keys = list(self.step_sizes.keys())
        for param in step_size_keys:
            if param not in self.fisher_params:
                self.step_sizes.pop(param, None)


    # functions called during initialization:

    def get_param_file(self, feedback=False):
        # TODO: docstring
        # look for an input param file in the output directory:
        input_param_file = os.path.join(self.fisher_dir, 'fiducial_params.yaml')
        if os.path.exists(input_param_file) and (not self.overwrite):
            param_file = input_param_file
        else:
            param_file = self.datalib.fiducial_param_file(feedback=feedback)
        return param_file


    def get_fisher_steps_file(self, feedback=False):
        # TODO: docstring
        # look for an input param file in the output directory:
        input_steps_file = os.path.join(self.fisher_dir, 'step_sizes.yaml')
        if os.path.exists(input_steps_file) and (not self.overwrite):
            fisher_steps_file = input_steps_file
        else:
            fisher_steps_file = self.datalib.fiducial_fisher_steps_file(feedback=feedback)
        return fisher_steps_file


    def save_param_steps_values(self):
        # TODO: docstring
        param_values = self.fid_params.copy()
        param_steps = self.step_sizes.copy()
        # ensure float values are actually `float` (not, e.g., `numpy.float64`):
        for param, value in param_values.items():
            if type(value) not in [str, bool, int]:
                param_values[param] = float(value)
        for param, step_size in param_steps.items():
            # all step sizes should be floating point numbers
            param_steps[param] = {'step_size': float(step_size), 'step_type': 'absolute'}
        # save the files
        if mpi.rank == 0: 
            fid_params_fname = os.path.join(self.fisher_dir, 'fiducial_params.yaml')
            with open(fid_params_fname, 'w') as f:
                yaml.dump(param_values, f,  default_flow_style=False)
            step_sizes_fname = os.path.join(self.fisher_dir, 'step_sizes.yaml')
            with open(step_sizes_fname, 'w') as f:
                yaml.dump(param_steps,  f,  default_flow_style=False)


    def check_H0_theta(self):
        # TODO: docstring
        has_H0 = 'H0' in self.fisher_params
        has_theta = ('theta' in self.fisher_params) or ('cosmomc_theta' in self.fisher_params)
        if has_H0 and has_theta:
            if self.use_H0:
                theta_key = 'theta' if ('theta' in self.fisher_params) else 'cosmomc_theta'
                warnings.warn(f"Removing '{theta_key}' from the list of varied parameters, because `use_H0=True`.")
                self.fid_params.pop(theta_key, None)
                self.step_sizes.pop(theta_key, None)
                self.fisher_params.remove(theta_key)
            else:
                warnings.warn("Removing 'H0' from the list of varied parameters, because `use_H0=False`.")
                self.fid_params.pop('H0', None)
                self.step_sizes.pop('H0', None)
                self.fisher_params.remove('H0')
            

    # functions to calculate the derivatives of theory with respect to params:

    def get_varied_param_values(self):
        # TODO: docstring
        # we need to choose between H0 or theta, if both are in `fisher_params`;
        #  the choice is based on the `use_H0` flag:
        self.check_H0_theta()
        all_param_values = get_varied_param_values(self.fid_params, self.step_sizes)
        if not self.overwrite: # don't re-compute theory and derivatives
            param_values = []
            for param_info in all_param_values:
                param, value = param_info
                if param is None: # fiducial case
                    step_direction = None
                else:
                    step_direction = 'up' if (value > self.fid_params[param]) else 'down'
                # check if CMB and BAO theory is already saved
                theo_fnames = [config.fisher_bao_theo_fname(self.theo_dir, param, step_direction, use_H0=self.use_H0)]
                for cmb_type in self.datalib.cmb_types:
                    theo_fnames.append(config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, step_direction, use_H0=self.use_H0))
                if not all([os.path.exists(fname) for fname in theo_fnames]):
                    param_values.append((param, value))
        else:
            param_values = all_param_values
        mpi.comm.barrier()
        return param_values

    
    def calculate_fisher_derivs(self):
        # TODO: docstring
        # get a list of varied parameter names and values:
        param_values = self.get_varied_param_values()
        # distribute the theory calculations among the mpi processes, by
        #  getting a list of indices in `varied_param_values` that this MPI
        #  process will use:
        ntasks = len(param_values)
        task_idxs = mpi.distribute(ntasks, mpi.size, mpi.rank)
        # loop through the theory calculations, and save the theory for each:
        for idx in task_idxs:
            param, value = param_values[idx]
            self.calculate_theory_for_deriv(param, value)
        mpi.comm.barrier()
        # load the theory to calculate and save the derivatives:
        if mpi.rank == 0:
            for param in self.fisher_params:
                self.calculate_deriv(param)


    def calculate_theory_for_deriv(self, param, value):
        # TODO: docstring
        # will pass the `cosmo_params` dict to the `theory.Theory` class, to
        #  overide the param value from its fiducial value; we also keep track
        #  of which way the parameter is varied to use in the output file name:
        if param is None: # fiducial case
            cosmo_params = {}
            step_direction = None
        else:
            cosmo_params = {param: value}
            step_direction = 'up' if (value > self.fid_params[param]) else 'down'
            if param == 'logA': # set 'As' to `None`, so 'logA' is actually used
                cosmo_params['As'] = None
        # do the calculation:
        theolib = theory.Theory(self.lmax, self.theo_dir, param_file=self.param_file, nlkk=self.nlkk, recon_lmin=self.Lmin, recon_lmax=self.Lmax, use_H0=self.use_H0, **cosmo_params)
        cmb_theo = theolib.get_theory(save=False)
        rs_dv = theolib.get_rs_dv(self.z, save=False)
        # save the theory
        header_info = f'{param} = {value}\n'
        for cmb_type in self.datalib.cmb_types:
            cmb_theo_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, step_direction, use_H0=self.use_H0)
            utils.save_to_file(cmb_theo_fname, cmb_theo[cmb_type], keys=self.datalib.theo_cols, extra_header_info=header_info)
        bao_theo_fname = config.fisher_bao_theo_fname(self.theo_dir,  param, step_direction, use_H0=self.use_H0)
        utils.save_to_file(bao_theo_fname, {'z': self.z, 'rs_dv': rs_dv}, keys=['z', 'rs_dv'],  extra_header_info=header_info)



    def calculate_deriv(self, param):
        # TODO: docstring
        delta_param = 2 * self.step_sizes[param]
        header_info = f'{param}: fiducial = {self.fid_params[param]}, step size = {self.step_sizes[param]}\n'
        # BAO:
        bao_theo_up_fname = config.fisher_bao_theo_fname(self.theo_dir, param, 'up', use_H0=self.use_H0)
        z, rs_dv_up = np.loadtxt(bao_theo_up_fname, unpack=True)
        bao_theo_down_fname = config.fisher_bao_theo_fname(self.theo_dir, param, 'down', use_H0=self.use_H0)
        z, rs_dv_down = np.loadtxt(bao_theo_down_fname, unpack=True)
        rs_dv_deriv = (rs_dv_up - rs_dv_down) / delta_param
        bao_fname = config.fisher_bao_deriv_fname(self.derivs_dir, param, use_H0=self.use_H0)
        utils.save_to_file(bao_fname, {'z': z, 'rs_dv': rs_dv_deriv}, keys=['z', 'rs_dv'], extra_header_info=header_info)
        # CMB:
        for cmb_type in self.datalib.cmb_types:
            cmb_theo_up_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, 'up', use_H0=self.use_H0)
            cmb_theo_up = utils.load_from_file(cmb_theo_up_fname, self.datalib.theo_cols)
            cmb_theo_down_fname = config.fisher_cmb_theo_fname(self.theo_dir, cmb_type, param, 'down', use_H0=self.use_H0)
            cmb_theo_down = utils.load_from_file(cmb_theo_down_fname, self.datalib.theo_cols)
            cmb_derivs = {'ells': cmb_theo_up['ells'].copy()}
            for s in self.datalib.cov_spectra:
                cmb_derivs[s] = (cmb_theo_up[s] - cmb_theo_down[s]) / delta_param
            cmb_fname = config.fisher_cmb_deriv_fname(self.derivs_dir, cmb_type, param, use_H0=self.use_H0)
            utils.save_to_file(cmb_fname, cmb_derivs, keys=self.datalib.theo_cols,  extra_header_info=header_info)


    # functions to calculate the Fisher matrix:

    def load_cmb_fisher_derivs(self, cmb_types=None, binned=False, use_H0=None):
        # TODO: docstring
        if use_H0 is None:
            use_H0 = self.use_H0
        spectra = self.datalib.cov_spectra
        if binned:
            bin_edges = self.datalib.load_bin_edges()
            ell_ranges = self.ell_ranges
        else:
            bin_edges = None
            ell_ranges = None
        ells, derivs = load_cmb_fisher_derivs(self.derivs_dir, cmb_types=cmb_types, spectra=spectra.copy(), use_H0=use_H0, bin_edges=bin_edges, ell_ranges=ell_ranges)
        return ells, derivs

    
    def load_bao_fisher_derivs(self, use_H0=None):
        # TODO: docstring
        if use_H0 is None:
            use_H0 = self.use_H0
        z, derivs = load_bao_fisher_derivs(self.derivs_dir, use_H0=use_H0)
        return z, derivs

    
    def calc_cmb_fisher(self, cmb_type, params=None, priors=None, use_H0=None, save=False, fname=None):
        # TODO: docstring
        if use_H0 is None:
            use_H0 = self.use_H0
        if save and (fname is None):
            raise ValueError(f"You set `save=True` but `fname` is None: you must provide a file name, `fname`, that will be used to save the Fisher matrix in the directory `{self.fmat_dir}`.")
        self.check_H0_theta()
        _, derivs = self.load_cmb_fisher_derivs(cmb_types=[cmb_type], binned=True, use_H0=use_H0)
        hd_lmax = self.lmax if (self.exp == 'hd') else None
        cmb_covmat = self.datalib.load_cmb_covmat(self.exp, cmb_type=cmb_type, include_fg=self.include_fg, hd_lmax=hd_lmax)
        if params is None:
            params = self.fisher_params.copy()
        has_H0 = 'H0' in params
        has_theta = ('theta' in params) or ('cosmomc_theta' in params)
        if has_H0 and has_theta:
            raise ValueError(f"Both 'H0' and '{theta_key}' are in `params`: only one can be used.")
        elif use_H0 and has_theta:
            theta_key = 'theta' if ('theta' in params) else 'cosmomc_theta'
            theta_idx = params.index(theta_key)
            warnings.warn(f"Replacing '{theta_key}' with 'H0' because `use_H0=True`.")
            params[theta_idx] = 'H0'
        elif (not use_H0) and has_H0:
            theta_key = 'theta' if ('theta' in self.all_varied_params) else 'cosmomc_theta'
            H0_idx = params.index('H0')
            warnings.warn(f"Replacing 'H0' with '{theta_key}' because `use_H0=False`.")
            params[H0_idx] = theta_key
        if save and (mpi.rank == 0):
            fisher_fname = os.path.join(self.fmat_dir, fname)
        else:
            fisher_fname = None
        fisher_matrix = calc_cmb_fisher(cmb_covmat, derivs[cmb_type], params, spectra=self.datalib.cov_spectra, priors=priors, fname=fisher_fname)
        return fisher_matrix.copy(), params.copy()


    def calc_bao_fisher(self, params=None, priors=None, use_H0=None, save=False, fname=None):
        # TODO: docstring
        if use_H0 is None:
            use_H0 = self.use_H0
        if save and (fname is None):
            raise ValueError(f"You set `save=True` but `fname` is None: you must provide a file name, `fname`, that will be used to save the Fisher matrix in the directory `{self.fmat_dir}`.")
        self.check_H0_theta()
        _, derivs = self.load_bao_fisher_derivs(use_H0=use_H0)
        bao_covmat = self.datalib.load_desi_covmat()
        if params is None:
            params = self.fisher_params.copy()
        has_H0 = 'H0' in params
        has_theta = ('theta' in params) or ('cosmomc_theta' in params)
        if has_H0 and has_theta:
            raise ValueError(f"Both 'H0' and '{theta_key}' are in `params`: only one can be used.")
        elif use_H0 and has_theta:
            theta_key = 'theta' if ('theta' in params) else 'cosmomc_theta'
            theta_idx = params.index(theta_key)
            warnings.warn(f"Replacing '{theta_key}' with 'H0' because `use_H0=True`.")
            params[theta_idx] = 'H0'
        elif (not use_H0) and has_H0:
            theta_key = 'theta' if ('theta' in self.all_varied_params) else 'cosmomc_theta'
            H0_idx = params.index('H0')
            warnings.warn(f"Replacing 'H0' with '{theta_key}' because `use_H0=False`.")
            params[H0_idx] = theta_key
        if save and (mpi.rank == 0):
            fisher_fname = os.path.join(self.fmat_dir, fname)
        else:
            fisher_fname = None
        fisher_matrix = calc_bao_fisher(bao_covmat, derivs, params, priors=priors, fname=fname)
        return fisher_matrix.copy(), params.copy()
        
    
    def get_fisher(self, cmb_type='delensed', params=None, priors=None, use_H0=None, with_desi=False, save=False, fname=None):
        # TODO: docstring
        if use_H0 is None:
            use_H0 = self.use_H0
        if save and (fname is None):
            raise ValueError(f"You set `save=True` but `fname` is None: you must provide a file name, `fname`, that will be used to save the Fisher matrix in the directory `{self.fmat_dir}`.")
        # check if we need to calculate a new Fisher matrix, or if we can load it:
        calc_fisher = True
        if fname is not None:
            fisher_fname = os.path.join(self.fmat_dir, fname)
            if os.path.exists(fisher_fname) and (not self.overwrite):
                fisher_matrix, fisher_params = load_fisher_matrix(fisher_fname)
                calc_fisher = False
        if calc_fisher:
            if with_desi:
                cmb_fisher_matrix, cmb_fisher_params = self.calc_cmb_fisher(cmb_type, params=params, use_H0=use_H0, save=False)
                bao_fisher_matrix, bao_fisher_params = self.calc_bao_fisher(params=params, use_H0=use_H0, save=False)
                fisher_matrix, fisher_params = add_fishers(cmb_fisher_matrix, cmb_fisher_params, bao_fisher_matrix, bao_fisher_params, priors=priors) 
            else:
                fisher_matrix, fisher_params = self.calc_cmb_fisher(cmb_type, params=params, priors=priors, use_H0=use_H0, save=False)
            if save and (mpi.rank == 0):
                fisher_fname = os.path.join(self.fmat_dir, fname)
                save_fisher_matrix(fisher_fname, fisher_matrix, fisher_params)
        return fisher_matrix, fisher_params


    def get_fisher_errors(self, cmb_type='delensed', params=None, priors=None, use_H0=None, with_desi=False, save=False, fname=None):
        # TODO: docstring
        if use_H0 is None:
            use_H0 = self.use_H0
        fisher_matrix, fisher_params = self.get_fisher(cmb_type=cmb_type, params=params, priors=priors, use_H0=use_H0, with_desi=with_desi, save=save, fname=fname)
        fisher_errors = get_fisher_errors(fisher_matrix, fisher_params)
        return fisher_errors



