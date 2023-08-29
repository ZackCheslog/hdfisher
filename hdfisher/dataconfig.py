import os
import warnings
import numpy as np
from . import utils, fisher, theory


class Data:
    """Holds the experimental configuration information for the experiments
    considered in MacInnis et. al. (2023), and provides access to their 
    associated files that are provided with `hdfisher`.
    """
    
    def __init__(self):
        """Initialization of the experimental configurations."""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
        self.data_path = lambda x: os.path.join(data_dir, x)

        # constants
        self.cmb_exps = ['so', 's4', 'hd']
        self.fsky = 0.6
        
        # columns for spectra in files
        self.theo_cols = ['ells', 'tt', 'te', 'ee', 'bb', 'kk']
        self.noise_cols = self.theo_cols[:-1].copy()
        self.fg_cols = ['ells', 'ksz', 'tsz', 'cib', 'radio'] # note that kSZ is reionization kSZ
        
        self.cmb_spectra = self.theo_cols[1:-1].copy()
        self.cov_spectra = self.theo_cols[1:].copy()
        
        # for theory spectra
        self.cmb_types = ['lensed', 'delensed', 'unlensed']
        
        # multipole ranges
        self.lmins = {'hd': 30, 'so': 30, 's4': 30}
        self.lmaxs = {'hd': 20100, 'so': 5000, 's4': 5000}
        self.lmaxsTT = {'hd': 20100, 'so': 3000, 's4': 3000}
        self.Lmaxs = {'hd': 20100, 'so': 3000, 's4': 3000}
        
        self.ell_ranges = {}
        for exp in self.cmb_exps:
            self.ell_ranges[exp] = {'tt': [self.lmins[exp], self.lmaxsTT[exp]],
                                    'te': [self.lmins[exp], self.lmaxs[exp]],
                                    'ee': [self.lmins[exp], self.lmaxs[exp]],
                                    'bb': [self.lmins[exp], self.lmaxs[exp]],
                                    'kk': [self.lmins[exp], self.Lmaxs[exp]]}
        
        # CMB-HD has some additional files calculated for a lower maximum multipole
        self.hd_lmaxs = [1000, 3000, 5000, 10000]
        
        # file names
        self.bin_edges_fname = self.data_path('bin_edges.txt')


    # ----- functions to check arguments passed to functions: -----

    def check_cmb_exp(self, exp, valid_exps=None):
        # TODO: docstring
        exp = exp.lower()
        if valid_exps is None:
            valid_exps = self.cmb_exps
        if exp not in valid_exps:
            raise ValueError(f"Invalid experiment name. You passed `exp = '{exp}'`, but `exp` must be one of {valid_exps}.")
        else:
            return exp

    def check_hd_lmax(self, hd_lmax):
        # TODO: docstring
        hd_lmax = int(hd_lmax)
        valid_hd_lmaxs = self.hd_lmaxs + [self.lmaxs['hd']]
        if hd_lmax not in valid_hd_lmaxs:
            raise ValueError(f"Invalid `hd_lmax`: {hd_lmax}. Valid options are: {valid_hd_lmaxs}.")
        else:
            return hd_lmax


    # ----- functions that resturn file names of data: -----

    def fiducial_param_file(self, feedback=False):
        """Returns the name of the YAML file holding the fiduical cosmological
        and accuracy parameters that are passed to CAMB when calculating the 
        CMB and BAO theory.

        Parameters
        ----------
        feedback : bool, default=False
            If `True`, the parameter file sets the CAMB `halofit_version`
            to `mead2020_feedback`, i.e. uses the HMCode 2020 + feedback
            non-linear model. Otherwise, the HMCode 2016 CDM-only model
            is used by setting `halofit_version` to `mead2016`.

        Returns
        -------
        fname : str
            The name of the parameter file, including its absolute path.
        """
        fid_params_dir = self.data_path('fiducial_params')
        fname = 'fiducial_params_feedback.yaml' if feedback else 'fiducial_params.yaml'
        return os.path.join(fid_params_dir, fname)

    
    def fiducial_fisher_steps_file(self, feedback=False):
        """Returns the name of the YAML file holding the fiduical parameter 
        step sizes used to calculate the Fisher matrices.

        Parameters
        ----------
        feedback : bool, default=False
            If `True`, the file includes a step size for the HMCode 2020 
            baryonic feedback parameter, `HMCode_logT_AGN`. Otherwise this 
            parameter is excluded.

        Returns
        -------
        fname : str
            The name of the file, including its absolute path.
        """
        fid_steps_dir = self.data_path('fisher_step_sizes')
        fname = 'fiducial_step_sizes_feedback.yaml' if feedback else 'fiducial_step_sizes.yaml'
        return os.path.join(fid_steps_dir, fname)


    def cmb_theory_fname(self, exp, spectrum_type, hd_lmax=None, feedback=False):
        """Returns the name of the file containing the theory CMB and lensing
        spectra for a given CMB experiment and CMB type (e.g. delensed).
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        spectrum_type : str
            The name of the kind of spectra. Must be either `'lensed'`, 
            `'delensed'`, or `'unlensed'` for files containing the CMB TT, TE, 
            EE, and BB spectra along with the lensing (kappa kappa) spectrum;
            or `'clkk_res'` for files containing only the residual lensing power.
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower maximum 
            multipole than the baseline case.
        feedback : bool, default=False
            Used for CMB-HD only. If `True`, the file name returned will be for
            a file holding theory calculated with the HMCode2020 + baryonic
            feedback non-linear model, as opposed to the HMCode2016 CDM-only
            model.
        
        Returns
        -------
        fname : str
            The absolute path and name of the requested file.
    
        Raises
        ------
        ValueError 
            If the requested file does not exist, based on the experiment name
            and spectrum type. Note that the file name may still not exist.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored, or if 
        the requested file does not exist (but an error wasn't raised).

        Note
        ----
        If `cmb_type = 'clkk_res'`, the file will contain a single column
        holding the residual CMB lensing power spectrum. Otherwise, the
        file will have a column for the multipoles of the spectra, the 
        CMB TT, TE, EE, and BB power spectra (in units of uK^2, without
        any multiplicative factors applied), and the lensing power spectrum,
        using the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where
        L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.
        """
        # check the input
        exp = self.check_cmb_exp(exp)
        spectrum_type = spectrum_type.lower()
        valid_spec_types = self.cmb_types + ['clkk_res']
        if spectrum_type not in valid_spec_types:
            raise ValueError(f"Invalid spectrum type. You passed `spectrum_type = '{spectrum_type}'`, but `spectrum_type` must be one of {valid_spec_types}.")
        if ((hd_lmax is not None) or feedback) and (exp in self.cmb_exps[:-1]):
            warnings.warn(f"Ignoring the `hd_lmax` and `feedback` arguments for `exp = '{exp}'`.")
        # get the file name
        if (exp in self.cmb_exps[:-1]) or (not feedback):
            feedback_info = ''
        else:
            feedback_info = '_hmcode2020_feedback'
        lmin = self.lmins[exp]
        if (hd_lmax is None) or (exp in self.cmb_exps[:-1]):
            lmax = self.lmaxs[exp]
            Lmax = self.Lmaxs[exp]
        else:
            hd_lmax = self.check_hd_lmax(hd_lmax)
            lmax = hd_lmax
            Lmax = hd_lmax # for HD, lmax and Lmax will be the same
        if spectrum_type in self.cmb_types:
            spec_info = f'{spectrum_type}_cls'
        else:
            spec_info = spectrum_type
        theo_dir = self.data_path('theory')
        fname = os.path.join(theo_dir, f'{exp}{feedback_info}_lmin{lmin}lmax{lmax}Lmax{Lmax}_{spec_info}.txt')
        if not os.path.exists(fname):
            warnings.warn(f"The requested file {fname} does not exist.")
        return fname


    def cmb_theory_fnames(self, exp, hd_lmax=None, feedback=False):
        """Returns the name of the files containing the lensed, delensed, 
        and unlensed theory CMB and lensing spectra, and the residual lensing 
        power spectrum (used for delensing), for a given CMB experiment.
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower maximum 
            multipole than the baseline case.
        feedback : bool, default=False
            Used for CMB-HD only. If `True`, the file name returned will be for
            a file holding theory calculated with the HMCode2020 + baryonic
            feedback non-linear model, as opposted to the HMCode2016 CDM-only
            model.
    
        Returns
        -------
        fnames : dict of str
            A dictionary with the absolute path and names of the requested 
            files. The keys are `'lensed'`, `'delensed'`, or `'unlensed'` 
            for files containing the CMB TT, TE, EE, and BB spectra along
            with the lensing (kappa kappa) spectrum; and `'clkk_res'` for
            the file containing only the residual lensing power.
    
        Raises
        ------
        ValueError 
            If the experiment name is invalid.

        See also
        --------
        Data.cmb_theory_fname
        """
        exp = self.check_cmb_exp(exp)
        spectrum_types = self.cmb_types + ['clkk_res']
        fnames = {}
        for spectrum_type in spectrum_types:
            fnames[spectrum_type] = self.cmb_theory_fname(exp, spectrum_type, hd_lmax=hd_lmax, feedback=feedback)
        return fnames


    def cmb_noise_fname(self, exp, include_fg=True):
        """Returns the name of the file containing the power spectra of the
        noise on the CMB TT, TE, EE, and BB spectra.
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. You may also pass `exp = 'aso'` for an advanced SO-like 
            experiment. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, the temperature noise in the returned file is the sum of
            the instrumental noise and the residual extragalactic foreground
            power spectrum. If `False`, it will only contain instrumental noise.
            Used only when `exp = 'hd'`.

        Returns
        -------
        fname : str
            The name of the file holding the requested noise spectra.

        Raises
        ------
        ValueError 
            If the `exp` is invalid.

        Warns
        -----
        If the value of `include_fg` was changed from its default, but 
        will be ignored.

        Note
        ----
        The returned file will have a column for the multipoles of the spectra, 
        and columns for the CMB TT, TE, EE, and BB noise spectra (in units 
        of uK^2, without any multiplicative factors applied).
        """
        # check the input
        valid_exps = ['aso'] + self.cmb_exps
        exp = self.check_cmb_exp(exp, valid_exps=valid_exps)
        noise_dir = self.data_path('noise')
        if exp in valid_exps[:-1]:
            fname = os.path.join(noise_dir, f'{exp}_coaddf090f150_cmb_noise_cls_lmax5000.txt')
            if not include_fg:
                warnings.warn(f"Ignoring the `include_fg` argument for `exp = '{exp}'`.")
        else:
            fg_info = 'withfg' if include_fg else 'nofg'
            fname = os.path.join(noise_dir, f'hd_coaddf090f150_cmb_noise_cls_lmax30000_ASObelow1000_{fg_info}.txt')
        if not os.path.exists(fname):
            warnings.warn(f"The requested file {fname} does not exist.")
        return fname


    def cmb_lensing_noise_fname(self, exp, include_fg=True, hd_Lmax=None):
        """Returns the name of the file containing the CMB lensing noise 
        spectrum for the given CMB experiment.

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        include_fg : bool, default=True
            If `True`, return the file name for lensing noise that was 
            calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for lensing
            noise that was calculated by neglecting the effects. 
            Used only when `exp = 'hd'`.
        hd_Lmax : int or None, default=None
            The maximum lensing multipole used in the calculation of the 
            noise, if lower than the baseline value of 20100. The allowed
            values are contained in the list `Data.hd_lmaxs`. Used only
            when `exp = 'hd'`.

        Returns
        -------
        fname : str
            The name of the file holding the requested lensing noise spectrum.

        Raises
        ------
        ValueError 
            If either the `exp` or value of `hd_Lmax` are invalid.

        Warns
        -----
        If the values of `include_fg` or `hd_Lmax` were provided, but will
        be ignored.

        Note
        ----
        The returned file contains two columns: L, N_L^kk, where L is the
        CMB lensing multipole and N_L^kk is the noise on the CMB lensing
        power spectrum, C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, where 
        C_L^phiphi is the CMB lensing potential power spectrum.
        """
        # check the input
        exp = self.check_cmb_exp(exp)
        if ((hd_Lmax is not None) or (not include_fg)) and (exp in self.cmb_exps[:-1]):
            warnings.warn(f"Ignoring the `hd_Lmax` and `include_fg` arguments for `exp = '{exp}'`.")
        # get the file name
        if include_fg or (exp in self.cmb_exps[:-1]):
            extra_info = ''
        else:
            extra_info = f'_nofg'
        lmin = self.lmins[exp]
        if (hd_Lmax is None) or (exp in self.cmb_exps[:-1]):
            lmax = self.lmaxs[exp]
            Lmax = self.Lmaxs[exp]
        else:
            hd_Lmax = self.check_hd_lmax(hd_Lmax)
            lmax = hd_Lmax # for HD, lmax and Lmax will be the same
            Lmax = hd_Lmax
        fname = os.path.join(self.data_path('noise'), f'{exp}{extra_info}_lmin{lmin}lmax{lmax}Lmax{Lmax}_nlkk.txt')
        if not os.path.exists(fname):
            warnings.warn(f"The requested file {fname} does not exist.")
        return fname

    
    def hd_fg_fname(self, frequency='coadd'):
        """Returns the name of the file containing the residual extragalactic 
        foreground power spectra (or a single coadded spectrum) for CMB-HD.

        Parameters
        ----------
        frequency : str or int, default='coadd'
            If `'coadd'`, the file will contain the coadded foreground
            power spectrum for the combination of 90 and 150 GHz. Otherwise,
            pass `90` or `'f090'` for a file containing columns for the
            different foreground components at 90 GHz, or pass `150` or 
            `'f150'` for the corresponding file at 150 GHz.

        Returns
        -------
        fname : str
            The file name (including its absolute path).

        Raises
        ------
        ValueError
            If an invalid `frequency` was passed.
        """
        freq = str(frequency).lower()
        fg_dir = self.data_path('fg')
        if 'coadd' in freq:
            fname = os.path.join(fg_dir, 'cmbhd_coadd_f090f150_total_fg_cls.txt')
        elif '90' in freq:
            fname = os.path.join(fg_dir, 'cmbhd_fg_cls_f090.txt')
        elif '150' in freq:
            fname = os.path.join(fg_dir, 'cmbhd_fg_cls_f150.txt')
        else:
            raise ValueError(f"Invalid `frequency`. You passed `frequency = '{freq}'`; valid choices are `'coadd'`, `'f090'`, or `'f150'`.")
        return fname


    def cmb_covmat_fname(self, exp, cmb_type='delensed', include_fg=True, hd_lmax=None):
        """Returns the name of the file holding the covariance matrix for the
        mock CMB TT, TE, EE, BB and CMB lensing power spectra corresponding to 
        the given experimental configuration and CMB type (lensed or delensed).
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a covariance matrix for
            delensed CMB TT, TE, EE, and BB power spectra, in addition to the
            CMB lensing spectrum. If `cmb_type='lensed'`, the covariance matrix
            is for lensed CMB spectra instead, but otherwise includes the same
            set of power spectra as the delensed case. Note that passing 
            `cmb_type='lensed'` is only an option for `exp='hd'` and 
            `hd_lmax=None`.
        include_fg : bool, default=True
            If `True`, return the file name for CMB-HD covariance matrix that 
            was calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for the covariance 
            matrix that was calculated by neglecting the effects. 
            Used only when `exp = 'hd'`; `include_fg=False` is only possible
            when `cmb_type='lensed'`.
        hd_lmax : int, default=None
            Used to return CMB-HD covariance matrices that were calculated with 
            a lower maximum multipole than the baseline case. Only used when
            `exp='hd'`.

        Returns
        -------
        fname : str
            The name of the file that contains the requested covariance matrix.
        
        Raises
        ------
        ValueError 
            If either the `exp` or value of `hd_lmax` are invalid, or if the
            requested covariance matrix does not exist.

        Warns
        -----
        If the values of `include_fg` or `hd_lmax` were provided, but will
        be ignored.
        """
        # check the input
        exp = self.check_cmb_exp(exp)
        cmb_type = cmb_type.lower()
        if ((hd_lmax is not None) or (not include_fg)) and (exp in self.cmb_exps[:-1]):
            warnings.warn(f"Ignoring the `hd_lmax` and `include_fg` arguments for `exp = '{exp}'`.")
        # get the file name
        if include_fg or (exp in self.cmb_exps[:-1]):
            extra_info = ''
        else:
            extra_info = f'_nofg'
        lmin = self.lmins[exp]
        if (hd_lmax is None) or (exp in self.cmb_exps[:-1]):
            lmax = self.lmaxs[exp]
            lmaxTT = self.lmaxsTT[exp]
            Lmax = self.Lmaxs[exp]
        else:
            hd_lmax = self.check_hd_lmax(hd_lmax)
            lmax = hd_lmax
            lmaxTT = hd_lmax # for HD, lmax and TT lmax will be the same
            Lmax = hd_lmax # for HD, lmax and Lmax will be the same
        if cmb_type == 'lensed':
            if exp != 'hd':
                raise ValueError(f"No covariance matrix for lensed {exp.upper()} data.")
            elif lmax < self.lmaxs['hd']:
                raise ValueError(f"No covariance matrix for lensed HD data with lmax < {self.lmaxs['hd']}.")
        elif cmb_type != 'delensed':
            raise ValueError("Invalid `cmb_type`. You must pass `cmb_type = 'delensed'`; you may also pass `cmb_type = 'lensed'` for CMB-HD.")
        if ((exp == 'hd') and (cmb_type == 'delensed')) and (not include_fg):
            raise ValueError("No covariance matrix for CMB-HD delensed data without foregrounds; you must pass `cmb_type = 'lensed'`.")
        # get the file name
        ell_info = f'lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}'
        fname = os.path.join(self.data_path('covmats'), f'{exp}{extra_info}_fsky0pt6_{ell_info}_binned_{cmb_type}_cov.txt')
        if not os.path.exists(fname):
            warnings.warn(f"The requested file {fname} does not exist.")
        return fname


    def desi_theory_fname(self):
        """Returns the name of the file containing the theoretical BAO 
        measurement r_s/d_V(z) for mock DESI BAO. The first column of
        the file contains the redshift z, and the second contains the
        quantity r_s/d_V evaluated at that redshift.
        """
        return os.path.join(self.data_path('bao'), 'mock_desi_bao_rs_over_DV_fid_data.txt')


    def desi_covmat_fname(self):
        """Returns the name of the covariance matrix for the mock DESI BAO
        measurements r_s/d_V(z)."""
        return os.path.join(self.data_path('bao'), 'mock_desi_bao_rs_over_DV_fid_cov.txt')


    def precomputed_desi_fisher_fname(self, use_H0=False):
        """Returns the name of a file holding a Fisher matrix calculated
        from the mock DESI BAO measurements and covariance matrix. The
        parameters in the Fisher matrix are the six LCDM parameters, the
        effective number of relativistic species, and the sum of the neutrino 
        masses.

        Parameters
        ----------
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the angular 
            scale of the sound horizon at last scattering (multiplied by 100)
            is used instead.

        Returns
        -------
        fname : str
            The requested file name.
        """
        H0_info = '_useH0' if use_H0 else ''
        fname = os.path.join(self.data_path('fisher_matrices'), f'desi_bao{H0_info}_fisher.txt')
        return fname


    def precomputed_cmb_fisher_fname(self, exp, cmb_type='delensed', use_H0=False, with_desi=False, hd_lmax=None, include_fg=True, feedback=False):
        """Returns the name of the file containing a Fisher matrix calculated
        for the given experiment and kind of CMB spectra (lensed or delensed).
        The parameters in the Fisher matrix are the six LCDM parameters, the
        effective number of relativistic species, and the sum of the neutrino
        masses. Note that a Gaussian prior of width 0.007 has already been 
        applied. 

        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str, default='delensed'
            If `cmb_type='delensed'`, the file holds a Fisher matrix calculated 
            from delensed CMB TT, TE, EE, and BB power spectra, in addition to 
            the CMB lensing spectrum. If `cmb_type='lensed'`, the Fisher matrix
            was computed with lensed CMB spectra instead, as well as the CMB
            lensing spectrum. Note that passing `cmb_type='lensed'` is only an
            option for `exp='hd'`, `hd_lmax=None`, and `feedback=False`.
        use_H0: bool, default=False
            If `True`, the Hubble constant is used as one of the six LCDM
            parameters. If `False`, the cosmoMC approximation to the angular 
            scale of the sound horizon at last scattering (multiplied by 100)
            is used instead.
        with_desi : bool, default=False
            If `False`, the Fisher matrix was calculated using only CMB spectra.
            If `True`, the Fisher matrix is the sum of a CMB and a mock DESI BAO
            Fisher matrix.
        include_fg : bool, default=True
            If `True`, return the file name for CMB-HD covariance matrix that 
            was calculated including the effects of residual extragalactic
            foregrounds. If `False`, return the file name for the covariance 
            matrix that was calculated by neglecting the effects. 
            Used only when `exp = 'hd'`; `include_fg=False` is only possible
            when `cmb_type='lensed'`.
        hd_lmax : int, default=None
            Used to return CMB-HD covariance matrices that were calculated with 
            a lower maximum multipole than the baseline case. Only used when
            `exp='hd'`.
        feedback : bool, default=False
            If `True`, the Fisher matrix was calculated with the HMCode2020 
            + baryonic feedback non-linear model, and also contains the 
            baryonic feedback parameter of this model (i.e., a total of 9
            parameters). If `False`, the Fisher matrix was calculated with the 
            HMCode2016 CDM-only model. Used only when `exp='hd'`, for 
            `cmb_type='delensed'` and `hd_lmax=None`.

        Returns
        -------
        fname : str
            The requested file name.
        """
        # TODO: docstring
        exp = self.check_cmb_exp(exp)
        cmb_type = cmb_type.lower()
        if (exp != 'hd') and (((hd_lmax is not None) or (not include_fg)) or feedback):
            warnings.warn(f"Ignoring the `hd_lmax`, `include_fg`, and `feedback` arguments for `exp = '{exp}'`.")
        # we only have pre-computed Fisher matrices from lensed power spectra 
        #  for CMB-HD with lmax, Lmax = 20100
        has_lensed = False 
        if (exp == 'hd') and (not feedback):
            if hd_lmax is None:
                has_lensed = True
            elif int(hd_lmax) == self.lmaxs['hd']:
                has_lensed = True
        if (cmb_type == 'lensed') and (not has_lensed):
            raise ValueError("There are no precomputed Fisher matrices from lensed theory for `exp = '{exp}'`, `feedback = {feedback}`, and `hd_lmax = {hd_lmax}`.")
        # we only have Fisher matrices excluding foregrounds for HD lensed data
        if (exp == 'hd') and ((cmb_type == 'delensed') and (not include_fg)):
            raise ValueError("There are no precomputed Fisher matrices for mock delensed CMB-HD data without foregrounds.")
        # make sure the `cmb_type` is valid in general (e.g., no unlensed)
        if cmb_type not in self.cmb_types[:-1]:
            raise ValueError(f"Invalid `cmb_type`: '{cmb_type}'. The `cmb_type` must be one of: {self.cmb_types[:-1]}.")
        # get the file name
        H0_info = '_useH0' if use_H0 else ''
        desi_info = '_desi_bao' if with_desi else ''
        if (exp == 'hd') and (not include_fg):
            fg_info = '_nofg'
        else:
            fg_info = ''
        if (exp == 'hd') and feedback:
            feedback_info = '_feedback'
        else:
            feedback_info = ''
        lmin = self.lmins[exp]
        if (exp == 'hd') and (hd_lmax is not None):
            hd_lmax = self.check_hd_lmax(hd_lmax)
            lmax = hd_lmax
            lmaxTT = lmax
            Lmax = lmax
        else:
            lmax = self.lmaxs[exp]
            lmaxTT = self.lmaxsTT[exp]
            Lmax = self.Lmaxs[exp]
        fname_root = f'{exp}{fg_info}_fsky0pt6_lmin{lmin}lmax{lmax}lmaxTT{lmaxTT}Lmax{Lmax}_{cmb_type}{desi_info}{feedback_info}{H0_info}'
        fisher_dir = self.data_path(f'fisher_matrices')
        fname = os.path.join(fisher_dir, f'{fname_root}_fisher.txt')
        return fname
        


    # ----- functions that load the data: -----

    def load_cmb_theory_spectra(self, exp, cmb_type, output_lmax=None, 
            hd_lmax=None, feedback=False):
        """Returns a dictionary containing the theory CMB and lensing
        power spectra, and the corresponding multipoles, for a given CMB 
        experiment and CMB type (e.g. delensed).
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        cmb_type : str
            The name of the kind of spectra. Must be either `'lensed'`, 
            `'delensed'`, or `'unlensed'`. 
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower maximum 
            multipole than the baseline case.
        feedback : bool, default=False
            Used for CMB-HD only. If `True`, the power spectra were calculated 
            with the HMCode2020 + baryonic feedback non-linear model. Otherwise,
            they were calculated with the HMCode2016 CDM-only model.
        
        Returns
        -------
        theo : dict of array_like of float
            A dictionary with a key `'ells'` holding the multipoles for the
            power spectra; keys `'tt'`, `'te'`, `'ee'`, and `'bb'` for the
            CMB power spectra for the requested `cmb_type`; and a key`'kk'`
            for the CMB lensing spectrum.
    
        Raises
        ------
        ValueError 
            If the `exp` or `cmb_type` is invalid.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored, or if 
        the requested file does not exist (but an error wasn't raised).

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, 
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.

        See also
        --------
        dataconfig.load_all_cmb_theory_spectra
        dataconfig.cmb_theory_fname
        """
        cmb_type = cmb_type.lower()
        if cmb_type not in self.cmb_types:
            if cmb_type == 'clkk_res':
                raise ValueError("Use the `load_residual_cmb_lensing_spectrum` method of the `Data` class to load the resudial lensing power spectrum.")
            else:
                raise ValueError(f"Invalid `cmb_type`. You passed `cmb_type = '{cmb_type}'`; valid options are {self.cmb_types}.")
        # get the file name and load in the spectra
        fname = self.cmb_theory_fname(exp, cmb_type, hd_lmax=hd_lmax, feedback=feedback)
        theo = utils.load_from_file(fname, self.theo_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            theo_lmax = int(theo['ells'][-1])
            if lmax > theo_lmax:
                warnings.warn(f"You requested theory power spectra out to `output_lmax = {output_lmax}`, but the spectra were only computed out to {theo_lmax}.")
            for key in theo.keys():
                theo[key] = theo[key][:lmax+1]
        return theo
    
    
    def load_all_cmb_theory_spectra(self, exp, output_lmax=None, hd_lmax=None, feedback=False):
        """Returns a nested dictionary containing the lensed, delensed, and 
        unlensed theory CMB and lensing power spectra, and the corresponding 
        multipoles, for a given CMB experiment.
        
        Parameters
        ----------
        exp : str
            The name of a valid CMB experiment. Must be either `'SO'`, `'S4'`, 
            or `'HD`'. The name is case-insensitive.
        hd_lmax : int, default=None
            Used for CMB-HD spectra that were calculated with a lower maximum 
            multipole than the baseline case.
        feedback : bool, default=False
            Used for CMB-HD only. If `True`, the power spectra were calculated 
            with the HMCode2020 + baryonic feedback non-linear model. Otherwise,
            they were calculated with the HMCode2016 CDM-only model.
        
        Returns
        -------
        theo : dict of dict of array_like of float
            A nested dictionary whose first set of keys are `'lensed'`, 
            `'delensed'`, and `'unlensed'`. Each key holds a dictionary with 
            a key `'ells'` holding the multipoles for the power spectra; keys 
            `'tt'`, `'te'`, `'ee'`, and `'bb'` for the CMB power spectra for
            that `cmb_type`; and a key`'kk'` for the CMB lensing spectrum.
    
        Raises
        ------
        ValueError 
            If the `exp` is invalid.

        Warns
        -----
        If the `hd_lmax` and `feedback` arguments will be ignored, or if 
        the requested file does not exist (but an error wasn't raised).

        Note
        ----
        The CMB TT, TE, EE, and BB power spectra are in units of uK^2,
        without any multiplicative factors applied. The CMB lensing power
        spectrum uses the convention C_L^kk = [L(L+1)]^2 * C_L^phiphi / 4, 
        where L is the lensing multipole and C_L^phiphi is the CMB lensing
        potential power spectrum.

        See also
        --------
        dataconfig.load_cmb_theory_spectra
        dataconfig.cmb_theory_fname
        """
        theo = {}
        for cmb_type in self.cmb_types:
            theo[cmb_type] = self.load_cmb_theory_spectra(exp, cmb_type, output_lmax=output_lmax, hd_lmax=hd_lmax, feedback=feedback)
        return theo


    def load_residual_cmb_lensing_spectrum(self, exp, output_Lmax=None, hd_Lmax=None, feedback=False): 
        # TODO: docstring
        fname = self.cmb_theory_fname(exp, 'clkk_res', hd_lmax=hd_Lmax, feedback=feedback)
        clkk_res = np.loadtxt(fname)
        L = np.arange(len(clkk_res))
        if output_Lmax is not None:
            Lmax = int(output_Lmax)
            theo_Lmax = len(clkk_res) - 1
            if Lmax > theo_Lmax:
                warnings.warn(f"You requested the residual lensing power spectrum out to `output_Lmax = {output_Lmax}`, but it was only computed out to {theo_Lmax}.")
                clkk_res = clkk_res[:Lmax+1]
                L = L[:Lmax+1]
        return L, clkk_res


    def load_cmb_noise_spectra(self, exp, include_fg=True, output_lmax=None):
        # TODO: docstring
        fname = self.cmb_noise_fname(exp, include_fg=include_fg)
        noise = utils.load_from_file(fname, self.noise_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            noise_lmax = int(noise['ells'][-1])
            if lmax > noise_lmax:
                warnings.warn(f"You requested noise power spectra out to `output_lmax = {output_lmax}`, but the spectra were only computed out to {noise_lmax}.")
        else: # automatically trim to be consistent with theory spectra
            lmax = self.lmaxs[exp[-2:].lower()]
        for key in noise.keys():
            noise[key] = noise[key][:lmax+1]
        return noise


    def load_cmb_lensing_noise_spectrum(self, exp, include_fg=True, 
            output_Lmax=None, hd_Lmax=None):
        # TODO: docstring
        fname = self.cmb_lensing_noise_fname(exp, include_fg=include_fg, hd_Lmax=hd_Lmax)
        L, nlkk = np.loadtxt(fname, unpack=True)
        if output_Lmax is not None:
            Lmax = int(output_Lmax)
            theo_Lmax = int(L[-1])
            if Lmax > theo_Lmax:
                warnings.warn(f"You requested the lensing noise power spectrum out to `output_Lmax = {output_Lmax}`, but it was only computed out to {theo_Lmax}.")
                nlkk = nlkk[:Lmax+1]
                L = L[:Lmax+1]
        return L, nlkk

   
    def load_hd_fg_spectra(self, frequency, output_lmax=None):
        # TODO: docstring
        freq = str(frequency)
        if ('90' not in freq) or ('150' not in freq):
            if 'coadd' in freq:
                raise ValueError("Use the `load_hd_coadd_fg_spectrum` method of the `Data` class to load the total coadded foreground power spectrum.")
            else:
                raise ValueError(f"Invalid `frequency`: you passed `frequency = '{frequency}'`; valid options (of type `str` or `int`) are `90` and `150`.")
        fname = self.hd_fg_fname(frequency=frequency)
        fgs = utils.load_from_file(fname, self.fg_cols)
        if output_lmax is not None:
            lmax = int(output_lmax)
            fg_lmax = int(fgs['ells'][-1])
            if lmax > fg_lmax:
                warnings.warn(f"You requested foreground power spectra out to `output_lmax = {output_lmax}`, but the spectra were only computed out to {fg_lmax}.")
        else: # automatically trim to be consistent with theory spectra
            lmax = self.lmaxs[exp[-2:].lower()]
        for key in fgs.keys():
            fgs[key] = fgs[key][:lmax+1]
        return fgs


    def load_hd_coadd_fg_spectrum(self, output_lmax=None):
        # TODO: docstring
        fname = self.hd_fg_fname(frequency='coadd')
        ells, coadd_fg_cls = np.loadtxt(fname, unpack=True)
        if output_lmax is not None:
            lmax = int(output_lmax)
            fg_lmax = int(ells[-1])
            if lmax > fg_lmax:
                warnings.warn(f"You requested the coadded foreground power spectrum out to `output_lmax = {output_lmax}`, but it was only computed out to {fg_lmax}.")
            ells = ells[:lmax+1]
            coadd_fg_cls = coadd_fg_cls[:lmax+1]
        return ells, coadd_fg_cls


    def load_cmb_covmat(self, exp, cmb_type='delensed', include_fg=True, hd_lmax=None):
        # TODO: docstring
        fname = self.cmb_covmat_fname(exp, cmb_type=cmb_type, include_fg=include_fg, hd_lmax=hd_lmax)
        covmat = np.loadtxt(fname)
        return covmat


    def load_desi_theory(self):
        # TODO: docstring
        fname = self.desi_theory_fname()
        z, rs_dv = np.loadtxt(fname, unpack=True)
        return z, rs_dv


    def load_desi_covmat(self):
        # TODO: docstring
        fname = self.desi_covmat_fname()
        covmat = np.loadtxt(fname)
        return covmat


    def load_bin_edges(self):
        # TODO: docstring
        bin_edges = np.loadtxt(self.bin_edges_fname)
        return bin_edges
    
    
    def load_precomputed_desi_fisher(self, use_H0=False):
        # TODO: docstring
        fname = self.precomputed_desi_fisher_fname(use_H0=use_H0)
        fisher_matrix, fisher_params = fisher.load_fisher_matrix(fname)
        return fisher_matrix, fisher_params


    def load_precomputed_cmb_fisher(self, exp, cmb_type='delensed', use_H0=False, with_desi=False, hd_lmax=None, include_fg=True, feedback=False):
        # TODO: docstring
        fname = self.precomputed_cmb_fisher_fname(exp, cmb_type=cmb_type, use_H0=use_H0, with_desi=with_desi, hd_lmax=hd_lmax, include_fg=include_fg, feedback=feedback)
        fisher_matrix, fisher_params = fisher.load_fisher_matrix(fname)
        return fisher_matrix, fisher_params





    # ----- convenience functions -----

    def load_cmb_lensing_spectrum(self, exp, output_lmax=None, hd_lmax=None, feedback=False):
        # TODO: docstring 
        theo = self.load_cmb_theory_spectra(exp, 'lensed', output_lmax=output_lmax, hd_lmax=hd_lmax, feedback=feedback)
        return theo['ells'], theo['kk']


    def desi_redshifts(self):
        # TODO: docstring
        z, _ = self.load_desi_theory()
        return z


    def binning_matrix(self, exp):
        # TODO: docstring 
        exp = self.check_cmb_exp(exp)
        bin_edges = self.load_bin_edges()
        bmat = utils.binning_matrix(bin_edges, lmin=self.lmins[exp], lmax=self.lmaxs[exp], start_at_ell=2)
        return bmat


    def lbin(self, exp):
        # TODO: docstring
        exp = self.check_cmb_exp(exp)
        bmat = self.binning_matrix(exp)
        ells = np.arange(2, self.lmaxs[exp] + 1)
        lbin = bmat @ ells
        return lbin


    def fiducial_params(self, param_names=None, feedback=False):
        # TODO: docstring
        fid_params = theory.get_params(param_file=self.fiducial_param_file(feedback=feedback))
        params = {}
        if param_names is not None:
            for param in param_names:
                if param not in fid_params.keys():
                    raise ValueError(f"Invalid parameter name `'{param}'` in `param_names`: there are only fiducial values set for {fid_params.keys()}.")
                else:
                    params[param] = fid_params[param]
        else:
            params = fid_params.copy()
        return params
