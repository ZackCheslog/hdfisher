import os
from hdfisher import fisher, mpi

# the `exp` is used to load the covariance matrix for the mock data spectra,
#  and determines the multipole ranges for theory power spectra:
exp = 'hd'
# set the `fisher_dir` that holds the output:
fisher_root = f'{exp}_example'
output_dir = os.path.join(os.getcwd(), 'example_output')
fisher_dir = os.path.join(output_dir, fisher_root)


for use_H0 in [False, True]:
    fisherlib = fisher.Fisher(exp, fisher_dir, use_H0=use_H0)
    fisherlib.calculate_fisher_derivs()
    mpi.comm.barrier()
