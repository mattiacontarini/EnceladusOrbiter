"""
Order of interior parameters:
 0. rho_shell
 1. mu_shell
 2. d_ocean
 3. R_core
 4. mu_core

 Order of observations:
 0. k2_real
 1. k2_imag
 2. h2
 3. libration
"""

# Files import
from InteriorModelInversionObject import InteriorModelInversion

# General imports
import sys
sys.path.append("/Users/mattiacontarini/miniconda3/envs/tudat-bundle-fork/lib/python3.11/site-packages")
import datetime
import os
import numpy as np

def perform_interior_model_inversion(nb_walkers,
                                     nb_steps,
                                     seed,
                                     convergence_tolerance,
                                     output_path,
                                     save_results_flag):
    UDP = InteriorModelInversion.from_config()
    UDP.save_results_flag = save_results_flag
    UDP.run_mcmc(nb_walkers,
                 nb_steps,
                 seed,
                 output_path)


def main():
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Set output path
    output_folder = "./output/interior_parameters_analysis/interior_model_inversion"
    output_path = os.path.join(output_folder, time_stamp)

    get_central_observation_values_flag = False
    if get_central_observation_values_flag:
        UDP = InteriorModelInversion.from_config()
        observations = UDP.compute_observations(x=[920.0, 3.3e9, 23.0, 198.0, 1e9])
        print(observations)

    perform_interior_model_inversion_flag = True
    if perform_interior_model_inversion_flag:
        nb_walkers = 20
        nb_steps = 1000
        seed = 1234
        convergence_tolerance = 5 # %
        perform_interior_model_inversion(nb_walkers,
                                         nb_steps,
                                         seed,
                                         convergence_tolerance,
                                         output_path,
                                         save_results_flag=True)

if __name__ == "__main__":
    main()
