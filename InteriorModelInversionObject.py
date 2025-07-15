
# Files and variables import
from auxiliary import InteriorModelInversionConfig as InteriorModelInvConfig
from auxiliary.utilities import InterioriModelInversionUtilities as Util

import src.lov3dpythonPackage.lov3d as lov3d


# General imports
import sys
sys.path.append("/Users/mattiacontarini/miniconda3/envs/tudat-bundle-fork/lib/python3.11/site-packages")
import numpy as np
import emcee
import os
import corner
import matplotlib.pyplot as plt

class InteriorModelInversion:

    def __init__(self,
                 interior_parameters_range,
                 observations_central_value,
                 observations_std,
                 chains_burn_in,
                 save_results_flag,
                 ):
        self.interior_parameters_range = interior_parameters_range
        self.observations_std = observations_std
        self.observations_central_value = observations_central_value
        self.chains_burn_in = chains_burn_in
        self.save_results_flag = save_results_flag

    @classmethod
    def from_config(cls):
        observations_central_value = InteriorModelInvConfig.observations
        observations_std = InteriorModelInvConfig.observations_std
        interior_parameters_range = InteriorModelInvConfig.interior_parameters_range
        chains_burn_in = InteriorModelInvConfig.chains_burn_in_steps
        save_results_flag = True
        return cls(interior_parameters_range,
                   observations_central_value,
                   observations_std,
                   chains_burn_in,
                   save_results_flag)

    def tidal_response(self, Interior_Model, Numerics, Forcing, eng=None):

        quit = False
        if eng is None:
            eng = lov3d.initialize()
            quit = True

        LoveSpectra = eng.compute_Love(Interior_Model, Numerics, Forcing)
        k2 = LoveSpectra["k"]
        h2 = LoveSpectra["h"]

        libration = eng.compute_Libration(Interior_Model, Forcing)

        if quit:
            eng.quit()

        return k2, h2, libration

    def compute_observations(self, x):

        # Auxiliary base layer (not core)
        interior_model_base_layer = InteriorModelInvConfig.nominal_interior_model_base_layer

        R_ocean = x[2] + x[3]

        rho_ocean = Util.get_ocean_density(x[3]*1e3, R_ocean*1e3, x[0])
        rho_core = Util.get_core_density(x[3]*1e3, R_ocean*1e3, x[0], rho_ocean)

        # Core
        interior_model_core_layer = InteriorModelInvConfig.nominal_interior_model_core_layer
        interior_model_core_layer["R0"] = x[3]
        interior_model_core_layer["rho0"] = rho_core
        interior_model_core_layer["mu0"] = x[4]

        # Ocean
        interior_model_ocean_layer = InteriorModelInvConfig.nominal_interior_model_ocean_layer
        interior_model_ocean_layer["R0"] = R_ocean
        interior_model_ocean_layer["rho0"] = rho_ocean

        # Ice shell
        interior_model_shell_layer = InteriorModelInvConfig.nominal_interior_model_shell_layer
        interior_model_shell_layer["rho0"] = x[0]
        interior_model_shell_layer["mu0"] = x[1]

        interior_model = [interior_model_base_layer,
                          interior_model_core_layer,
                          interior_model_ocean_layer,
                          interior_model_shell_layer]
        numerics = InteriorModelInvConfig.Numerics
        forcing = InteriorModelInvConfig.Forcing

        k2, h2, libration_dict = self.tidal_response(interior_model, numerics, forcing)
        libration = libration_dict["amplitude_rad"][0][0]
        if str(libration) == "nan":
            libration = 0
        computed_observations = {
            "k2_real": k2.real,
            "k2_imag": k2.imag,
            "h2": h2.real,
            "libration": libration,
        }

        return computed_observations


    def log_probability_prior(self, x):

        interior_parameters_labels = list(self.interior_parameters_range.keys())
        prior = 0
        for i in range(len(x)):
            param = interior_parameters_labels[i]
            if x[i] < self.interior_parameters_range[param][0] or x[i] > self.interior_parameters_range[param][1]:
                prior = -np.inf
                break

        return prior


    def log_probability(self, x):
        computed_observations = self.compute_observations(x)
        exponent = 0
        for label in list(self.observations_central_value.keys()):
            delta = computed_observations[label] - self.observations_central_value[label]
            exponent += delta ** 2 / self.observations_std[label] ** 2

        log_prior_probability = self.log_probability_prior(x)
        log_probability = -0.5 * exponent + log_prior_probability

        return log_probability.real

    def arrange_interior_parameters_range(self):
        interior_parameters_label = list(self.interior_parameters_range.keys())
        interior_parameters_range_array = np.zeros((2, len(interior_parameters_label)))
        for i in range(len(interior_parameters_label)):
            label = interior_parameters_label[i]
            interior_parameters_range_array[0, i] = self.interior_parameters_range[label][0]
            interior_parameters_range_array[1, i] = self.interior_parameters_range[label][1]

        return interior_parameters_range_array

    def run_mcmc(self, nb_walkers, nb_steps, seed, output_path, labels):

        # Set seed
        np.random.seed(seed)

        # Get variability range of interior parameters
        interior_parameters_variability_range = self.arrange_interior_parameters_range()

        # Draw initial samples assuming uniform distribution
        nb_interior_control_variables = len(self.interior_parameters_range.keys())
        x0 = np.zeros((nb_walkers, nb_interior_control_variables))
        for i in range(nb_walkers):
            for j in range(nb_interior_control_variables):

                # Sample core rigidity in log space
                if list(self.interior_parameters_range.keys())[j] == "mu_core":
                    log_sample = np.random.uniform(np.log10(interior_parameters_variability_range[0, j]), np.log10(interior_parameters_variability_range[1, j]))
                    x0[i, j] = 10**log_sample
                else:
                    x0[i, j] = np.random.uniform(interior_parameters_variability_range[0, j], interior_parameters_variability_range[1, j])

        # Initialise Ensemble Sampler
        sampler = emcee.EnsembleSampler(nb_walkers, nb_interior_control_variables, self.log_probability)

        # Run MCMC
        output = sampler.run_mcmc(x0, nb_steps + self.chains_burn_in)

        # Retrieve output
        state_out = output[0]
        log_prob_out = output[1]
        samples = sampler.get_chain(discard=self.chains_burn_in)

        # Retrieve autocorrelation time
        autocorrelation_time = sampler.get_autocorr_time(discard=self.chains_burn_in)

        # Compute standard deviation of control variables
        std_out = []
        for i in range(nb_interior_control_variables):
            data = state_out[:, i]
            std_out.append(np.std(data))

        # Save results and figures of merit to file
        if self.save_results_flag:

            # Save output solution
            os.makedirs(output_path, exist_ok=True)
            solution_filename = os.path.join(output_path, "mcmc_state_out")
            np.savetxt(solution_filename, state_out)

            # Save log probability of output solution
            log_prob_out_filename = os.path.join(output_path, "mcmc_log_probability_out")
            np.savetxt(log_prob_out_filename, log_prob_out)

            # Save samples history
            samples_filename = os.path.join(output_path, "mcmc_samples_out")
            np.savetxt(samples_filename, samples)

            # Save autocorrelation time
            time_filename = os.path.join(output_path, "mcmc_autocorrelation_time")
            np.savetxt(time_filename, autocorrelation_time)

            # Save standard deviation of output solution
            std_out_filename = os.path.join(output_path, "mcmc_std_state_out")
            np.savetxt(std_out_filename, std_out)

            # Generate corner plot
            flat_samples = sampler.get_chain(discard=self.chains_burn_in, thin=15, flat=True)
            fig = corner.corner(
                data=flat_samples, labels=labels, truths=InteriorModelInvConfig.control_variables_truth_values
            )
            plt.savefig(os.path.join(output_path, "corner_plot.pdf"))

