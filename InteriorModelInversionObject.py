
# Files and variables import
from auxiliary import InteriorModelInversionConfig as InteriorModelInvConfig
from auxiliary.utilities import InterioriModelInversionUtilities as Util

import src.lov3dpythonPackage.lov3d as lov3d


# General imports
import sys
sys.path.append("/Users/mattiacontarini/miniconda3/envs/tudat-bundle-fork/lib/python3.11/site-packages")
import numpy as np
import emcee

class InteriorModelInversion:

    def __init__(self,
                 interior_parameters_range,
                 observations_central_value,
                 observations_std,
                 chains_burn_in,
                 psrf_threshold,
                 ):
        self.interior_parameters_range = interior_parameters_range
        self.observations_std = observations_std
        self.observations_central_value = observations_central_value
        self.chains_burn_in = chains_burn_in
        self.psrf_threshold = psrf_threshold

    @classmethod
    def from_config(cls):
        observations_central_value = InteriorModelInvConfig.observations
        observations_std = InteriorModelInvConfig.observations_std
        interior_parameters_range = InteriorModelInvConfig.interior_parameters_range
        chains_burn_in = InteriorModelInvConfig.chains_burn_in_steps
        psrf_threshold = InteriorModelInvConfig.psrf_threshold
        return cls(interior_parameters_range,
                   observations_central_value,
                   observations_std,
                   chains_burn_in,
                   psrf_threshold)

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
        print(rho_ocean, rho_core)

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
        print("Interior model:", interior_model)
        numerics = InteriorModelInvConfig.Numerics
        forcing = InteriorModelInvConfig.Forcing

        print("x:", x)

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

        print("log_probability:", log_probability, log_prior_probability, exponent)
        return log_probability.real

    def arrange_interior_parameters_range(self):
        interior_parameters_label = list(self.interior_parameters_range.keys())
        interior_parameters_range_array = np.zeros((2, len(interior_parameters_label)))
        for i in range(len(interior_parameters_label)):
            label = interior_parameters_label[i]
            interior_parameters_range_array[0, i] = self.interior_parameters_range[label][0]
            interior_parameters_range_array[1, i] = self.interior_parameters_range[label][1]

        return interior_parameters_range_array

    def run_mcmc(self, nb_walkers, nb_steps, seed, convergence_tolerance):

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

        # Initialise Ensemble Sampler and iterate until convergence
        sampler = emcee.EnsembleSampler(nb_walkers, nb_interior_control_variables, self.log_probability)
        counter = 0
        convergence_check = False
        #while not convergence_check:
        output = sampler.run_mcmc(x0, nb_steps)
        state_out = output[0]
        log_prob_out = output[1]
        autocorrelation_time = sampler.get_autocorr_time(discard=self.chains_burn_in)
        print("autocorrelation_time:", autocorrelation_time)

        print(output)

        return state_out, counter

    #def check_convergence(self, counter):
    #    if counter > self.chains_burn_in:



