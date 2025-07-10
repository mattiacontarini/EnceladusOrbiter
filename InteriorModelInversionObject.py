
# Files and variables import
from auxiliary import InteriorModelInversionConfig as InteriorModelInvConfig

import src.lov3dpythonPackage.lov3d as lov3d


# General imports
import sys
sys.path.append("/Users/mattiacontarini/miniconda3/envs/tudat-bundle-fork/lib/python3.11/site-packages")
import numpy as np

class InteriorModelInversion:

    def __init__(self,
                 interior_parameters_range,
                 observations_central_value,
                 observations_std
                 ):
        self.interior_parameters_range = interior_parameters_range
        self.observations_std = observations_std
        self.observations_central_value = observations_central_value

    @classmethod
    def from_config(cls):
        observations_central_value = InteriorModelInvConfig.observations
        observations_std = InteriorModelInvConfig.observations_std
        interior_parameters_range = InteriorModelInvConfig.interior_parameters_range
        return cls(interior_parameters_range,
                   observations_central_value,
                   observations_std)

    def tidal_response(self, Interior_Model, Numerics, Forcing, eng=None):

        quit = False
        if eng is None:
            eng = lov3d.initialize()
            quit = True

        LoveSpectra = eng.compute_Love(Interior_Model, Numerics, Forcing)
        k2 = LoveSpectra["k"]
        h2 = LoveSpectra["h"]

        libration = eng.compute_Libration(Interior_Model, Numerics, Forcing)

        if quit:
            eng.quit()

        return k2, h2, libration

    def compute_observations(self, x):

        # Auxiliary base layer (not core)
        interior_model_base_layer = InteriorModelInvConfig.nominal_interior_model_base_layer

        # Core
        interior_model_core_layer = InteriorModelInvConfig.nominal_interior_model_core_layer
        #interior_model_core_layer["R0"] = ...
        #interior_model_core_layer["rho0"] = ...
        #interior_model_core_layer["mu0"] = x[4]

        # Ocean
        interior_model_ocean_layer = InteriorModelInvConfig.nominal_interior_model_ocean_layer
        #interior_model_ocean_layer["R0"] = x[3]
        #interior_model_ocean_layer["rho0"] = x[2]

        # Ice shell
        interior_model_shell_layer = InteriorModelInvConfig.nominal_interior_model_shell_layer
        #interior_model_shell_layer["rho0"] = x[0]
        #interior_model_shell_layer["mu0"] = x[1]

        interior_model = [interior_model_base_layer,
                          interior_model_core_layer,
                          interior_model_ocean_layer,
                          interior_model_shell_layer]
        numerics = InteriorModelInvConfig.Numerics
        forcing = InteriorModelInvConfig.Forcing

        k2, h2, libration = self.tidal_response(interior_model, numerics, forcing)
        computed_observations = [k2.real, k2.imag, h2.real, libration]

        return computed_observations

"""
    def probability(self, x):
        computed_observations = self.compute_observations(x)
        exponent = 0
        for i in range(len(self.observations_central_value)):
            delta = computed_observations[i] - self.observations_central_value[i]
            exponent += delta ** 2 / self.observations_std[i] ** 2

        probability = np.exp(-0.5 * exponent)
        return probability

    def arrange_interior_parameters_range(self):
        interior_parameters_label = list(self.interior_parameters_range.keys())
        interior_parameters_range_array = np.zeros((2, len(interior_parameters_label)))
        for i in range(len(interior_parameters_label)):
            label = interior_parameters_label[i]
            interior_parameters_range_array[0, i] = self.interior_parameters_range[label][0]
            interior_parameters_range_array[1, i] = self.interior_parameters_range[label][1]

        return interior_parameters_range_array

    def run_mcmc(self, nb_walkers, nb_steps, seed):

        # Set seed
        np.random.seed(seed)

        # Get variability range of interior parameters
        interior_parameters_variability_range = self.arrange_interior_parameters_range()

        # Draw initial samples assuming uniform distribution
        nb_interior_control_variables = len(self.interior_parameters_range.keys())
        x0 = np.zeros((nb_walkers, nb_interior_control_variables))
        for i in range(nb_walkers):
            for j in range(nb_interior_control_variables):
                x0[i, j] = np.random.uniform(interior_parameters_variability_range[0, j], interior_parameters_variability_range[1, j])

        # Initialise Ensemble Sampler
        sampler = emcee.EnsembleSampler(nb_walkers, nb_interior_control_variables, self.probability)
        state_out = sampler.run_mcmc(x0, nb_steps)

        return state_out
"""
