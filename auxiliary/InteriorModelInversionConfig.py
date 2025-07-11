# General imports
import sys
sys.path.append("/Users/mattiacontarini/miniconda3/envs/tudat-bundle-fork/lib/python3.11/site-packages")
import numpy as np

# Central value for gravitational Love number of Enceladus
k2_real = 0.02 # Genova et al. (2024)
k2_imag = 0.01 # Genova et al. (2024)
k2_real_std = ...
k2_imag_std = ...

# Central value for diurnal libration amplitude of Enceladus
diurnal_libration_amplitude = np.deg2rad(-0.091) # Park et al. (2024)
diurnal_libration_amplitude_std = np.rad2deg(2e-6)

# Central value for radial displacement Love number of Enceladus
h2 = ...
h2_std = 5e-4

# Nominal value for the mass of Enceladus
mass = ...

# Nominal value for the MoI of Enceladus
moi = ...

# Define set of observations
observations = [k2_real, k2_imag, h2, diurnal_libration_amplitude]
observations_std = [k2_real_std, k2_imag_std, h2_std, diurnal_libration_amplitude_std]

# Ranges for interior parameters
interior_parameters_range = dict(
    rho_shell = [900.0, 1000.0],
    rho_ocean = [1000.0, 1300.0],
    R_ocean = [191.1, 291.0],
    mu_core = [50.0e9, 70.0e9],
    mu_shell = [2.5e9, 4.5e9],
)

# Nominal values for the mass, MoI, radius of Enceladus
R_Enceladus = 252.1e3 # Porco et al. (2006)
M_Enceladus = 1.08e20 # Flandes et al. (2023)
MoI_Enceladus = 0.335 * M_Enceladus * R_Enceladus**2 # Iess et al. (2014)

## Nominal values for the interior parameters of Enceladus
# Auxiliary base layer (not core)
nominal_interior_model_base_layer = {
    "R0": 5.0,
    "rho0": 5500.0,
    "mu0": 1e9,
    "Ks0": 10.0e9,
    "eta0": 1e20,
    "ocean": 0,
}

# Core
nominal_interior_model_core_layer = {
    "R0": 191.0,
    "rho0": 2422.0,
    "mu0": 1e9,
    "Ks0": 10.0e9,
    "eta0": 1e20,
    "ocean": 0,
}

# Ocean
nominal_interior_model_ocean_layer = {
    "R0": 229.0,
    "rho0": 1000.0,
    "mu0": 3.3e-1,
    "Ks0": 2.2e9,
    "eta0": 1.9e-3,
    "ocean": 1,
}

# Ice shell
nominal_interior_model_shell_layer = {
    "R0": 252.1,
    "rho0": 920.0,
    "mu0": 3.3e9,
    "Ks0": 33.0e9,
    "eta0": 1.0e18,
    "ocean": 0,
}

# Input parameter necessary to compute the observations
Numerics = {
    "Nlayers": 4,
    "method": "variable",
    "Nrbase": 200.0,
    "parallel_sol": 0.0,
    "parallel_gen": 0.0,
    "perturbation_order": 2.0,
}


# Forcing
Forcing = {
    "Td": 33.0*3600,
    "n": 2.0,
    "m": 0.0,
    "F": 1.0,
    "eccen": 0.0047
}
