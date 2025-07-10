
# Central value for gravitational Love number of Enceladus
k2_real = ...
k2_imag = ...
k2_real_std = ...
k2_imag_std = ...

# Central value for diurnal libration amplitude of Enceladus
diurnal_libration_amplitude = ...
diurnal_libration_amplitude_std = ...

# Central value for radial displacement Love number of Enceladus
h2 = ...
h2_std = ...

# Nominal value for the mass of Enceladus
mass = ...

# Nominal value for the MoI of Enceladus
moi = ...

# Define set of observations
observations = [k2_real, k2_imag, h2, diurnal_libration_amplitude]
observations_std = [k2_real_std, k2_imag_std, h2_std, diurnal_libration_amplitude_std]

# Ranges for interior parameters
interior_parameters_range = dict(
    rho_shell = ...,
    rho_ocean = ...,
    R_ocean = ...,
    mu_core = ...,
    mu_shell = ...,
)

# Nominal values for the mass, MoI, radius of Enceladus
MoI_Enceladus = ...
R_Enceladus = 252.0
M_Enceladus = ...

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
    "R0": 200.0,
    "rho0": 2422.0,
    "mu0": 1e9,
    "Ks0": 10.0e9,
    "eta0": 1e20,
    "ocean": 0,
}

# Ocean
nominal_interior_model_ocean_layer = {
    "R0": 226.0,
    "rho0": 1000.0,
    "mu0": 3.3e-1,
    "Ks0": 2.2e9,
    "eta0": 1.9e-3,
    "ocean": 1,
}

# Ice shell
nominal_interior_model_shell_layer = {
    "R0": 252.0,
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
