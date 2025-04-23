
# Files and variables import


# Tudat import
from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro import gravitation
from tudatpy import numerical_simulation
from tudatpy.astro import element_conversion

# Packages import
import numpy as np


def get_gravity_field_settings_enceladus_park(max_degree):
    enceladus_gravitational_parameter = 7.210366688598896E+9

    enceladus_reference_radius = 256600.0  # m

    enceladus_unnormalized_cosine_coeffs = np.zeros((max_degree+1, max_degree+1))  # Park et al., 2024
    enceladus_unnormalized_cosine_coeffs[0, 0] = 1
    enceladus_unnormalized_cosine_coeffs[2, 0] = -5477.45E-06
    enceladus_unnormalized_cosine_coeffs[2, 1] = 7.86E-06
    enceladus_unnormalized_cosine_coeffs[2, 2] = 1517.9E-06
    enceladus_unnormalized_cosine_coeffs[3, 0] = 177.82E-06

    enceladus_unnormalized_sine_coeffs = np.zeros((max_degree+1, max_degree+1))  # Park et al., 2024
    enceladus_unnormalized_sine_coeffs[2, 1] = 7.6E-06
    enceladus_unnormalized_sine_coeffs[2, 2] = -275.31E-06

    """
    enceladus_unnormalized_cosine_coeffs = np.array([  # Park et al.
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [-5477.45E-06, 7.86E-06, 1517.9E-06, 0],
        [177.82E-06, 0, 0, 0]
    ])

    enceladus_unnormalized_sine_coeffs = np.array([  # Park et al.
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 7.6E-06, -275.31E-06, 0],  # 2 0 - 2 1 - 2 2
        [0, 0, 0, 0]
    ])
    
    """
    enceladus_normalized_coeffs = gravitation.normalize_spherical_harmonic_coefficients(
        enceladus_unnormalized_cosine_coeffs, enceladus_unnormalized_sine_coeffs
    )

    enceladus_normalized_cosine_coeffs = enceladus_normalized_coeffs[0]
    enceladus_normalized_sine_coeffs = enceladus_normalized_coeffs[1]

    enceladus_associated_reference_frame = "IAU_Enceladus"

    enceladus_gravity_field_settings = numerical_simulation.environment_setup.gravity_field.spherical_harmonic(
        enceladus_gravitational_parameter,
        enceladus_reference_radius,
        enceladus_normalized_cosine_coeffs,
        enceladus_normalized_sine_coeffs,
        enceladus_associated_reference_frame
    )

    return enceladus_gravity_field_settings


def get_synodic_rotation_model_enceladus(simulation_initial_epoch):
    saturn_gravitational_parameter = 3.793120749865224E+16
    initial_state_enceladus = spice.get_body_cartesian_state_at_epoch("Enceladus",
                                                                      "Saturn",
                                                                      "J2000",
                                                                      "None",
                                                                      simulation_initial_epoch)
    keplerian_state_enceladus = element_conversion.cartesian_to_keplerian(initial_state_enceladus,
                                                                          saturn_gravitational_parameter)
    rotation_rate_enceladus = np.sqrt(saturn_gravitational_parameter / keplerian_state_enceladus[0] ** 3)

    return rotation_rate_enceladus


def get_rotation_settings_enceladus_park():

    ## Input parameters in [deg] from Park et al. (2024)
    # Linear terms
    RA0 = 40.59
    RAdot = -0.0902111773
    DE0 = 83.534180
    DEdot = -0.0071054901
    W0 = 8.325383
    Wdot = 262.7318870466

    # Sinusoidal terms
    amplitudes_sinusoidal_terms = [0.026616,
                                   0.000686,
                                   -0.000472,
                                   -0.000897,
                                   0.002970,
                                   0.001127,
                                   0.000519,
                                   0.000228,
                                   0.036804,
                                   -0.001107,
                                   0.073107,
                                   -0.000167,
                                   -0.000376,
                                   0.000248,
                                   -0.000137]


def atmospheric_density_function_enceladus(h, lon, lat, time):
    data_coordinates = np.array([
        [48.0e3, np.deg2rad(-20), np.deg2rad(360-135)],
        [29.0e3, np.deg2rad(-28), np.deg2rad(360-97)],
        [103.0e3, np.deg2rad(-82), np.deg2rad(360-159)],
        [103.0e3, np.deg2rad(-89), np.deg2rad(360-147)],
        [9.0e3, np.deg2rad(-90), np.deg2rad(360-128)],
        [49.0e3, np.deg2rad(-87.5), np.deg2rad(360-71)]
    ])



def get_atmosphere_model_settings_enceladus():

    enceladus_atmosphere_model_settings = numerical_simulation.environment_setup.atmosphere.custom_constant_temperature(

    )

    return enceladus_atmosphere_model_settings


def get_gravity_field_settings_saturn_iess():
    saturn_gravitational_parameter = 3.793120749865224E+16
    saturn_reference_radius = 60330000.0  # From Iess et al. (2019)
    saturn_unnormalized_cosine_coeffs = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1.629061510215236E-02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
        [-9.519974025353707E-08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
        [9.351185734877162E-04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [5.984128286091720E-08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [-8.676367491774778E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
        [-4.808382695890572E-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
        [1.393087926846997E-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
        [-8.921515415583946E-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
        [-5.425691388908470E-06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10

    ])
    saturn_unnormalized_sine_coeffs = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    saturn_normalized_coeffs = gravitation.normalize_spherical_harmonic_coefficients(
        saturn_unnormalized_cosine_coeffs, saturn_unnormalized_sine_coeffs
    )

    saturn_normalized_cosine_coeffs = saturn_normalized_coeffs[0]
    saturn_normalized_sine_coeffs = saturn_normalized_coeffs[1]

    saturn_associated_reference_frame = "IAU_Saturn"

    saturn_gravity_field_settings = numerical_simulation.environment_setup.gravity_field.spherical_harmonic(
        saturn_gravitational_parameter,
        saturn_reference_radius,
        saturn_normalized_cosine_coeffs,
        saturn_normalized_sine_coeffs,
        saturn_associated_reference_frame
    )

    return saturn_gravity_field_settings
