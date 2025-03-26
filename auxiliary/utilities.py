"""
File used to store utilities of general application.
"""

# Tudat import
from tudatpy.data import save2txt
from tudatpy.kernel.interface import spice_interface, spice
from tudatpy.kernel.astro import gravitation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.astro import element_conversion

# Packages import
import os
import numpy as np


def save_results(state_history,
                 dependent_variables_history,
                 folder,
                 time_stamp):
    output_folder = "./output/propagator_selection"
    os.makedirs(output_folder, exist_ok=True)

    results_folder = output_folder + "/" + time_stamp
    os.makedirs(results_folder, exist_ok=True)

    save2txt(state_history,
             "state_history.dat",
             results_folder)
    save2txt(dependent_variables_history,
             "dependent_variables_history.dat",
             results_folder)


def get_gravity_field_settings_enceladus_benedikter():
    # mu_enceladus = 7.211292085479989E+9
    enceladus_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Enceladus")
    radius_enceladus = 252240.0
    cosine_coef = np.zeros((10, 10))
    sine_coef = np.zeros((10, 10))

    cosine_coef[0, 0] = 1.0

    cosine_coef[2, 0] = 5.4352E-03 / gravitation.legendre_normalization_factor(2, 0)
    cosine_coef[2, 1] = 9.2E-06 / gravitation.legendre_normalization_factor(2, 1)
    cosine_coef[2, 2] = 1.5498E-03 / gravitation.legendre_normalization_factor(2, 2)

    cosine_coef[3, 0] = -1.15E-04 / gravitation.legendre_normalization_factor(3, 0)

    sine_coef[2, 1] = 3.98E-05 / gravitation.legendre_normalization_factor(2, 1)
    sine_coef[2, 2] = 2.26E-05 / gravitation.legendre_normalization_factor(2, 2)

    enceladus_gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(enceladus_gravitational_parameter, radius_enceladus, cosine_coef, sine_coef,
                                                              "IAU_Enceladus")
    return enceladus_gravity_field_settings


def get_synodic_rotation_model_enceladus(simulation_initial_epoch):

    saturn_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Saturn")
    initial_state_enceladus = spice.get_body_cartesian_state_at_epoch("Enceladus",
                                                                          "Saturn",
                                                                          "J2000",
                                                                          "None",
                                                                          simulation_initial_epoch)
    keplerian_state_enceladus = element_conversion.cartesian_to_keplerian(initial_state_enceladus,
                                                                              saturn_gravitational_parameter)
    rotation_rate_enceladus = np.sqrt(saturn_gravitational_parameter / keplerian_state_enceladus[0] ** 3)

    return rotation_rate_enceladus


def get_gravity_field_settings_saturn_benedikter():
    saturn_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Saturn")
    saturn_reference_radius = 60330000.0  # From Iess et al. (2019)
    saturn_unnormalized_cosine_coeffs = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [16290.573E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.059E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-935.314E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.224E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [86.340E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.108E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-14.624E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.369E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4.672E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.317E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-0.997E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.41E6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    saturn_unnormalized_sine_coeffs = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    saturn_normalized_coeffs = gravitation.normalize_spherical_harmonic_coefficients(
       saturn_unnormalized_cosine_coeffs, saturn_unnormalized_sine_coeffs
    )

    saturn_normalized_cosine_coeffs = saturn_normalized_coeffs[0]
    saturn_normalized_sine_coeffs = saturn_normalized_coeffs[1]

    saturn_associated_reference_frame = "IAU_Saturn"

    saturn_gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
        saturn_gravitational_parameter,
        saturn_reference_radius,
        saturn_normalized_cosine_coeffs,
        saturn_normalized_sine_coeffs,
        saturn_associated_reference_frame
    )

    return saturn_gravity_field_settings