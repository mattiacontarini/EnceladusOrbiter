# Packages import
import numpy as np
import cmath

# Tudat import
from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro import gravitation
from tudatpy import numerical_simulation
from tudatpy.astro import element_conversion
from tudatpy import constants


def get_gravity_field_settings_enceladus_park(max_degree):
    enceladus_gravitational_parameter = 7.210366688598896E+9

    enceladus_reference_radius = 256600.0  # m

    enceladus_unnormalized_cosine_coeffs = np.zeros((max_degree + 1, max_degree + 1))  # Park et al., 2024
    enceladus_unnormalized_cosine_coeffs[0, 0] = 1
    enceladus_unnormalized_cosine_coeffs[2, 0] = -5477.45E-06
    enceladus_unnormalized_cosine_coeffs[2, 1] = 7.86E-06
    enceladus_unnormalized_cosine_coeffs[2, 2] = 1517.9E-06
    enceladus_unnormalized_cosine_coeffs[3, 0] = 177.82E-06

    enceladus_unnormalized_sine_coeffs = np.zeros((max_degree + 1, max_degree + 1))  # Park et al., 2024
    enceladus_unnormalized_sine_coeffs[2, 1] = 7.6E-06
    enceladus_unnormalized_sine_coeffs[2, 2] = -275.31E-06

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


def get_gravity_field_variation_list_enceladus():
    tide_raising_body = "Saturn"
    love_numbers = dict()
    love_numbers[2] = complex(0.02, 0.01)  # From Genova et al. 2024

    gravity_field_variation_list = [
        numerical_simulation.environment_setup.gravity_field_variation.solid_body_tide_degree_variable_complex_k(
            tide_raising_body, love_numbers
        )
    ]
    return gravity_field_variation_list


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


def get_rotation_model_settings_enceladus_park(base_frame,
                                               target_frame):
    JULIAN_CENTURY = 100 * constants.JULIAN_YEAR

    ## Input parameters in [deg] and [deg/JC]from Park et al. (2024)
    # Linear terms
    RA0 = 40.59
    RAdot = -0.0902111773
    DE0 = 83.534180
    DEdot = -0.0071054901
    W0 = 8.325383 - 6.019951804413526
    Wdot = 262.7318870466

    # Sinusoidal terms
    RA_amplitude_sinusoidal_terms = dict(
        S01=0.026616,
        S02=0.000686,
        S03=-0.000472,
        S04=-0.000897,
        S05=0.002970,
        S06=0.001127,
        S07=0.000519,
        S08=0.000228,
        S09=0.036804,
        S10=-0.001107,
        S11=0.073107,
        S12=-0.000167,
        S13=0.0,
        S14=0.0,
        S15=-0.000376,
        S16=0.000248,
        S17=-0.000137,
        S18=0.0)
    DE_amplitude_sinusoidal_terms = dict(
        S01=0.004398,
        S02=-0.000264,
        S03=-0.000185,
        S04=-0.000093,
        S05=-0.000068,
        S06=-0.000236,
        S07=0.0,
        S08=-0.000028,
        S09=0.004141,
        S10=-0.000124,
        S11=0.008229,
        S12=0.000007,
        S13=0.0,
        S14=0.0,
        S15=-0.000039,
        S16=0.000026,
        S17=-0.000016,
        S18=0.0)
    W_amplitude_sinusoidal_terms = dict(
        S01=-0.026447,
        S02=-0.000682,
        S03=0.000469,
        S04=-0.005118,
        S05=0.036955,
        S06=-0.013111,
        S07=0.014206,
        S08=-0.006687,
        S09=-0.036404,
        S10=0.001082,
        S11=-0.072604,
        S12=-0.266358,
        S13=-0.188429,
        S14=-0.004710,
        S15=0.000337,
        S16=-0.000183,
        S17=-0.001724,
        S18=-0.091295
    )

    # Phase (deg) [:, 0] and frequency (deg/JC) [:, 1] of sinusoidal terms
    phase_frequency_sinusoidal_terms = dict(
        S01=[335.844470, 51.7682239],
        S02=[355.351814, 101.6467750],
        S03=[9.369346, 1004.8728024],
        S04=[129.755966, 1223.2050690],
        S05=[219.755966, 1223.2050690],
        S06=[159.835559, 2445.2902118],
        S07=[249.835559, 2445.2902118],
        S08=[117.392885, 3667.0200695],
        S09=[280.169482, 7226.3782354],
        S10=[6.997174, 36506.5422127],
        S11=[196.673251, 15227.2035409],
        S12=[253.848856, 3258.6617087],
        S13=[136.859155, 9266.8742489],
        S14=[144.630256, 12292.3910895],
        S15=[9.821866, 16090.5831593],
        S16=[226.334387, 17383.5986496],
        S17=[93.360491, 18531.0794323],
        S18=[10.9818392, 9583937.8056363]
    )

    # Conversion to SI units
    RA0, DE0, W0 = np.deg2rad(RA0), np.deg2rad(DE0), np.deg2rad(W0)
    RAdot, DEdot, Wdot = (np.deg2rad(RAdot) / JULIAN_CENTURY, np.deg2rad(DEdot) / JULIAN_CENTURY,
                          np.deg2rad(Wdot) / constants.JULIAN_DAY)
    for key in list(RA_amplitude_sinusoidal_terms.keys()):
        RA_amplitude_sinusoidal_terms[key] = np.deg2rad(RA_amplitude_sinusoidal_terms[key])
        DE_amplitude_sinusoidal_terms[key] = np.deg2rad(DE_amplitude_sinusoidal_terms[key])
        W_amplitude_sinusoidal_terms[key] = np.deg2rad(W_amplitude_sinusoidal_terms[key])
        phase_frequency_sinusoidal_terms[key][0] = np.deg2rad(phase_frequency_sinusoidal_terms[key][0])
        phase_frequency_sinusoidal_terms[key][1] = np.deg2rad(phase_frequency_sinusoidal_terms[key][1]) / JULIAN_CENTURY

    # Format sinusoidal coefficients
    pole_periodic_terms = dict()
    for key in list(phase_frequency_sinusoidal_terms.keys()):
        amps = np.array([RA_amplitude_sinusoidal_terms[key], DE_amplitude_sinusoidal_terms[key]])
        phase = phase_frequency_sinusoidal_terms[key][0]
        freq = phase_frequency_sinusoidal_terms[key][1]
        pole_periodic_terms[freq] = (amps, phase)
    meridian_periodic_terms = dict()
    for key in list(phase_frequency_sinusoidal_terms.keys()):
        amp = W_amplitude_sinusoidal_terms[key]
        phase = phase_frequency_sinusoidal_terms[key][0]
        freq = phase_frequency_sinusoidal_terms[key][1]
        meridian_periodic_terms[freq] = (amp, phase)

    enceladus_rotation_model_settings = numerical_simulation.environment_setup.rotation_model.iau_rotation_model(
        base_frame=base_frame,
        target_frame=target_frame,
        nominal_meridian=W0,
        nominal_pole=np.array([RA0, DE0]),
        rotation_rate=Wdot,
        pole_precession=np.array([RAdot, DEdot]),
        meridian_periodic_terms=meridian_periodic_terms,
        pole_periodic_terms=pole_periodic_terms,
    )
    return enceladus_rotation_model_settings


def get_rotation_model_settings_enceladus_park_simplified(base_frame,
                                                          target_frame):
    JULIAN_CENTURY = 100 * constants.JULIAN_YEAR

    ## Input parameters in [deg] and [deg/JC] from Park et al. (2024)
    # Linear terms
    RA0 = 40.59
    RAdot = -0.0902111773
    DE0 = 83.534180
    DEdot = -0.0071054901
    W0 = 1.0 + 1.4075923449758585 + 0.00701704368404985
    Wdot = 262.7318870466

    # Sinusoidal terms
    W_amplitude_sinusoidal_terms = dict(
        S18=-0.091295
    )

    # Phase (deg) [:, 0] and frequency (deg/JC) [:, 1] of sinusoidal terms
    phase_frequency_sinusoidal_terms = dict(
        S18=[10.9818392, 9583937.8056363]
    )

    # Conversion to SI units
    RA0, DE0, W0 = np.deg2rad(RA0), np.deg2rad(DE0), np.deg2rad(W0)
    RAdot, DEdot, Wdot = (np.deg2rad(RAdot) / JULIAN_CENTURY, np.deg2rad(DEdot) / JULIAN_CENTURY,
                          np.deg2rad(Wdot) / constants.JULIAN_DAY)
    for key in list(W_amplitude_sinusoidal_terms.keys()):
        W_amplitude_sinusoidal_terms[key] = np.deg2rad(W_amplitude_sinusoidal_terms[key])
        phase_frequency_sinusoidal_terms[key][0] = np.deg2rad(phase_frequency_sinusoidal_terms[key][0])
        phase_frequency_sinusoidal_terms[key][1] = np.deg2rad(phase_frequency_sinusoidal_terms[key][1]) / JULIAN_CENTURY

    # Format sinusoidal coefficients
    pole_periodic_terms = {
        0.0: (np.array([0.0, 0.0]), 0.0)
    }
    meridian_periodic_terms = dict()
    for key in list(phase_frequency_sinusoidal_terms.keys()):
        amp = W_amplitude_sinusoidal_terms[key]
        phase = phase_frequency_sinusoidal_terms[key][0]
        freq = phase_frequency_sinusoidal_terms[key][1]
        meridian_periodic_terms[freq] = (amp, phase)

    enceladus_rotation_model_settings = numerical_simulation.environment_setup.rotation_model.iau_rotation_model(
        base_frame=base_frame,
        target_frame=target_frame,
        nominal_meridian=W0,
        nominal_pole=np.array([RA0, DE0]),
        rotation_rate=Wdot,
        pole_precession=np.array([RAdot, DEdot]),
        meridian_periodic_terms=meridian_periodic_terms,
        pole_periodic_terms=pole_periodic_terms,
    )
    return enceladus_rotation_model_settings


def atmospheric_density_function_enceladus(h, lon, lat, time):
    data_coordinates = np.array([
        [48.0e3, np.deg2rad(-20), np.deg2rad(360 - 135)],
        [29.0e3, np.deg2rad(-28), np.deg2rad(360 - 97)],
        [103.0e3, np.deg2rad(-82), np.deg2rad(360 - 159)],
        [103.0e3, np.deg2rad(-89), np.deg2rad(360 - 147)],
        [9.0e3, np.deg2rad(-90), np.deg2rad(360 - 128)],
        [49.0e3, np.deg2rad(-87.5), np.deg2rad(360 - 71)]
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
