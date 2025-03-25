"""
Code used to assess the stability of an orbit with a given initial state around Enceladus.
"""

# Files and variables import
import BenedikterInitialStates as Benedikter
import VehicleParameters as VehicleParam

# Packages import
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.kernel.interface import spice_interface
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro.element_conversion import keplerian_to_cartesian
from tudatpy.util import result2array
from tudatpy.kernel.astro import gravitation
from tudatpy.astro import element_conversion
import numpy as np


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
    saturn_unnormalized_cosine_coeffs = [
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
    ]
    saturn_unnormalized_sine_coeffs = [
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
    ],
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


def main():

    simulation_start_epoch = 0 * constants.JULIAN_YEAR  # From Benedikter et al. (2022)
    arc_duration = 20.52 * constants.JULIAN_DAY  # From Benedikter et al. (2022)
    simulation_end_epoch = simulation_start_epoch + arc_duration

    # Load spice kernels
    spice.load_standard_kernels()

    # Create settings for celestial bodies
    bodies_to_create = ["Sun",
                        "Saturn",
                        "Enceladus",
                        "Mimas",
                        "Tethys",
                        "Dione",
                        "Rhea",
                        "Titan",
                        "Hyperion"]
    global_frame_origin = "Enceladus"
    global_frame_orientation = "ECLIPJ2000"

    # Retrieve default body settings
    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                global_frame_origin,
                                                                global_frame_orientation)

    # Add Jupiter barycenter
    body_settings.add_empty_settings("Jupiter_barycenter")
    body_settings.get("Jupiter_barycenter").gravity_field_settings = environment_setup.gravity_field.central(
        spice_interface.get_body_gravitational_parameter("5")
    )
    body_settings.get("Jupiter_barycenter").ephemeris_settings = environment_setup.ephemeris.direct_spice(
        body_name_to_use="5"
    )

    # Set rotation model settings for Enceladus
    synodic_rotation_rate_enceladus = get_synodic_rotation_model_enceladus(simulation_start_epoch)
    initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000",
                                                                                 "IAU_Enceladus",
                                                                                 simulation_start_epoch)
    body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
        "J2000", "IAU_Enceladus", initial_orientation_enceladus,
        simulation_start_epoch, synodic_rotation_rate_enceladus)

    # Set gravity field settings for Enceladus
    body_settings.get("Enceladus").gravity_field_settings = get_gravity_field_settings_enceladus_benedikter()

    # Set gravity field settings for Saturn
    body_settings.get("Saturn").gravity_field_settings = get_gravity_field_settings_saturn_benedikter()
    body_settings.get("Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle object and add properties
    bodies.create_empty_body("Vehicle")
    bodies.get("Vehicle").mass = VehicleParam.mass

    # Create aerodynamic coefficient settings
    reference_area = VehicleParam.drag_reference_area
    drag_coefficient = VehicleParam.drag_coefficient
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0.0, 0.0]
    )

    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, "Vehicle", aero_coefficient_settings)

    # Create radiation pressure settings
    reference_area = VehicleParam.radiation_pressure_reference_area
    radiation_pressure_coefficient = VehicleParam.radiation_pressure_coefficient
    occulting_bodies = {"Sun": ["Enceladus"]}
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area, radiation_pressure_coefficient, occulting_bodies
    )

    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_interface(bodies, "Vehicle", radiation_pressure_settings)

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ["Vehicle"]
    central_bodies = ["Enceladus"]

    # Define accelerations acting on vehicle
    acceleration_settings_on_vehicle = dict(
        Sun=[
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Saturn=[
            propagation_setup.acceleration.spherical_harmonic_gravity(12, 12)
        ],
        Enceladus=[
            propagation_setup.acceleration.spherical_harmonic_gravity(3, 3)
        ],
        Jupiter_barycenter=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Mimas=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Tethys=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Dione=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Rhea=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Titan=[
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Hyperion=[
            propagation_setup.acceleration.point_mass_gravity()
        ]
    )

    # Create global accelerations dictionary
    acceleration_settings = {"Vehicle": acceleration_settings_on_vehicle}

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    initial_state = Benedikter.K1_initial_cartesian_state

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position("Vehicle", "Enceladus")
    ]

    # Create numerical integrator settings
    fixed_step_size = 1
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size,
        propagation_setup.integrator.CoefficientSets.rkf_78,
        order_to_use=propagation_setup.integrator.OrderToIntegrate.higher
    )

    # Create propagation settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings,
        output_variables=dependent_variables_to_save,
    )

    # Create simulation object and propagate dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # Retrieve simulation results
    propagation_results = dynamics_simulator.propagation_results

    # Extract numerical solution for states and dependent variables
    state_history = propagation_results.state_history
    dependent_variables_history = propagation_results.dependent_variables_history
    dependent_variables_array = result2array(dependent_variables_history)


def propagate_orbit(simulation_start_epoch,
                    simulation_end_epoch,
                    initial_state,
                    body_settings,
                    bodies_to_propagate,
                    central_bodies,
                    acceleration_settings,
                    integrator_settings,
                    dependent_variables_to_save,
                    ):

    # Create environment
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle object and add properties
    bodies.create_empty_body("Vehicle")
    bodies.get("Vehicle").mass = VehicleParam.mass

    # Create aerodynamic coefficient settings
    reference_area = VehicleParam.drag_reference_area
    drag_coefficient = VehicleParam.drag_coefficient
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, [drag_coefficient, 0.0, 0.0]
    )

    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, "Vehicle", aero_coefficient_settings)

    # Create radiation pressure settings
    reference_area = VehicleParam.radiation_pressure_reference_area
    radiation_pressure_coefficient = VehicleParam.radiation_pressure_coefficient
    occulting_bodies = {"Sun": ["Enceladus"]}
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area, radiation_pressure_coefficient, occulting_bodies
    )

    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_interface(bodies, "Vehicle", radiation_pressure_settings)

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )

    # Create propagation settings
    termination_settings = propagation_setup.propagator.time_termination(simulation_end_epoch)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings,
        output_variables=dependent_variables_to_save,
    )

    # Create simulation object and propagate dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # Retrieve simulation results
    propagation_results = dynamics_simulator.propagation_results

    # Extract numerical solution for states and dependent variables
    state_history = propagation_results.state_history
    dependent_variables_history = propagation_results.dependent_variables_history
    dependent_variables_array = result2array(dependent_variables_history)



if __name__ == '__main__':
    main()