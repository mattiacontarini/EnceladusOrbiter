"""
Code used to assess the stability of an orbit with a given initial state around Enceladus.
"""

# Files and variables import
from auxiliary import VehicleParameters as VehicleParam
from auxiliary import StableOrbitOptimizationConfig as OptConfig
from auxiliary import utilities as Util
# from StableOrbitProblem import StableOrbitProblem
from StableOrbitProblem2 import StableOrbitProblem

# Tudat import
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.kernel.interface import spice_interface
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro import element_conversion

# Packages import
import numpy as np
import datetime
import pygmo as pg


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
    occulting_bodies = ["Enceladus"]
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
    dependent_variables_history = propagation_results.dependent_variable_history

    return [state_history, dependent_variables_history]


#def main():

seed = OptConfig.seed

#original_cartesian_state =
#original_keplerian_state = element_conversion.cartesian_to_keplerian(original_cartesian_state,
#                                                                     spice_interface.get_body_gravitational_parameter("Enceladus"))
pg.set_global_rng_seed(seed)

    # Select optimization algorithm
algo = pg.algorithm(pg.pso(gen=1, seed=seed))

    # Define problem
UDP = StableOrbitProblem.from_config()
prob = pg.problem(UDP)

    # Define initial population
population_size = 11 * 6
pop = pg.population(prob, size=population_size, seed=seed)

pop = algo.evolve(pop)

print(pop)



"""
    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Define start epoch and simulation duration
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
    synodic_rotation_rate_enceladus = Util.get_synodic_rotation_model_enceladus(simulation_start_epoch)
    initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("ECLIPJ2000",
                                                                                 "IAU_Enceladus",
                                                                                 simulation_start_epoch)
    body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
        "ECLIPJ2000", "IAU_Enceladus", initial_orientation_enceladus,
        simulation_start_epoch, synodic_rotation_rate_enceladus)

    # Set gravity field settings for Enceladus
    body_settings.get("Enceladus").gravity_field_settings = Util.get_gravity_field_settings_enceladus_benedikter()

    # Set gravity field settings for Saturn
    body_settings.get("Saturn").gravity_field_settings = Util.get_gravity_field_settings_saturn_benedikter()
    body_settings.get("Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

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

    acceleration_settings_on_vehicle_neumann = dict(
        Sun=[
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.point_mass_gravity()
        ],
        Saturn=[
            propagation_setup.acceleration.spherical_harmonic_gravity(8, 8)
        ],
        Enceladus=[
            propagation_setup.acceleration.spherical_harmonic_gravity(3, 3)
        ],
    )

    # Create global accelerations dictionary
    acceleration_settings = {"Vehicle": acceleration_settings_on_vehicle}

    initial_state = OptConfig.initial_state

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_distance("Vehicle", "Enceladus")
    ]

    # Create numerical integrator settings
    fixed_step_size = 10
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size,
        propagation_setup.integrator.CoefficientSets.rkf_78,
        order_to_use=propagation_setup.integrator.OrderToIntegrate.higher
    )



    [state_history, dependent_variables_history] = propagate_orbit(simulation_start_epoch,
                                                                 simulation_end_epoch,
                                                                 initial_state,
                                                                 body_settings,
                                                                 bodies_to_propagate,
                                                                 central_bodies,
                                                                 acceleration_settings,
                                                                 integrator_settings,
                                                                 dependent_variables_to_save)

    output_folder = "./output/propagator_selection"
    Util.save_results(state_history, dependent_variables_history, output_folder, time_stamp)

"""

#if __name__ == '__main__':
#    main()