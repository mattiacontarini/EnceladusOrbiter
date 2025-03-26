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


def main():

    seed = OptConfig.seed

    pg.set_global_rng_seed(seed)

    # Select optimization algorithm
    algo = pg.algorithm(pg.pso(gen=1, seed=seed))

    # Define problem
    UDP = StableOrbitProblem.from_config()
    prob = pg.problem(UDP)

    # Define initial population
    population_size = OptConfig.no_individuals_per_decision_variable * 6
    pop = pg.population(prob, size=population_size, seed=seed)

    pop = algo.evolve(pop)


if __name__ == '__main__':
    main()