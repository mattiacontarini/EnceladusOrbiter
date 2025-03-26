# Files and variables import
from auxiliary import VehicleParameters as VehicleParam
from auxiliary import StableOrbitOptimizationConfig as OptConfig
from auxiliary import StableOrbitUtilities as OptUtil

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro import element_conversion
from tudatpy.kernel.interface import spice_interface

class StableOrbitProblem:
    """
    Class to initialise, simulate, and optimise a stable orbit around Enceladus.
    The class is created specifically for this problem.
    """

    def __init__(self,
                 escape_eccentricity,
                 escape_altitude,
                 decision_variable_range,
                 simulation_start_epoch,
                 simulation_end_epoch,
                 bodies,
                 bodies_to_propagate,
                 central_bodies,
                 acceleration_settings,
                 integrator_settings,
                 dependent_variables_to_save,
                 stop_altitude,
                 penalty_coefficients,
                 ):

        self.escape_eccentricity = escape_eccentricity
        self.escape_altitude = escape_altitude
        self.decision_variable_range = decision_variable_range
        self.simulation_start_epoch = simulation_start_epoch
        self.simulation_end_epoch = simulation_end_epoch
        self.bodies = lambda: bodies
        self.bodies_to_propagate = bodies_to_propagate
        self.central_bodies = central_bodies
        self.acceleration_settings = acceleration_settings
        self.integrator_settings = integrator_settings
        self.dependent_variables_to_save = dependent_variables_to_save
        self.enceladus_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Enceladus")
        self.termination_altitude = stop_altitude
        self.penalty_coefficients = penalty_coefficients


    @classmethod
    def from_config(cls):
        escape_eccentricity = OptConfig.escape_eccentricity
        escape_altitude = OptConfig.escape_altitude
        decision_variable_range = OptConfig.decision_variable_range
        simulation_start_epoch = OptConfig.simulation_start_epoch
        simulation_end_epoch = OptConfig.simulation_end_epoch
        bodies = OptUtil.get_bodies(simulation_start_epoch)
        bodies_to_propagate = OptUtil.get_bodies_to_propagate()
        central_bodies = OptUtil.get_central_bodies()
        acceleration_settings = OptUtil.get_acceleration_settings()
        integrator_settings = OptUtil.get_integrator_settings()
        dependent_variables_to_save = OptUtil.get_dependent_variables_to_save()
        stop_altitude = OptConfig.termination_altitude
        penalty_coefficients = OptConfig.penalty_coefficients

        return cls(escape_eccentricity,
                   escape_altitude,
                   decision_variable_range,
                   simulation_start_epoch,
                   simulation_end_epoch,
                   bodies,
                   bodies_to_propagate,
                   central_bodies,
                   acceleration_settings,
                   integrator_settings,
                   dependent_variables_to_save,
                   stop_altitude,
                   penalty_coefficients)

    def fitness(self,
                decision_variables,
                ):
        initial_cartesian_state = element_conversion.keplerian_to_cartesian(decision_variables,
                                                                            self.enceladus_gravitational_parameter)
        [state_history, dependent_variable_history] = self.propagate_orbit(initial_cartesian_state)
        epochs = dependent_variable_history.keys()
        crash_penalty = abs((epochs[-1] - self.simulation_end_epoch)/self.simulation_end_epoch) * self.penalty_coefficients["crash_time"]

        final_state = state_history[epochs[-1]]
        final_keplerian_state = element_conversion.cartesian_to_keplerian(final_state,
                                                                          self.enceladus_gravitational_parameter)
        eccentricity_variation = abs((final_keplerian_state[1] - decision_variables[1])/decision_variables[1])
        semimajor_axis_variation = abs((final_keplerian_state[0] - decision_variables[0])/decision_variables[0])
        escape_penalty = eccentricity_variation * self.penalty_coefficients["escape_eccentricity"] + semimajor_axis_variation * self.penalty_coefficients["escape_semimajor_axos"]

        fitness = crash_penalty + escape_penalty
        return fitness

    def get_bounds(self):
        return self.decision_variable_range


    def propagate_orbit(self,
                        initial_state,
                        ):

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            self.bodies, self.acceleration_settings, self.bodies_to_propagate, self.central_bodies
        )

        # Create termination settings
        time_termination_settings = propagation_setup.propagator.time_termination(self.simulation_end_epoch)
        altitude_termination_variable = propagation_setup.dependent_variable.altitude(
            "Vehicle", "Enceladus"
        )
        altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
            dependent_variable_settings=altitude_termination_variable,
            limit_value=self.termination_altitude,
            use_as_lower_limit=True,
            terminate_exactl_on_final_condition=True
        )
        termination_settings_list = [time_termination_settings, altitude_termination_settings]
        termination_settings = propagation_setup.propagator.hybrid_termination(
            termination_settings_list, fulfill_single_condition=True
        )

        # Create propagator settings
        propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            acceleration_models,
            self.bodies_to_propagate,
            initial_state,
            self.simulation_start_epoch,
            self.integrator_settings,
            termination_settings,
            output_variables=self.dependent_variables_to_save,
        )

        # Create simulation object and propagate dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            self.bodies, propagator_settings
        )

        # Retrieve simulation results
        propagation_results = dynamics_simulator.propagation_results

        # Extract numerical solution for states and dependent variables
        state_history = propagation_results.state_history
        dependent_variables_history = propagation_results.dependent_variable_history

        return [state_history, dependent_variables_history]

