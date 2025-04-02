# Files and variables import
from auxiliary import VehicleParameters as VehicleParam
from auxiliary import StableOrbitOptimizationConfig as OptConfig
from auxiliary import StableOrbitUtilities as OptUtil
from auxiliary import utilities as Util

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro import element_conversion
from tudatpy.kernel.interface import spice_interface, spice
from tudatpy import constants


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
                 termination_altitude,
                 penalty_coefficients,
                 n_gens_termination,
                 max_n_gens,
                 fitness_change_termination_condition
                 ):

        # Simulation parameters
        self.escape_eccentricity = escape_eccentricity
        self.escape_altitude = escape_altitude
        self.decision_variable_range = decision_variable_range
        self.simulation_start_epoch = simulation_start_epoch
        self.simulation_end_epoch = simulation_end_epoch
        self.enceladus_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Enceladus")
        self.termination_altitude = termination_altitude
        self.penalty_coefficients = penalty_coefficients

        # Termination conditions
        self.n_gens_termination = n_gens_termination
        self.max_n_gens = max_n_gens
        self.fitness_change_termination_condition = fitness_change_termination_condition
        self.fitness_change = 1
        self.fitness_change_store = []
        self.fitness_change_from_previous_generation_store = []

    @classmethod
    def from_config(cls):
        escape_eccentricity = OptConfig.escape_eccentricity
        escape_altitude = OptConfig.escape_altitude
        decision_variable_range = OptConfig.decision_variable_range
        simulation_start_epoch = OptConfig.simulation_start_epoch
        simulation_end_epoch = OptConfig.simulation_end_epoch
        termination_altitude = OptConfig.termination_altitude
        penalty_coefficients = OptConfig.penalty_coefficients
        n_gens_termination = OptConfig.n_gens_termination
        max_n_gens = OptConfig.max_n_gens
        fitness_change_termination_condition = OptConfig.fitness_change_termination_condition

        return cls(escape_eccentricity,
                   escape_altitude,
                   decision_variable_range,
                   simulation_start_epoch,
                   simulation_end_epoch,
                   termination_altitude,
                   penalty_coefficients,
                   n_gens_termination,
                   max_n_gens,
                   fitness_change_termination_condition)

    def fitness(self,
                decision_variables,
                ):
        initial_cartesian_state = element_conversion.keplerian_to_cartesian(decision_variables,
                                                                            self.enceladus_gravitational_parameter)
        [state_history, dependent_variable_history] = self.retrieve_history(initial_cartesian_state)
        epochs = list(dependent_variable_history.keys())
        crash_penalty = abs((epochs[-1] - self.simulation_end_epoch) / self.simulation_end_epoch) * \
                        self.penalty_coefficients["crash_time"]

        final_state = state_history[epochs[-1]]
        final_keplerian_state = element_conversion.cartesian_to_keplerian(final_state,
                                                                          self.enceladus_gravitational_parameter)
        eccentricity_variation = abs((final_keplerian_state[1] - decision_variables[1]) / decision_variables[1])
        semimajor_axis_variation = abs((final_keplerian_state[0] - decision_variables[0]) / decision_variables[0])
        escape_penalty = eccentricity_variation * self.penalty_coefficients[
            "escape_eccentricity"] + semimajor_axis_variation * self.penalty_coefficients["escape_semimajor_axis"]

        fitness = crash_penalty + escape_penalty
        return [fitness]

    def get_bounds(self):
        return self.decision_variable_range

    def retrieve_history(self,
                         initial_state,
                         ):

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
        synodic_rotation_rate_enceladus = Util.get_synodic_rotation_model_enceladus(self.simulation_start_epoch)
        initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("ECLIPJ2000",
                                                                                     "IAU_Enceladus",
                                                                                     self.simulation_start_epoch)
        body_settings.get("Enceladus").rotation_model_settings = environment_setup.rotation_model.simple(
            "ECLIPJ2000", "IAU_Enceladus", initial_orientation_enceladus,
            self.simulation_start_epoch, synodic_rotation_rate_enceladus)

        # Set gravity field settings for Enceladus
        body_settings.get("Enceladus").gravity_field_settings = Util.get_gravity_field_settings_enceladus_benedikter()

        # Set gravity field settings for Saturn
        body_settings.get("Saturn").gravity_field_settings = Util.get_gravity_field_settings_saturn_benedikter()
        body_settings.get(
            "Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

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

        dependent_variables_to_save = [
            propagation_setup.dependent_variable.relative_distance("Vehicle", "Enceladus"),
            propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
        ]

        # Create numerical integrator settings
        fixed_step_size = 10
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
            fixed_step_size,
            propagation_setup.integrator.CoefficientSets.rkf_78,
            order_to_use=propagation_setup.integrator.OrderToIntegrate.higher
        )

        [state_history, dependent_variables_history] = self.propagate_orbit(initial_state,
                                                                            body_settings,
                                                                            bodies_to_propagate,
                                                                            central_bodies,
                                                                            acceleration_settings,
                                                                            integrator_settings,
                                                                            dependent_variables_to_save)

        return [state_history, dependent_variables_history]

    def propagate_orbit(
            self,
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
        #environment_setup.add_radiation_pressure_target_model(bodies, "Vehicle", radiation_pressure_settings)

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
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
            terminate_exactly_on_final_condition=False
        )
        termination_settings_list = [time_termination_settings, altitude_termination_settings]
        termination_settings = propagation_setup.propagator.hybrid_termination(
            termination_settings_list, fulfill_single_condition=True
        )

        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state,
            self.simulation_start_epoch,
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


    def fitness_not_converged(self, n_iter, fitness_list, verbose=False):

        # Take the last given generations if available, otherwise take all the available generations
        if n_iter > self.n_gens_termination:
            last_fitness_values = fitness_list[-self.n_gens_termination - 1 :]
            self.fitness_change = Util.compute_change_of_average_fitness(
                last_fitness_values
            )
            fitness_change_from_previous_generation = (
                Util.compute_change_of_average_fitness(fitness_list[-2:])
            )
            checked_generation = n_iter - self.n_gens_termination
        elif 0 < n_iter <= self.n_gens_termination:
            last_fitness_values = fitness_list
            self.fitness_change = Util.compute_change_of_average_fitness(
                last_fitness_values
            )
            fitness_change_from_previous_generation = (
                Util.compute_change_of_average_fitness(fitness_list[-2:])
            )
            checked_generation = 0
        else:
            self.fitness_change = [1, 1]
            fitness_change_from_previous_generation = [1, 1]
            checked_generation = 0

        self.fitness_change_store.append(self.fitness_change)
        self.fitness_change_from_previous_generation_store.append(
            fitness_change_from_previous_generation
        )

        if verbose:
            print(
                "Generation: "
                + str(n_iter)
                + "; fitness change w.r.t. generation "
                + str(checked_generation)
                + ": "
                + str(self.fitness_change)
            )

        convergence_flag = (
            (
                self.fitness_change > self.fitness_change_termination_condition
            )
            and n_iter < self.max_n_gens
        ) or n_iter <= self.n_gens_termination

        return convergence_flag
