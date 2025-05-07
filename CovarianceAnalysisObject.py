#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Files and variables import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary import VehicleParameters as VehicleParam
from auxiliary.utilities import utilities as Util
from auxiliary.utilities import plotting_utilities as PlottingUtil
from auxiliary.utilities import environment_setup_utilities as EnvUtil
from auxiliary.utilities import covariance_analysis_utilities as CovUtil

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array
from tudatpy.kernel.interface import spice
from tudatpy.math import interpolators
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import constants

# Packages import
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class CovarianceAnalysis:

    def __init__(self,
                 initial_state_index: int,
                 save_simulation_results_flag: bool,
                 save_covariance_results_flag: bool,
                 simulation_duration: float,
                 arc_duration: float,
                 tracking_arc_duration: float,
                 kaula_constraint_multiplier: float,
                 a_priori_empirical_accelerations: float,
                 a_priori_lander_position: float,
                 lander_to_include: list[str],
                 include_lander_range_observable_flag: bool,
                 ):
        self.initial_state_index = initial_state_index
        self.save_simulation_results_flag = save_simulation_results_flag
        self.save_covariance_results_flag = save_covariance_results_flag
        self.simulation_duration = simulation_duration
        self.arc_duration = arc_duration
        self.tracking_arc_duration = tracking_arc_duration
        self.kaula_constraint_multiplier = kaula_constraint_multiplier
        self.a_priori_empirical_accelerations = a_priori_empirical_accelerations
        self.a_priori_lander_position = a_priori_lander_position
        self.lander_to_include = lander_to_include
        self.include_lander_range_observable_flag = include_lander_range_observable_flag

    @classmethod
    def from_config(cls):
        initial_state_index = CovAnalysisConfig.initial_state_index
        save_simulation_results_flag = False
        save_covariance_results_flag = False
        simulation_duration = CovAnalysisConfig.simulation_duration
        arc_duration = CovAnalysisConfig.arc_duration
        tracking_arc_duration = CovAnalysisConfig.tracking_arc_duration
        kaula_constraint_multiplier = CovAnalysisConfig.kaula_constraint_multiplier
        a_priori_empirical_accelerations = CovAnalysisConfig.a_priori_empirical_accelerations
        a_priori_lander_position = CovAnalysisConfig.a_priori_lander_position
        lander_to_include = CovAnalysisConfig.lander_names
        include_lander_range_observable_flag = True
        return cls(initial_state_index,
                   save_simulation_results_flag,
                   save_covariance_results_flag,
                   simulation_duration,
                   arc_duration,
                   tracking_arc_duration,
                   kaula_constraint_multiplier,
                   a_priori_empirical_accelerations,
                   a_priori_lander_position,
                   lander_to_include,
                   include_lander_range_observable_flag)

    def save_problem_configuration(self,
                                   output_directory: str):
        if len(self.lander_to_include) == 0:
            lander_to_include = "None"
        else:
            lander_to_include = self.lander_to_include[0]
            for lander in self.lander_to_include[1:]:
                lander_to_include = lander_to_include + "; " + lander

        problem_configuration = {
            "initial_state_index": self.initial_state_index,
            "simulation_duration [days]": self.simulation_duration / constants.JULIAN_DAY,
            "arc_duration [days]": self.arc_duration / constants.JULIAN_DAY,
            "tracking_arc_duration [hours]": self.tracking_arc_duration / 3600.0,
            "kaula_constraint_multiplier": self.kaula_constraint_multiplier,
            "a_priori_empirical_accelerations": self.a_priori_empirical_accelerations,
            "a_priori_lander_position": self.a_priori_lander_position,
            "save_simulation_results_flag": self.save_simulation_results_flag,
            "save_covariance_results_flag": self.save_covariance_results_flag,
            "lander_to_include": lander_to_include,
            "include_lander_range_observable_flag": self.include_lander_range_observable_flag,
        }

        save2txt(problem_configuration, "problem_configuration.txt", output_directory)

    def perform_covariance_analysis(self,
                                    output_path: str):

        # Determine final epoch of simulation
        simulation_end_epoch = CovAnalysisConfig.simulation_start_epoch + self.simulation_duration

        ###################################################################################################################
        ### Environment setup #############################################################################################
        ###################################################################################################################

        # Retrieve default body settings
        body_settings = numerical_simulation.environment_setup.get_default_body_settings(
            CovAnalysisConfig.bodies_to_create,
            CovAnalysisConfig.global_frame_origin,
            CovAnalysisConfig.global_frame_orientation)

        # Set rotation model settings for Enceladus
        body_settings.get("Enceladus").rotation_model_settings = EnvUtil.get_rotation_model_settings_enceladus_park(
            base_frame=CovAnalysisConfig.global_frame_orientation,
            target_frame="IAU_Enceladus"
        )

        # Set gravity field settings for Enceladus
        body_settings.get("Enceladus").gravity_field_settings = EnvUtil.get_gravity_field_settings_enceladus_park(
            CovAnalysisConfig.maximum_degree_gravity_enceladus)
        body_settings.get(
            "Enceladus").gravity_field_settings.scaled_mean_moment_of_inertia = CovAnalysisConfig.Enceladus_scaled_mean_moment_of_inertia

        # Set gravity field settings for Saturn
        body_settings.get("Saturn").gravity_field_settings = EnvUtil.get_gravity_field_settings_saturn_iess()
        body_settings.get(
            "Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = CovAnalysisConfig.Saturn_scaled_mean_moment_of_inertia

        # Set atmosphere settings for Enceladus
        # body_settings.get("Enceladus").atmosphere_settings = Util.get_atmosphere_model_settings_enceladus()

        # Create vehicle object
        body_settings.add_empty_settings("Vehicle")
        body_settings.get("Vehicle").constant_mass = VehicleParam.mass

        # Create aerodynamic coefficient interface settings
        aero_coefficient_settings = numerical_simulation.environment_setup.aerodynamic_coefficients.constant(
            VehicleParam.drag_reference_area, [VehicleParam.drag_coefficient, 0.0, 0.0]
        )

        # Add the aerodynamic interface to the environment
        body_settings.get("Vehicle").aerodynamic_coefficient_settings = aero_coefficient_settings

        # Create radiation pressure settings
        radiation_pressure_settings = numerical_simulation.environment_setup.radiation_pressure.cannonball_radiation_target(
            VehicleParam.radiation_pressure_reference_area, VehicleParam.radiation_pressure_coefficient,
            CovAnalysisConfig.occulting_bodies
        )

        # Add the radiation pressure interface to the environment
        body_settings.get("Vehicle").radiation_pressure_target_settings = radiation_pressure_settings

        # Create empty multi-arc ephemeris for the vehicle
        empty_ephemeris_dict = dict()
        vehicle_ephemeris = numerical_simulation.environment_setup.ephemeris.tabulated(
            empty_ephemeris_dict,
            CovAnalysisConfig.global_frame_origin,
            CovAnalysisConfig.global_frame_orientation
        )
        vehicle_ephemeris.make_multi_arc_ephemeris = True
        body_settings.get("Vehicle").ephemeris_settings = vehicle_ephemeris

        # Create system of bodies
        bodies = numerical_simulation.environment_setup.create_system_of_bodies(body_settings)

        ###################################################################################################################
        ### Propagation setup #############################################################################################
        ###################################################################################################################

        # Define bodies that are propagated
        bodies_to_propagate = ["Vehicle"]

        # Define central bodies of propagation
        central_bodies = ["Enceladus"]

        # Create global accelerations dictionary
        acceleration_settings = {"Vehicle": CovAnalysisConfig.acceleration_settings_on_vehicle}

        # Create acceleration models
        acceleration_models = numerical_simulation.propagation_setup.create_acceleration_models(
            bodies,
            acceleration_settings,
            bodies_to_propagate,
            central_bodies
        )

        # Retrieve the nominal base orbit
        if self.initial_state_index == 1:
            nominal_state_history_array = np.loadtxt(f"nominal_orbits/simulation_duration_{self.simulation_duration}/nominal_state_history_1.dat")
            nominal_state_history = Util.array2dict(nominal_state_history_array)
        elif self.initial_state_index == 2:
            nominal_state_history_array = np.loadtxt(f"nominal_orbits/simulation_duration_{self.simulation_duration}/nominal_state_history_2.dat")
            nominal_state_history = Util.array2dict(nominal_state_history_array)
        elif self.initial_state_index == 3:
            nominal_state_history_array = np.loadtxt(f"nominal_orbits/simulation_duration_{self.simulation_duration}/nominal_state_history_3.dat")
            nominal_state_history = Util.array2dict(nominal_state_history_array)
        else:
            raise ValueError("Initial state index not valid")

        # Create numerical integrator settings
        integrator_settings = CovAnalysisConfig.integrator_settings

        arc_start_times = []
        arc_end_times = []
        arc_start = CovAnalysisConfig.simulation_start_epoch
        while arc_start + self.arc_duration <= simulation_end_epoch:
            arc_start_times.append(arc_start)
            arc_end_times.append(arc_start + self.arc_duration)
            arc_start += self.arc_duration

        # Extract total number of propagationa arcs during science phase
        nb_arcs = len(arc_start_times)
        print(f'Total number of arcs for the science phase: {nb_arcs}')

        # Define arc-wise initial states for the vehicle wrt Enceladus
        initial_states = []
        for i in range(nb_arcs):
            if i == 0:
                initial_state = nominal_state_history[CovAnalysisConfig.simulation_start_epoch]
                initial_states.append(initial_state)
            else:
                lagrange_interpolation_settings = interpolators.lagrange_interpolation(
                    number_of_points=CovAnalysisConfig.number_of_points
                )
                interpolator = interpolators.create_one_dimensional_vector_interpolator(nominal_state_history,
                                                                                        lagrange_interpolation_settings)
                initial_state = interpolator.interpolate(arc_start_times[i])
                initial_states.append(initial_state)

        # Define arc-wise propagator settings
        propagator_settings_list = []
        for i in range(nb_arcs):
            propagator_settings_list.append(
                numerical_simulation.propagation_setup.propagator.translational(
                    central_bodies,
                    acceleration_models,
                    bodies_to_propagate,
                    initial_states[i],
                    arc_start_times[i],
                    integrator_settings,
                    numerical_simulation.propagation_setup.propagator.time_termination(arc_end_times[i]),
                    numerical_simulation.propagation_setup.propagator.cowell,
                    CovAnalysisConfig.dependent_variables_to_save
                )
            )

        # Concatenate all arc-wise propagator settings into multi-arc propagator settings
        propagator_settings = numerical_simulation.propagation_setup.propagator.multi_arc(
            propagator_settings_list,
            False,
            numerical_simulation.propagation_setup.propagator.multi_arc_processing_settings()
        )

        # Propagate dynamics and retrieve simulation results
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
        simulation_results = dynamics_simulator.propagation_results.single_arc_results

        ###################################################################################################################
        ### Observations setup ############################################################################################
        ###################################################################################################################

        # Create ground station settings
        ground_station_names = CovAnalysisConfig.ground_station_names
        ground_station_coordinates = CovAnalysisConfig.ground_station_coordinates
        ground_station_coordinates_type = CovAnalysisConfig.ground_station_coordinates_type
        for ground_station_name in ground_station_names:
            ground_station_settings = numerical_simulation.environment_setup.ground_station.basic_station(
                ground_station_name,
                ground_station_coordinates[ground_station_name],
                ground_station_coordinates_type[ground_station_name]
            )
            numerical_simulation.environment_setup.add_ground_station(bodies.get_body("Earth"), ground_station_settings)

        # Create landers settings
        lander_names = CovAnalysisConfig.lander_names
        for lander_name in lander_names:
            lander_settings = numerical_simulation.environment_setup.ground_station.basic_station(
                lander_name,
                CovAnalysisConfig.lander_coordinates[lander_name],
                CovAnalysisConfig.lander_coordinates_type[lander_name]
            )
            numerical_simulation.environment_setup.add_ground_station(bodies.get_body("Enceladus"), lander_settings)

        # Define link ends for two-way Doppler and range observables, for each ground station and lander
        link_ends = []
        for station in ground_station_names:
            link_ends_per_station = dict()
            link_ends_per_station[observation.transmitter] = observation.body_reference_point_link_end_id(
                "Earth", station
            )
            link_ends_per_station[observation.receiver] = observation.body_reference_point_link_end_id(
                "Earth", station
            )
            link_ends_per_station[observation.reflector1] = observation.body_origin_link_end_id("Vehicle")
            link_ends.append(link_ends_per_station)
        for lander in self.lander_to_include:
            link_ends_per_station = dict()
            link_ends_per_station[observation.transmitter] = observation.body_reference_point_link_end_id(
                "Enceladus", lander
            )
            link_ends_per_station[observation.receiver] = observation.body_reference_point_link_end_id(
                "Enceladus", lander
            )
            link_ends_per_station[observation.reflector1] = observation.body_origin_link_end_id("Vehicle")
            link_ends.append(link_ends_per_station)

        # Define tracking arcs
        tracking_arcs_start = []
        tracking_arcs_end = []
        for arc_start in arc_start_times:
            tracking_arc_start = arc_start + CovAnalysisConfig.tracking_delay_after_stat_of_propagation
            tracking_arcs_start.append(tracking_arc_start)
            tracking_arcs_end.append(tracking_arc_start + self.tracking_arc_duration)

        # Create observation settings for each link ends and observable
        # Define light-time calculations settings
        light_time_correction_settings = observation.first_order_relativistic_light_time_correction(["Sun"])
        # Define range biases settings
        biases = []
        for i in range(nb_arcs):
            biases.append(np.array([CovAnalysisConfig.range_bias]))
        range_bias_settings = observation.arcwise_absolute_bias(tracking_arcs_start, biases, observation.receiver)

        # Define observation settings list
        observation_settings_list = []
        for i in range(len(ground_station_names)):
            link_end = link_ends[i]
            link_definition = observation.LinkDefinition(link_end)
            range_observation = observation.two_way_range(link_definition,
                                                          [light_time_correction_settings],
                                                          range_bias_settings)
            doppler_observation = observation.two_way_doppler_averaged(link_definition,
                                                                       [light_time_correction_settings])
            observation_settings_list.append(doppler_observation)
            observation_settings_list.append(range_observation)
        for i in range(len(ground_station_names), len(ground_station_names) + len(self.lander_to_include)):
            link_end = link_ends[i]
            link_definition = observation.LinkDefinition(link_end)
            doppler_observation = observation.two_way_doppler_averaged(
                link_definition,
                [light_time_correction_settings]
            )
            observation_settings_list.append(doppler_observation)

            if self.include_lander_range_observable_flag:
                range_observation = observation.two_way_range(
                    link_definition,
                    [light_time_correction_settings],
                    range_bias_settings
                )
                observation_settings_list.append(range_observation)

        observation_times_doppler = []
        observation_times_range = []

        for i in range(nb_arcs):

            # Doppler observables
            t = tracking_arcs_start[i]
            while t <= tracking_arcs_end[i]:
                observation_times_doppler.append(t)
                t += CovAnalysisConfig.doppler_cadence

            # Range observables
            t = tracking_arcs_start[i]
            while t <= tracking_arcs_end[i]:
                observation_times_range.append(t)
                t += CovAnalysisConfig.range_cadence

        observation_times_per_type = dict()
        observation_times_per_type[observation.n_way_averaged_doppler_type] = observation_times_doppler
        observation_times_per_type[observation.n_way_range_type] = observation_times_range

        # Define observation settings for both observables, and all link ends (i.e., all ground stations)
        observation_simulation_settings = []
        for link_end in link_ends:
            # Doppler observables
            observation_simulation_settings.append(observation.tabulated_simulation_settings(
                observation.n_way_averaged_doppler_type,
                observation.LinkDefinition(link_end),
                observation_times_per_type[observation.n_way_averaged_doppler_type]
            ))

            # Range observables
            observation_simulation_settings.append(observation.tabulated_simulation_settings(
                observation.n_way_range_type,
                observation.LinkDefinition(link_end),
                observation_times_per_type[observation.n_way_range_type]
            ))

        # Create viability settings which define when an observation is feasible
        viability_settings = []

        # For all tracking stations, check if elevation is sufficient
        for station in ground_station_names:
            viability_settings.append(observation.elevation_angle_viability(
                ["Earth", station],
                CovAnalysisConfig.minimum_elevation_angle_visibility)
            )
        for lander in self.lander_to_include:
            viability_settings.append(observation.elevation_angle_viability(
                ["Enceladus", lander],
                CovAnalysisConfig.minimum_elevation_angle_visibility
            ))

        # Check whether Enceladus or Saturn are occulting the signal
        viability_settings.append(observation.body_occultation_viability(["Vehicle", ""], "Enceladus"))
        viability_settings.append(observation.body_occultation_viability(["Vehicle", ""], "Saturn"))

        # Check whether SEP angle is sufficiently large
        viability_settings.append(observation.body_avoidance_viability(["Vehicle", ""],
                                                                       "Sun",
                                                                       CovAnalysisConfig.minimum_sep_angle))

        # Apply viability checks to all simulated observations
        observation.add_viability_check_to_all(
            observation_simulation_settings,
            viability_settings
        )

        ###################################################################################################################
        ### Estimation setup ##############################################################################################
        ###################################################################################################################

        # Define parameters to estimate
        # Add arc-wise initial states of the vehicle wrt Enceladus
        parameter_settings = numerical_simulation.estimation_setup.parameter.initial_states(propagator_settings,
                                                                                            bodies,
                                                                                            arc_start_times)
        # Add gravitational parameter of Enceladus
        parameter_settings.append(numerical_simulation.estimation_setup.parameter.gravitational_parameter("Enceladus"))
        # Add spherical harmonic coefficients of Enceladus
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.spherical_harmonics_c_coefficients("Enceladus",
                                                                                               minimum_degree=CovAnalysisConfig.minimum_degree_c_enceladus,
                                                                                               minimum_order=CovAnalysisConfig.minimum_order_c_enceladus,
                                                                                               maximum_degree=CovAnalysisConfig.maximum_degree_gravity_enceladus,
                                                                                               maximum_order=CovAnalysisConfig.maximum_degree_gravity_enceladus)
        )
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.spherical_harmonics_s_coefficients("Enceladus",
                                                                                               minimum_degree=CovAnalysisConfig.minimum_degree_s_enceladus,
                                                                                               minimum_order=CovAnalysisConfig.minimum_order_s_enceladus,
                                                                                               maximum_degree=CovAnalysisConfig.maximum_degree_gravity_enceladus,
                                                                                               maximum_order=CovAnalysisConfig.maximum_degree_gravity_enceladus)
        )
        # Add empirical accelerations
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.arcwise_empirical_accelerations("Vehicle",
                                                                                            "Enceladus",
                                                                                            CovAnalysisConfig.empirical_acceleration_components_to_estimate,
                                                                                            arc_start_times)
        )
        # Add landers position
        for lander in self.lander_to_include:
            parameter_settings.append(
                numerical_simulation.estimation_setup.parameter.ground_station_position("Enceladus",
                                                                                        lander)
            )

        # Define consider parameters
        # Add arc-wise lander range biases as consider parameters
        consider_parameter_settings = []
        for link_end in link_ends:
            consider_parameter_settings.append(
                numerical_simulation.estimation_setup.parameter.arcwise_absolute_observation_bias(
                    observation.LinkDefinition(link_end), observation.n_way_range_type, tracking_arcs_start, observation.receiver
                )
            )


        # Create parameters to estimate object
        parameters_to_estimate = numerical_simulation.estimation_setup.create_parameter_set(parameter_settings,
                                                                                            bodies,
                                                                                            propagator_settings,
                                                                                            consider_parameter_settings)
        numerical_simulation.estimation_setup.print_parameter_names(parameters_to_estimate)
        nb_parameters = len(parameters_to_estimate.parameter_vector)
        print(f"Total number of parameters to estimate: {nb_parameters}")

        # Create the estimator
        estimator = numerical_simulation.Estimator(bodies,
                                                   parameters_to_estimate,
                                                   observation_settings_list,
                                                   propagator_settings)

        # Simulate required observations
        simulated_observations = numerical_simulation.estimation.simulate_observations(
            observation_simulation_settings,
            estimator.observation_simulators,
            bodies
        )

        ###################################################################################################################
        ### Covariance analysis
        ###################################################################################################################

        # Define a priori covariance matrix
        inv_apriori = np.zeros((nb_parameters, nb_parameters))

        # Set a priori constraints for vehicle states
        indices_states = parameters_to_estimate.indices_for_parameter_type((
            numerical_simulation.estimation_setup.parameter.arc_wise_initial_body_state_type, ("Vehicle", "")))[0]
        for i in range(indices_states[1] // 6):
            for j in range(3):
                inv_apriori[indices_states[0] + i * 6 + j, indices_states[
                    0] + i * 6 + j] = CovAnalysisConfig.a_priori_position ** -2
                inv_apriori[indices_states[0] + i * 6 + j + 3, indices_states[
                    0] + i * 6 + j + 3] = CovAnalysisConfig.a_priori_velocity ** -2

        # Set a priori constraint for Enceladus' gravitational parameter
        indices_mu = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.gravitational_parameter_type, ("Enceladus", "")))[0]
        for i in range(indices_mu[1]):
            inv_apriori[
                indices_mu[0] + i, indices_mu[
                    0] + i] = CovAnalysisConfig.a_priori_gravitational_parameter_enceladus ** -2

        # Set a priori constraint for Enceladus' gravity field coefficients
        indices_cosine_coef = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.spherical_harmonics_cosine_coefficient_block_type,
             ("Enceladus", "")))[0]
        indices_sine_coef = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.spherical_harmonics_sine_coefficient_block_type,
             ("Enceladus", "")))[0]
        # Apply Kaula's constraint to Enceladus' gravity field
        inv_apriori = CovUtil.apply_kaula_constraint_a_priori(CovAnalysisConfig.kaula_constraint_multiplier,
                                                              CovAnalysisConfig.maximum_degree_gravity_enceladus,
                                                              indices_cosine_coef,
                                                              indices_sine_coef,
                                                              inv_apriori)
        # Overwrite Kaula's rule with existing uncertainties
        inv_apriori[indices_cosine_coef[0], indices_cosine_coef[0]] = CovAnalysisConfig.a_priori_c20 ** -2
        inv_apriori[indices_cosine_coef[0] + 1, indices_cosine_coef[0] + 1] = CovAnalysisConfig.a_priori_c21 ** -2
        inv_apriori[indices_cosine_coef[0] + 2, indices_cosine_coef[0] + 2] = CovAnalysisConfig.a_priori_c22 ** -2
        inv_apriori[indices_cosine_coef[0] + 3, indices_cosine_coef[0] + 3] = CovAnalysisConfig.a_priori_c30 ** -2
        inv_apriori[indices_sine_coef[0], indices_sine_coef[0]] = CovAnalysisConfig.a_priori_s21 ** -2
        inv_apriori[indices_sine_coef[0] + 1, indices_sine_coef[0] + 1] = CovAnalysisConfig.a_priori_s22 ** -2

        # Set a priori constraint for empirical accelerations
        indices_empirical_acceleration_components = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.arc_wise_empirical_acceleration_coefficients_type,
             ("Vehicle", "Enceladus")))[0]
        for i in range(indices_empirical_acceleration_components[1]):
            inv_apriori[indices_empirical_acceleration_components[0] + i, indices_empirical_acceleration_components[
                0] + i] = self.a_priori_empirical_accelerations ** -2

        # Set a priori constraint for landers' position
        for lander_name in self.lander_to_include:
            indices_lander_position = parameters_to_estimate.indices_for_parameter_type(
                (numerical_simulation.estimation_setup.parameter.ground_station_position_type,
                 ("Enceladus", lander_name)))[0]
            for i in range(indices_lander_position[1]):
                inv_apriori[indices_lander_position[0] + i, indices_lander_position[
                    0] + i] = self.a_priori_lander_position ** -2

        # Retrieve full vector of a priori constraints
        apriori_constraints = np.reciprocal(np.sqrt(np.diagonal(inv_apriori)))

        # Define consider parameters covariance
        nb_consider_parameters = nb_arcs * (len(ground_station_names) + len(self.lander_to_include))
        consider_parameter_covariance = np.zeros((nb_consider_parameters, nb_consider_parameters))
        # Set consider covariance for range biases
        for ground_station_name in ground_station_names:
            indices_range_bias = (0, nb_arcs*len(ground_station_names))
            for i in range(indices_range_bias[1]):
                consider_parameter_covariance[indices_range_bias[0] + i, indices_range_bias[0] + i] = (
                        CovAnalysisConfig.a_priori_range_bias_Earth_ground_station ** -2)

        for lander_name in self.lander_to_include:
            indices_range_bias = (nb_arcs*len(ground_station_names), nb_arcs * (len(ground_station_names) + len(self.lander_to_include)) )
            for i in range(indices_range_bias[1]):
                consider_parameter_covariance[indices_range_bias[0] + i, indices_range_bias[0] + i] = (
                        CovAnalysisConfig.a_priori_range_bias_lander ** -2)


        # Create input object for covariance analysis
        covariance_input = numerical_simulation.estimation.CovarianceAnalysisInput(simulated_observations,
                                                                                   inv_apriori,
                                                                                   consider_parameter_covariance)
        covariance_input.define_covariance_settings(reintegrate_variational_equations=False, save_design_matrix=True)

        # Apply weights to simulated observations
        Earth_gs_doppler_weight = CovAnalysisConfig.doppler_noise_Earth_ground_station ** -2
        Earth_gs_range_weight = CovAnalysisConfig.range_noise_Earth_ground_station ** -2
        for ground_station_name in ground_station_names:
            # Doppler observations
            Earth_gs_doppler_parser_list = []
            Earth_gs_doppler_parser_list.append(
                numerical_simulation.estimation.observation_parser(
                    numerical_simulation.estimation_setup.observation.n_way_averaged_doppler_type
                )
            )
            Earth_gs_doppler_parser_list.append(
                numerical_simulation.estimation.observation_parser(
                    ("Earth", ground_station_name)
                )
            )
            Earth_gs_doppler_parser = numerical_simulation.estimation.observation_parser(
                Earth_gs_doppler_parser_list,
                combine_conditions = True
            )
            simulated_observations.set_constant_weight(
                Earth_gs_doppler_weight,
                Earth_gs_doppler_parser
            )
            # Range observations
            Earth_gs_range_parser_list = []
            Earth_gs_range_parser_list.append(
                numerical_simulation.estimation.observation_parser(
                    numerical_simulation.estimation_setup.observation.n_way_range_type
                )
            )
            Earth_gs_range_parser_list.append(
                numerical_simulation.estimation.observation_parser(
                    ("Earth", ground_station_name)
                )
            )
            Earth_gs_range_parser = numerical_simulation.estimation.observation_parser(
                Earth_gs_range_parser_list,
                combine_conditions = True
            )
            simulated_observations.set_constant_weight(
                Earth_gs_range_weight,
                Earth_gs_range_parser
            )
        Enceladus_lander_doppler_weight = CovAnalysisConfig.doppler_noise_lander ** -2
        Enceladus_lander_range_weight = CovAnalysisConfig.range_noise_lander ** -2
        for lander_name in self.lander_to_include:
            Enceladus_lander_doppler_parser_list = []
            Enceladus_lander_doppler_parser_list.append(
                numerical_simulation.estimation.observation_parser(
                    numerical_simulation.estimation_setup.observation.n_way_averaged_doppler_type
                )
            )
            Enceladus_lander_doppler_parser_list.append(
                numerical_simulation.estimation.observation_parser(
                    ("Enceladus", lander_name)
                )
            )
            Enceladus_lander_doppler_parser = numerical_simulation.estimation.observation_parser(
                Enceladus_lander_doppler_parser_list,
                combine_conditions=True
            )
            simulated_observations.set_constant_weight(
                Enceladus_lander_doppler_weight,
                Enceladus_lander_doppler_parser
            )

            if self.include_lander_range_observable_flag:
                Enceladus_lander_range_parser_list = []
                Enceladus_lander_range_parser_list.append(
                    numerical_simulation.estimation.observation_parser(
                        numerical_simulation.estimation_setup.observation.n_way_range_type
                    )
                )
                Enceladus_lander_range_parser_list.append(
                    numerical_simulation.estimation.observation_parser(
                        ("Enceladus", lander_name)
                    )
                )
                Enceladus_lander_range_parser = numerical_simulation.estimation.observation_parser(
                    Enceladus_lander_range_parser_list,
                    combine_conditions=True
                )
                simulated_observations.set_constant_weight(
                    Enceladus_lander_range_weight,
                    Enceladus_lander_range_parser
                )

        # Perform the covariance analysis
        covariance_output = estimator.compute_covariance(covariance_input)

        # Retrieve covariance results
        correlations = covariance_output.correlations
        covariance = covariance_output.covariance
        formal_errors = covariance_output.formal_errors
        partials = covariance_output.weighted_design_matrix

        # Retrieve results with consider parameters
        consider_covariance_contribution = covariance_output.consider_covariance_contribution
        covariance_with_consider_parameters = covariance_output.unnormalized_covariance_with_consider_parameters
        formal_errors_with_consider_parameters = np.sqrt(np.diag(covariance_with_consider_parameters))
        correlations_with_consider_parameters = covariance_with_consider_parameters
        for i in range(nb_parameters):
            for j in range(nb_parameters):
                correlations_with_consider_parameters[i, j] /= formal_errors_with_consider_parameters[i] * formal_errors_with_consider_parameters[j]

        # Propagate formal errors
        output_times = np.arange(CovAnalysisConfig.simulation_start_epoch, simulation_end_epoch, 3600.0)
        propagated_formal_errors = numerical_simulation.estimation.propagate_formal_errors_rsw_split_output(covariance_output, estimator, output_times)

        # Compute condition number of output covariance matrix
        condition_number = np.linalg.cond(covariance)

        # Retrieve formal error of SH zonal gravity coefficients
        formal_error_cosine_coef = formal_errors[indices_cosine_coef[0]:
                                                 indices_cosine_coef[0] + indices_cosine_coef[1]]
        a_priori_constraints_cosine_coef = apriori_constraints[indices_cosine_coef[0]:
                                                               indices_cosine_coef[0] + indices_cosine_coef[1]]
        formal_error_zonal_cosine_coef = []
        a_priori_constraints_zonal_cosine_coef = []
        degrees = np.arange(CovAnalysisConfig.minimum_degree_c_enceladus,
                            CovAnalysisConfig.maximum_degree_gravity_enceladus, 1)
        i = 0
        for degree in degrees:
            a_priori_constraint = a_priori_constraints_cosine_coef[i]
            formal_error = formal_error_cosine_coef[i]
            formal_error_zonal_cosine_coef.append(formal_error)
            a_priori_constraints_zonal_cosine_coef.append(a_priori_constraint)
            i = i + degree + 1
        # Determine when formal error of SH gravity coeffs converges to a priori constraint
        for i in range(len(degrees)):
            delta = (np.absolute(formal_error_zonal_cosine_coef[i] - a_priori_constraints_zonal_cosine_coef[i]) /
                     a_priori_constraints_zonal_cosine_coef[i])
            if delta < 0.1:
                max_estimatable_degree_gravity_field = degrees[i] - 1
                break
            else:
                max_estimatable_degree_gravity_field = degrees[-1]

        plots_output_path = os.path.join(output_path, "plots")
        if self.save_covariance_results_flag:

            # Build output paths
            covariance_results_output_path = os.path.join(output_path, "covariance_results")
            os.makedirs(covariance_results_output_path, exist_ok=True)
            os.makedirs(plots_output_path, exist_ok=True)
            covariance_filename = os.path.join(covariance_results_output_path, "covariance_matrix.dat")
            np.savetxt(covariance_filename, covariance)

            correlations_filename = os.path.join(covariance_results_output_path, "correlations_matrix.dat")
            np.savetxt(correlations_filename, correlations)

            formal_errors_filename = os.path.join(covariance_results_output_path, "formal_errors.dat")
            np.savetxt(formal_errors_filename, formal_errors)

            partials_filename = os.path.join(covariance_results_output_path, "partials_matrix.dat")
            np.savetxt(partials_filename, partials)

            inv_apriori_constraints_filename = os.path.join(covariance_results_output_path,
                                                            "inv_a_priori_constraints.dat")
            np.savetxt(inv_apriori_constraints_filename, inv_apriori)

            apriori_constraints_filename = os.path.join(covariance_results_output_path, "a_priori_constraints.dat")
            np.savetxt(apriori_constraints_filename, apriori_constraints)

            condition_number_filename = os.path.join(covariance_results_output_path,
                                                     "condition_number_covariance_matrix.dat")
            np.savetxt(condition_number_filename, [condition_number])

            max_estimatable_degree_gravity_field_filename = os.path.join(covariance_results_output_path,
                                                                         "max_estimatable_degree_gravity_field.dat")
            np.savetxt(max_estimatable_degree_gravity_field_filename, [max_estimatable_degree_gravity_field])

            # Plot correlations
            PlottingUtil.plot_correlations(correlations,
                                           plots_output_path,
                                           "correlations.svg")

            PlottingUtil.plot_correlations(correlations,
                                           plots_output_path,
                                           "correlations.pdf")

            # Plot formal errors
            PlottingUtil.plot_formal_errors(formal_errors,
                                            plots_output_path,
                                            "formal_errors.pdf")

            # Plot formal errors of empirical accelerations
            formal_errors_empirical_accelerations = formal_errors[indices_empirical_acceleration_components[0]:
                                                                  indices_empirical_acceleration_components[0] +
                                                                  indices_empirical_acceleration_components[1]]
            formal_errors_empirical_accelerations_radial_direction = []
            formal_errors_empirical_accelerations_along_track_direction = []
            formal_errors_empirical_accelerations_across_track_direction = []
            nb_empirical_acceleration_groups = int(len(formal_errors_empirical_accelerations) / 3)
            for j in range(nb_empirical_acceleration_groups):
                formal_errors_empirical_accelerations_radial_direction.append(formal_errors_empirical_accelerations[j])
                formal_errors_empirical_accelerations_along_track_direction.append(
                    formal_errors_empirical_accelerations[j + 1])
                formal_errors_empirical_accelerations_across_track_direction.append(
                    formal_errors_empirical_accelerations[j + 2])
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.arange(1, nb_empirical_acceleration_groups + 1, 1),
                    formal_errors_empirical_accelerations_radial_direction,
                    label="Radial direction",
                    color="blue",
                    marker="o")
            ax.plot(np.arange(1, nb_empirical_acceleration_groups + 1, 1),
                    formal_errors_empirical_accelerations_along_track_direction,
                    label="Along track direction",
                    color="red",
                    marker="*")
            ax.plot(np.arange(1, nb_empirical_acceleration_groups + 1, 1),
                    formal_errors_empirical_accelerations_across_track_direction,
                    label="Across track direction",
                    color="green",
                    marker="x")
            fig.suptitle("Formal errors RSW empirical accelerations")
            ax.legend()
            ax.set_xlabel("Arc Index [-]")
            ax.set_ylabel(r"Formal Error  [m s$^{-2}$]")
            ax.set_yscale("log")
            ax.grid(True)
            os.makedirs(plots_output_path, exist_ok=True)
            file_output_path = os.path.join(plots_output_path, "formal_errors_empirical_accelerations_rsw.pdf")
            plt.savefig(file_output_path)
            plt.close(fig)

            # Plot formal error of SH gravity coefficients and a priori constraint
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(degrees, formal_error_zonal_cosine_coef, label="Formal error", color="blue")
            ax.plot(degrees, a_priori_constraints_zonal_cosine_coef, label="A priori constraint", color="orange")
            ax.set_xlabel("Degree zonal cosine coefficient  [-]")
            ax.set_ylabel(r"$\sigma$  [-]")
            ax.set_yscale("log")
            ax.grid(True)
            ax.set_title("Formal error zonal cosine coefficients")
            ax.legend(loc="lower right")
            file_output_path = os.path.join(plots_output_path, "formal_error_zonal_cosine_coefficients.pdf")
            plt.savefig(file_output_path)
            plt.close(fig)

            # Plot observation times for the entire mission
            sorted_observations = simulated_observations.sorted_observation_sets
            doppler_obs_times_malargue_current_arc = [(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t in
                                                      sorted_observations[observation.n_way_averaged_doppler_type][0][
                                                          0].observation_times]
            doppler_obs_time_newnorcia_current_arc = [(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t in
                                                      sorted_observations[observation.n_way_averaged_doppler_type][1][
                                                          0].observation_times]
            doppler_obs_time_cebreros_current_arc = [(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t in
                                                     sorted_observations[observation.n_way_averaged_doppler_type][2][
                                                         0].observation_times]
            PlottingUtil.plot_observation_times("entire mission",
                                                plots_output_path,
                                                "observation_times_entire_mission.pdf",
                                                doppler_obs_times_malargue_current_arc=doppler_obs_times_malargue_current_arc,
                                                doppler_obs_times_new_norcia_current_arc=doppler_obs_time_newnorcia_current_arc,
                                                doppler_obs_times_cebreros_current_arc=doppler_obs_time_cebreros_current_arc)

            # Save formal error interval for arc-wise initial position components
            formal_error_initial_state = formal_errors[indices_states[0]:indices_states[0] + indices_states[1]]
            formal_error_initial_position = []
            for i in range(nb_arcs):
                formal_error_initial_position_current_arc = formal_error_initial_state[6 * i: 6 * i + 3]
                for sigma in formal_error_initial_position_current_arc:
                    formal_error_initial_position.append(sigma)
            formal_error_initial_position_interval = [min(formal_error_initial_position), max(formal_error_initial_position)]
            formal_error_initial_position_interval_filename = os.path.join(covariance_results_output_path,
                                                                           "formal_error_initial_position_interval.dat")
            np.savetxt(formal_error_initial_position_interval_filename, formal_error_initial_position_interval)


        if self.save_simulation_results_flag:

            simulation_results_output_path = os.path.join(output_path, "simulation_results")
            os.makedirs(simulation_results_output_path, exist_ok=True)
            os.makedirs(plots_output_path, exist_ok=True)

            # Save simulation results for every arc
            formal_error_initial_position_radial_direction = []
            formal_error_initial_position_along_track_direction = []
            formal_error_initial_position_across_track_direction = []
            for i in range(nb_arcs):
                simulation_results_current_arc = simulation_results[i]
                state_history_current_arc = simulation_results_current_arc.state_history
                dependent_variable_history_current_arc = simulation_results_current_arc.dependent_variable_history

                dependent_variable_history_current_arc_array = result2array(dependent_variable_history_current_arc)
                dim = dependent_variable_history_current_arc_array.shape
                longitude_history = np.zeros((dim[0], 2))
                latitude_history = np.zeros((dim[0], 2))
                longitude_history[:, 0] = dependent_variable_history_current_arc_array[:, 0]
                latitude_history[:, 0] = dependent_variable_history_current_arc_array[:, 0]
                longitude_history[:, 1] = dependent_variable_history_current_arc_array[:,
                                          CovAnalysisConfig.indices_dependent_variables["longitude"][0]]
                latitude_history[:, 1] = dependent_variable_history_current_arc_array[:,
                                         CovAnalysisConfig.indices_dependent_variables["latitude"][0]]

                save2txt(state_history_current_arc,
                         f"state_history_arc_{i}.dat",
                         simulation_results_output_path)
                save2txt(dependent_variable_history_current_arc,
                         f"dependent_variable_history_arc_{i}.dat",
                         simulation_results_output_path)

                # Plot 3D trajectory of current arc
                PlottingUtil.plot_trajectory(state_history_current_arc,
                                             plots_output_path,
                                             f"trajectory_3d_arc_{i}.pdf",
                                             f"Arc {i}",
                                             "red")

                # Plot ground track of current arc
                PlottingUtil.plot_ground_track(latitude_history,
                                               longitude_history,
                                               plots_output_path,
                                               f"ground_track_arc_{i}.pdf",
                                               f"Arc {i}",
                                               "red")

                # Retrieve Doppler observation times for the current arc
                sorted_observations = simulated_observations.sorted_observation_sets
                doppler_obs_times_malargue_current_arc = [(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t
                                                          in
                                                          sorted_observations[observation.n_way_averaged_doppler_type][
                                                              0][
                                                              0].observation_times if
                                                          arc_start_times[i] <= t <= arc_start_times[i] +
                                                          self.arc_duration]
                doppler_obs_time_newnorcia_current_arc = [(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t
                                                          in
                                                          sorted_observations[observation.n_way_averaged_doppler_type][
                                                              1][
                                                              0].observation_times if
                                                          arc_start_times[i] <= t <= arc_start_times[i] +
                                                          self.arc_duration]
                doppler_obs_time_cebreros_current_arc = [(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t
                                                         in
                                                         sorted_observations[observation.n_way_averaged_doppler_type][
                                                             2][
                                                             0].observation_times if
                                                         arc_start_times[i] <= t <= arc_start_times[i] +
                                                         self.arc_duration]

                # Plot observation times
                PlottingUtil.plot_observation_times(f"Arc {i}",
                                                    plots_output_path,
                                                    f"observation_times_arc_{i}.pdf",
                                                    doppler_obs_times_malargue_current_arc=doppler_obs_times_malargue_current_arc,
                                                    doppler_obs_times_new_norcia_current_arc=doppler_obs_time_newnorcia_current_arc,
                                                    doppler_obs_times_cebreros_current_arc=doppler_obs_time_cebreros_current_arc)

                # Compute uncertainty in RSW coordinates for initial position elements
                initial_rsw_to_inertial_rotation_matrix = dependent_variable_history_current_arc_array[0,
                                                          CovAnalysisConfig.indices_dependent_variables[
                                                              "rsw_to_inertial_rotation_matrix"][0]:
                                                          CovAnalysisConfig.indices_dependent_variables[
                                                              "rsw_to_inertial_rotation_matrix"][1]]
                initial_rsw_to_inertial_rotation_matrix = np.reshape(initial_rsw_to_inertial_rotation_matrix, (3, 3))
                initial_inertial_to_rsw_rotation_matrix = np.linalg.inv(initial_rsw_to_inertial_rotation_matrix)
                formal_error_initial_inertial_position_current_arc = formal_errors[indices_states[0] + 6 * i:
                                                                                   indices_states[0] + 6 * i + 3].T
                formal_error_initial_position_rsw_current_arc = np.dot(initial_inertial_to_rsw_rotation_matrix, np.dot(
                    formal_error_initial_inertial_position_current_arc, initial_inertial_to_rsw_rotation_matrix))
                formal_error_initial_position_radial_direction.append(formal_error_initial_position_rsw_current_arc[0])
                formal_error_initial_position_along_track_direction.append(
                    formal_error_initial_position_rsw_current_arc[1])
                formal_error_initial_position_across_track_direction.append(
                    formal_error_initial_position_rsw_current_arc[2])

            fig, ax = plt.subplots(1, 1)
            ax.plot(np.arange(1, nb_arcs + 1, 1),
                    formal_error_initial_position_radial_direction,
                    label="Radial direction",
                    color="blue",
                    marker="o")
            ax.plot(np.arange(1, nb_arcs + 1, 1),
                    formal_error_initial_position_along_track_direction,
                    label="Along track direction",
                    color="red",
                    marker="*")
            ax.plot(np.arange(1, nb_arcs + 1, 1),
                    formal_error_initial_position_across_track_direction,
                    label="Across track direction",
                    color="green",
                    marker="x")
            fig.suptitle("Formal errors initial RSW position")
            ax.legend()
            ax.set_xlabel("Arc Index [-]")
            ax.set_ylabel("Formal Error  [m]")
            ax.grid(True)
            ax.set_yscale("log")
            os.makedirs(plots_output_path, exist_ok=True)
            file_output_path = os.path.join(plots_output_path, "formal_errors_initial_position_rsw.pdf")
            plt.savefig(file_output_path)
            plt.close(fig)

        print("Covariance analysis performed successfully!")
