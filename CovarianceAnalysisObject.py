#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Files and variables import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary import VehicleParameters as VehicleParam
from auxiliary.CovarianceAnalysisConfig import a_priori_libration_amplitude
from auxiliary.utilities import utilities as Util
from auxiliary.utilities import plotting_utilities as PlottingUtil
from auxiliary.utilities import environment_setup_utilities as EnvUtil
from auxiliary.utilities import covariance_analysis_utilities as CovUtil

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array
from tudatpy.math import interpolators
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import constants
from tudatpy import astro
from tudatpy.interface import spice

# Packages import
import numpy as np
import os
import matplotlib.pyplot as plt
import statistics
import matplotlib.lines as mlines
import time
#######################################################################################################################
### Class definition ##################################################################################################
#######################################################################################################################

class CovarianceAnalysis:

    def __init__(self,
                 initial_state_index: int,
                 save_simulation_results_flag: bool,
                 save_covariance_results_flag: bool,
                 save_correlation_matrix_flag: bool,
                 save_design_matrix_flag: bool,
                 save_inv_apriori_matrix_flag: bool,
                 save_obs_times_of_vehicle_flag: bool,
                 simulation_duration: float,
                 arc_duration: float,
                 tracking_arc_duration_Earth_GS: float,
                 tracking_arc_duration_Enceladus_lander: float,
                 kaula_constraint_multiplier: float,
                 a_priori_empirical_accelerations: float,
                 a_priori_lander_position: float,
                 a_priori_k2_love_number,
                 a_priori_rotation_pole_position: list[float],
                 a_priori_libration_amplitude: float,
                 a_priori_rotation_pole_rate: list[float],
                 a_priori_radiation_pressure_coefficient: float,
                 lander_to_include: list[str],
                 include_lander_range_observable_flag: bool,
                 use_range_bias_consider_parameter_flag: bool,
                 use_station_position_consider_parameter_flag: bool,
                 empirical_accelerations_arc_duration: float,
                 estimate_h2_love_number_flag: bool,
                 a_priori_h2_love_number: float,
                 ):
        self.initial_state_index = initial_state_index
        self.save_simulation_results_flag = save_simulation_results_flag
        self.save_covariance_results_flag = save_covariance_results_flag
        self.save_correlation_matrix_flag = save_correlation_matrix_flag
        self.save_design_matrix_flag = save_design_matrix_flag
        self.save_inv_apriori_matrix_flag = save_inv_apriori_matrix_flag
        self.save_obs_times_of_vehicle_flag = save_obs_times_of_vehicle_flag
        self.simulation_duration = simulation_duration
        self.arc_duration = arc_duration
        self.tracking_arc_duration_Earth_GS = tracking_arc_duration_Earth_GS
        self.tracking_arc_duration_Enceladus_lander = tracking_arc_duration_Enceladus_lander
        self.empirical_accelerations_arc_duration = empirical_accelerations_arc_duration
        self.kaula_constraint_multiplier = kaula_constraint_multiplier
        self.a_priori_empirical_accelerations = a_priori_empirical_accelerations
        self.a_priori_lander_position = a_priori_lander_position
        self.a_priori_k2_love_number = a_priori_k2_love_number
        self.a_priori_rotation_pole_position = a_priori_rotation_pole_position
        self.a_priori_libration_amplitude = a_priori_libration_amplitude
        self.a_priori_rotation_pole_rate = a_priori_rotation_pole_rate
        self.a_priori_radiation_pressure_coefficient = a_priori_radiation_pressure_coefficient
        self.lander_to_include = lander_to_include
        self.include_lander_range_observable_flag = include_lander_range_observable_flag
        self.use_range_bias_consider_parameter_flag = use_range_bias_consider_parameter_flag
        self.use_station_position_consider_parameter_flag = use_station_position_consider_parameter_flag
        self.estimate_h2_love_number_flag = estimate_h2_love_number_flag
        self.a_priori_h2_love_number = a_priori_h2_love_number

    @classmethod
    def from_config(cls):
        initial_state_index = CovAnalysisConfig.initial_state_index
        save_simulation_results_flag = False
        save_covariance_results_flag = False
        save_correlation_matrix_flag = False
        save_design_matrix_flag = False
        save_inv_apriori_matrix_flag = False
        save_obs_times_of_vehicle_flag = False
        simulation_duration = CovAnalysisConfig.simulation_duration
        arc_duration = CovAnalysisConfig.arc_duration
        tracking_arc_duration_Earth_gs = CovAnalysisConfig.tracking_arc_duration_Earth_GS
        tracking_arc_duration_Enceladus_lander = CovAnalysisConfig.tracking_arc_duration_Enceladus_lander
        kaula_constraint_multiplier = CovAnalysisConfig.kaula_constraint_multiplier
        a_priori_empirical_accelerations = CovAnalysisConfig.a_priori_empirical_accelerations
        a_priori_lander_position = CovAnalysisConfig.a_priori_lander_position
        a_priori_k2_love_number = CovAnalysisConfig.a_priori_k2_love_number
        a_priori_rotation_pole_position = CovAnalysisConfig.a_priori_rotation_pole_position
        a_priori_libration_amplitude = CovAnalysisConfig.a_priori_libration_amplitude
        a_priori_rotation_pole_rate = CovAnalysisConfig.a_priori_rotation_pole_rate
        a_priori_radiation_pressure_coefficient = CovAnalysisConfig.a_priori_radiation_pressure_coefficient
        lander_to_include = CovAnalysisConfig.lander_names
        include_lander_range_observable_flag = False
        use_range_bias_consider_parameter_flag = False
        use_station_position_consider_parameter_flag = True
        empirical_accelerations_arc_duration = CovAnalysisConfig.empirical_accelerations_arc_duration
        estimate_h2_love_number_flag = False
        a_priori_h2_love_number = CovAnalysisConfig.a_priori_h2_love_number
        return cls(initial_state_index,
                   save_simulation_results_flag,
                   save_covariance_results_flag,
                   save_correlation_matrix_flag,
                   save_design_matrix_flag,
                   save_inv_apriori_matrix_flag,
                   save_obs_times_of_vehicle_flag,
                   simulation_duration,
                   arc_duration,
                   tracking_arc_duration_Earth_gs,
                   tracking_arc_duration_Enceladus_lander,
                   kaula_constraint_multiplier,
                   a_priori_empirical_accelerations,
                   a_priori_lander_position,
                   a_priori_k2_love_number,
                   a_priori_rotation_pole_position,
                   a_priori_libration_amplitude,
                   a_priori_rotation_pole_rate,
                   a_priori_radiation_pressure_coefficient,
                   lander_to_include,
                   include_lander_range_observable_flag,
                   use_range_bias_consider_parameter_flag,
                   use_station_position_consider_parameter_flag,
                   empirical_accelerations_arc_duration,
                   estimate_h2_love_number_flag,
                   a_priori_h2_love_number,)

    def save_problem_configuration(self,
                                   output_directory: str):
        if len(self.lander_to_include) == 0:
            lander_to_include = "None"
        else:
            lander_to_include = self.lander_to_include[0]
            for lander in self.lander_to_include[1:]:
                lander_to_include = lander_to_include + "; " + lander

        problem_configuration = {
            "initial_state_index [-]": self.initial_state_index,
            "simulation_duration [days]": self.simulation_duration / constants.JULIAN_DAY,
            "arc_duration [days]": self.arc_duration / constants.JULIAN_DAY,
            "tracking_arc_duration_Earth_GS [hours]": self.tracking_arc_duration_Earth_GS / 3600.0,
            "tracking_arc_duration_Enceladus_lander [hours]": self.tracking_arc_duration_Enceladus_lander / 3600.0,
            "empirical_accelerations_arc_duration [hours]": self.empirical_accelerations_arc_duration / 3600.0,
            "kaula_constraint_multiplier [-]": self.kaula_constraint_multiplier,
            r"a_priori_empirical_accelerations [m s$^{-2}$]": self.a_priori_empirical_accelerations,
            "a_priori_lander_position [m]": self.a_priori_lander_position,
            "a_priori_k2_love_number [-]": complex(self.a_priori_k2_love_number[0], self.a_priori_k2_love_number[1]),
            "a_priori_h2_love_number [-]": self.a_priori_h2_love_number,
            "a_priori_rotation_pole_position [rad]": f"{self.a_priori_rotation_pole_position}",
            "a_priori_libration_amplitude [rad]": self.a_priori_libration_amplitude,
            r"a_priori_rotation_pole_rate [rad s$^{-1}$]": f"{self.a_priori_rotation_pole_rate}",
            "a_priori_radiation_pressure_coefficient [-]": self.a_priori_radiation_pressure_coefficient,
            "lander_to_include [-]": lander_to_include,
            "include_lander_range_observable_flag": self.include_lander_range_observable_flag,
            "use_range_bias_consider_parameter_flag": self.use_range_bias_consider_parameter_flag,
            "use_station_position_consider_parameter_flag": self.use_station_position_consider_parameter_flag,
            "estimate_h2_love_number_flag": self.estimate_h2_love_number_flag,
            "save_simulation_results_flag": self.save_simulation_results_flag,
            "save_covariance_results_flag": self.save_covariance_results_flag,
            "save_obs_times_of_vehicle_flag": self.save_obs_times_of_vehicle_flag
        }

        save2txt(problem_configuration, "problem_configuration.txt", output_directory)

    def perform_covariance_analysis(self,
                                    output_path: str):

        # Perform a feasibility check of problem setup
        if self.estimate_h2_love_number_flag is True and self.lander_to_include == []:
            raise ValueError("lander_to_include cannot be empty if you want to estimate h2 Love number.")

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
        body_settings.get("Enceladus").rotation_model_settings = EnvUtil.get_rotation_model_settings_enceladus_park_simplified(
            base_frame=CovAnalysisConfig.global_frame_orientation,
            target_frame="IAU_Enceladus"
        )

        # Set gravity field settings for Enceladus
        body_settings.get("Enceladus").gravity_field_settings = EnvUtil.get_gravity_field_settings_enceladus_park(
            CovAnalysisConfig.maximum_degree_gravity_enceladus)
        body_settings.get(
            "Enceladus").gravity_field_settings.scaled_mean_moment_of_inertia = CovAnalysisConfig.Enceladus_scaled_mean_moment_of_inertia
        gravity_field_variation_list_Enceladus = EnvUtil.get_gravity_field_variation_list_enceladus()
        body_settings.get("Enceladus").gravity_field_variation_settings = gravity_field_variation_list_Enceladus

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
            VehicleParam.drag_reference_area, np.array([VehicleParam.drag_coefficient, 0.0, 0.0])
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

        # Extract total number of propagation arcs during science phase
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
        lander_to_include = self.lander_to_include
        for lander_name in lander_to_include:
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

        # Define tracking arcs for Earth ground stations
        tracking_arcs_start_Earth_GS = []
        tracking_arcs_end_Earth_GS = []
        arc_start = CovAnalysisConfig.simulation_start_epoch
        while arc_start + constants.JULIAN_DAY + self.tracking_arc_duration_Earth_GS <= simulation_end_epoch:
            tracking_arc_start = arc_start + CovAnalysisConfig.tracking_delay_after_start_of_propagation_Earth_GS
            tracking_arcs_start_Earth_GS.append(tracking_arc_start)
            tracking_arcs_end_Earth_GS.append(tracking_arc_start + self.tracking_arc_duration_Earth_GS)
            arc_start = arc_start + constants.JULIAN_DAY

        # Define tracking arcs for surface landers
        tracking_arcs_start_Enceladus_lander = []
        tracking_arcs_end_Enceladus_lander = []
        arc_start = CovAnalysisConfig.simulation_start_epoch
        while arc_start + constants.JULIAN_DAY + self.tracking_arc_duration_Enceladus_lander <= simulation_end_epoch:
            tracking_arc_start = arc_start + CovAnalysisConfig.tracking_delay_after_start_of_propagation_Enceladus_lander
            tracking_arcs_start_Enceladus_lander.append(tracking_arc_start)
            tracking_arcs_end_Enceladus_lander.append(tracking_arc_start + self.tracking_arc_duration_Enceladus_lander)
            arc_start = arc_start + constants.JULIAN_DAY

        # Create observation settings for each link ends and observable
        # Define light-time calculations settings
        light_time_correction_settings = observation.first_order_relativistic_light_time_correction(["Sun"])

        # Define range biases settings
        biases_Earth_GS = []
        for i in range(len(tracking_arcs_start_Earth_GS)):
            biases_Earth_GS.append(np.array([CovAnalysisConfig.range_bias_Earth_GS]))
        range_bias_settings_Earth_GS = observation.arcwise_absolute_bias(tracking_arcs_start_Earth_GS, biases_Earth_GS, observation.receiver)
        # Define range biases settings
        biases_Enceladus_lander = []
        for i in range(len(tracking_arcs_start_Enceladus_lander)):
            biases_Enceladus_lander.append(np.array([CovAnalysisConfig.range_bias_Enceladus_lander]))
        range_bias_settings_Enceladus_lander = observation.arcwise_absolute_bias(tracking_arcs_start_Enceladus_lander, biases_Enceladus_lander, observation.receiver)

        # Define observation settings list
        observation_settings_list = []
        for i in range(len(ground_station_names)):
            link_end = link_ends[i]
            link_definition = observation.LinkDefinition(link_end)
            range_observation = observation.two_way_range(link_definition,
                                                          [light_time_correction_settings],
                                                          range_bias_settings_Earth_GS,)
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
                    range_bias_settings_Enceladus_lander
                )
                observation_settings_list.append(range_observation)

        # Set observation times for Earth GS
        observation_times_doppler_Earth_GS = []
        observation_times_range_Earth_GS = []
        for i in range(len(tracking_arcs_start_Earth_GS)):

            # Doppler observables
            t = tracking_arcs_start_Earth_GS[i]
            while t + CovAnalysisConfig.doppler_cadence <= tracking_arcs_end_Earth_GS[i]:
                observation_times_doppler_Earth_GS.append(t)
                t += CovAnalysisConfig.doppler_cadence

            # Range observables
            t = tracking_arcs_start_Earth_GS[i]
            while t + CovAnalysisConfig.range_cadence <= tracking_arcs_end_Earth_GS[i]:
                observation_times_range_Earth_GS.append(t)
                t += CovAnalysisConfig.range_cadence

        observation_times_per_type_Earth_GS = dict()
        observation_times_per_type_Earth_GS[observation.n_way_averaged_doppler_type] = observation_times_doppler_Earth_GS
        observation_times_per_type_Earth_GS[observation.n_way_range_type] = observation_times_range_Earth_GS

        # Set observation times for Enceladus landers
        observation_times_doppler_Enceladus_lander = []
        observation_times_range_Enceladus_lander = []
        for i in range(len(tracking_arcs_start_Enceladus_lander)):

            # Doppler observables
            t = tracking_arcs_start_Enceladus_lander[i]
            while t + CovAnalysisConfig.doppler_cadence <= tracking_arcs_end_Enceladus_lander[i]:
                observation_times_doppler_Enceladus_lander.append(t)
                t += CovAnalysisConfig.doppler_cadence

            # Range observables
            t = tracking_arcs_start_Enceladus_lander[i]
            while t + CovAnalysisConfig.range_cadence <= tracking_arcs_end_Enceladus_lander[i]:
                observation_times_range_Enceladus_lander.append(t)
                t += CovAnalysisConfig.range_cadence

        observation_times_per_type_Enceladus_lander = dict()
        observation_times_per_type_Enceladus_lander[observation.n_way_averaged_doppler_type] = observation_times_doppler_Enceladus_lander
        observation_times_per_type_Enceladus_lander[observation.n_way_range_type] = observation_times_range_Enceladus_lander

        # Define observation settings for both observables, and all link ends (i.e., all ground stations)
        observation_simulation_settings = []
        for i in range(len(ground_station_names)):
            link_end = link_ends[i]
            # Doppler observables
            observation_simulation_settings.append(observation.tabulated_simulation_settings(
                observation.n_way_averaged_doppler_type,
                observation.LinkDefinition(link_end),
                observation_times_per_type_Earth_GS[observation.n_way_averaged_doppler_type]
            ))
            # Range observables
            observation_simulation_settings.append(observation.tabulated_simulation_settings(
                observation.n_way_range_type,
                observation.LinkDefinition(link_end),
                observation_times_per_type_Earth_GS[observation.n_way_range_type]
            ))
        for i in range(len(ground_station_names), len(ground_station_names) + len(self.lander_to_include)):
            link_end = link_ends[i]
            # Doppler observables
            observation_simulation_settings.append(observation.tabulated_simulation_settings(
                observation.n_way_averaged_doppler_type,
                observation.LinkDefinition(link_end),
                observation_times_per_type_Enceladus_lander[observation.n_way_averaged_doppler_type]
            ))
            if self.include_lander_range_observable_flag:
                # Range observables
                observation_simulation_settings.append(observation.tabulated_simulation_settings(
                    observation.n_way_range_type,
                    observation.LinkDefinition(link_end),
                    observation_times_per_type_Enceladus_lander[observation.n_way_range_type]
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

        ###############################################################################################################
        ### Estimation setup ##########################################################################################
        ###############################################################################################################

        # Generate arc start times for empirical accelerations
        empirical_accelerations_arc_start_times = []
        empirical_accelerations_arc_end_times = []
        arc_start = CovAnalysisConfig.simulation_start_epoch
        while arc_start + self.empirical_accelerations_arc_duration <= simulation_end_epoch:
            empirical_accelerations_arc_start_times.append(arc_start)
            empirical_accelerations_arc_end_times.append(arc_start + self.empirical_accelerations_arc_duration)
            arc_start += self.empirical_accelerations_arc_duration

        # Define parameters to estimate
        # Add arc-wise initial states of the vehicle wrt Enceladus
        parameter_settings = numerical_simulation.estimation_setup.parameter.initial_states(propagator_settings,
                                                                                            bodies,
                                                                                            arc_start_times)
        # Add gravitational parameter of Enceladus
        parameter_settings.append(numerical_simulation.estimation_setup.parameter.gravitational_parameter("Enceladus"))
        # Add radiation pressure coefficient
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.radiation_pressure_coefficient("Vehicle")
        )
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
                                                                                            empirical_accelerations_arc_start_times)
        )
        # Add landers position
        for lander in self.lander_to_include:
            parameter_settings.append(
                numerical_simulation.estimation_setup.parameter.ground_station_position("Enceladus",
                                                                                        lander)
            )
        # Add k2 complex Love number
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.order_invariant_k_love_number(
                "Enceladus",
                2,
                True
            )
        )
        # Add pole position
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.iau_rotation_model_nominal_pole(
                "Enceladus",
            )
        )
        # Add libration amplitude
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.iau_rotation_model_longitudinal_libration(
                "Enceladus",
                CovAnalysisConfig.libration_angular_frequencies
            )
        )
        # Add pole rate
        parameter_settings.append(
            numerical_simulation.estimation_setup.parameter.iau_rotation_model_pole_rate(
                "Enceladus")
        )

        # Define consider parameters
        # Add arc-wise lander range biases as consider parameters
        consider_parameter_settings = []
        if self.use_range_bias_consider_parameter_flag:
            for link_end in link_ends:
                consider_parameter_settings.append(
                    numerical_simulation.estimation_setup.parameter.arcwise_absolute_observation_bias(
                        observation.LinkDefinition(link_end), observation.n_way_range_type, tracking_arcs_start_Earth_GS, observation.receiver
                    )
                )
        if self.use_station_position_consider_parameter_flag:
            for ground_station_name in ground_station_names:
                consider_parameter_settings.append(
                    numerical_simulation.estimation_setup.parameter.ground_station_position("Earth", ground_station_name)
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

        # Print content of observation collection
        simulated_observations.print_observation_sets_start_and_size()

        # Retrieve sorted observation sets
        sorted_observations = simulated_observations.sorted_observation_sets

        ###############################################################################################################
        ### Covariance analysis #######################################################################################
        ###############################################################################################################

        # Define a priori covariance matrix
        inv_apriori = np.zeros((nb_parameters, nb_parameters))

        # Initialise empty storage for indices of estimation parameters
        indices_estimation_parameters = []

        # Set a priori constraints for vehicle states
        indices_states = parameters_to_estimate.indices_for_parameter_type((
            numerical_simulation.estimation_setup.parameter.arc_wise_initial_body_state_type, ("Vehicle", "")))[0]
        indices_estimation_parameters.append(indices_states)
        for i in range(indices_states[1] // 6):
            for j in range(3):
                inv_apriori[indices_states[0] + i * 6 + j, indices_states[
                    0] + i * 6 + j] = CovAnalysisConfig.a_priori_position ** -2
                inv_apriori[indices_states[0] + i * 6 + j + 3, indices_states[
                    0] + i * 6 + j + 3] = CovAnalysisConfig.a_priori_velocity ** -2

        # Set a priori constraint for Enceladus' gravitational parameter
        indices_mu = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.gravitational_parameter_type, ("Enceladus", "")))[0]
        indices_estimation_parameters.append(indices_mu)
        for i in range(indices_mu[1]):
            inv_apriori[
                indices_mu[0] + i, indices_mu[0] + i
            ] = CovAnalysisConfig.a_priori_gravitational_parameter_enceladus ** -2

        # Set a priori constraint for radiation pressure coefficient
        indices_radiation_pressure_coefficient = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.radiation_pressure_coefficient_type, ("Vehicle", "")))[0]
        indices_estimation_parameters.append(indices_radiation_pressure_coefficient)
        for i in range(indices_radiation_pressure_coefficient[1]):
            inv_apriori[
                indices_radiation_pressure_coefficient[0] + i, indices_radiation_pressure_coefficient[0] + i
            ] = self.a_priori_radiation_pressure_coefficient ** -2

        # Set a priori constraint for Enceladus' gravity field coefficients
        indices_cosine_coef = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.spherical_harmonics_cosine_coefficient_block_type,
             ("Enceladus", "")))[0]
        indices_estimation_parameters.append(indices_cosine_coef)
        indices_sine_coef = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.spherical_harmonics_sine_coefficient_block_type,
             ("Enceladus", "")))[0]
        indices_estimation_parameters.append(indices_sine_coef)
        # Apply Kaula's constraint to Enceladus' gravity field
        inv_apriori = CovUtil.apply_kaula_constraint_a_priori(self.kaula_constraint_multiplier,
                                                              CovAnalysisConfig.maximum_degree_gravity_enceladus,
                                                              CovAnalysisConfig.minimum_degree_c_enceladus,
                                                              indices_cosine_coef,
                                                              indices_sine_coef,
                                                              inv_apriori)
        # Overwrite Kaula's rule with existing uncertainties
        inv_apriori[indices_cosine_coef[0] + 2, indices_cosine_coef[0] + 2] = CovAnalysisConfig.a_priori_c20 ** -2
        inv_apriori[indices_cosine_coef[0] + 3, indices_cosine_coef[0] + 3] = CovAnalysisConfig.a_priori_c21 ** -2
        inv_apriori[indices_cosine_coef[0] + 4, indices_cosine_coef[0] + 4] = CovAnalysisConfig.a_priori_c22 ** -2
        inv_apriori[indices_cosine_coef[0] + 5, indices_cosine_coef[0] + 5] = CovAnalysisConfig.a_priori_c30 ** -2
        inv_apriori[indices_sine_coef[0] + 1, indices_sine_coef[0] + 1] = CovAnalysisConfig.a_priori_s21 ** -2
        inv_apriori[indices_sine_coef[0] + 2, indices_sine_coef[0] + 2] = CovAnalysisConfig.a_priori_s22 ** -2

        # Set a priori constraint for empirical accelerations
        indices_empirical_acceleration_components = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.arc_wise_empirical_acceleration_coefficients_type,
             ("Vehicle", "Enceladus")))[0]
        indices_estimation_parameters.append(indices_empirical_acceleration_components)
        for i in range(indices_empirical_acceleration_components[1]):
            inv_apriori[indices_empirical_acceleration_components[0] + i, indices_empirical_acceleration_components[
                0] + i] = self.a_priori_empirical_accelerations ** -2

        # Set a priori constraint for landers' position
        for lander_name in self.lander_to_include:
            indices_lander_position = parameters_to_estimate.indices_for_parameter_type(
                (numerical_simulation.estimation_setup.parameter.ground_station_position_type,
                 ("Enceladus", lander_name)))[0]
            indices_estimation_parameters.append(indices_lander_position)
            for i in range(indices_lander_position[1]):
                inv_apriori[indices_lander_position[0] + i, indices_lander_position[
                    0] + i] = self.a_priori_lander_position ** -2

        # Set a priori constraint for k2 tidal love number
        indices_tidal_love_number = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.full_degree_tidal_love_number_type,
            ("Enceladus", "")))[0]
        indices_estimation_parameters.append(indices_tidal_love_number)
        for i in range(indices_tidal_love_number[1]):
            inv_apriori[indices_tidal_love_number[0] + i, indices_tidal_love_number[0] + i] = (
                    self.a_priori_k2_love_number[i] ** -2)

        # Set a priori constraint for pole position
        indices_pole_position = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.nominal_rotation_pole_position_type,
            ("Enceladus", "")))[0]
        indices_estimation_parameters.append(indices_pole_position)
        for i in range(indices_pole_position[1]):
            inv_apriori[indices_pole_position[0] + i, indices_pole_position[0] + i] = (
                    self.a_priori_rotation_pole_position[i] ** -2)

        # Set a priori constraint for libration amplitude
        indices_libration_amplitude = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.rotation_longitudinal_libration_terms_type,
            ("Enceladus", "")))[0]
        indices_estimation_parameters.append(indices_libration_amplitude)
        for i in range(indices_libration_amplitude[1]):
            inv_apriori[indices_libration_amplitude[0] + i, indices_libration_amplitude[0] + i] = (
                    self.a_priori_libration_amplitude ** -2)

        # Set a priori constraint for pole rate
        indices_pole_rate = parameters_to_estimate.indices_for_parameter_type(
            (numerical_simulation.estimation_setup.parameter.rotation_pole_position_rate_type, ("Enceladus", "")))[0]
        for i in range(indices_pole_rate[1]):
            inv_apriori[indices_pole_rate[0] + i, indices_pole_rate[0] + i] = (
                    self.a_priori_rotation_pole_rate[i] ** -2)

        # Retrieve full vector of a priori constraints
        apriori_constraints = np.reciprocal(np.sqrt(np.diagonal(inv_apriori)))

        # Define consider parameters covariance
        if self.use_range_bias_consider_parameter_flag or self.use_station_position_consider_parameter_flag:
            nb_consider_parameters = 0
            if self.use_range_bias_consider_parameter_flag:
                nb_consider_parameters += nb_arcs * (len(ground_station_names) + len(self.lander_to_include))
            if self.use_station_position_consider_parameter_flag:
                nb_consider_parameters += 3 * len(ground_station_names)

            consider_parameter_covariance = np.zeros((nb_consider_parameters, nb_consider_parameters))
            # Set consider covariance for range biases
            if self.use_range_bias_consider_parameter_flag:
                indices_range_bias_Earth_ground_station = (0, nb_arcs*len(ground_station_names))
                for i in range(indices_range_bias_Earth_ground_station[1]):
                    consider_parameter_covariance[indices_range_bias_Earth_ground_station[0] + i, indices_range_bias_Earth_ground_station[0] + i] = (
                            CovAnalysisConfig.a_priori_range_bias_Earth_ground_station ** 2)
                indices_range_bias_lander = (nb_arcs*len(ground_station_names), nb_arcs * len(self.lander_to_include) )
                for i in range(indices_range_bias_lander[1]):
                    consider_parameter_covariance[indices_range_bias_lander[0] + i, indices_range_bias_lander[0] + i] = (
                            CovAnalysisConfig.a_priori_range_bias_lander ** 2)

            # Set consider covariance for ground station and lander position
            if self.use_station_position_consider_parameter_flag:
                if self.use_range_bias_consider_parameter_flag:
                    indices_station_position_Earth_ground_station = (
                        nb_arcs * (len(ground_station_names) + len(self.lander_to_include)), 3 * len(ground_station_names)
                    )
                else:
                    indices_station_position_Earth_ground_station = (0, 3 * len(ground_station_names))
                for i in range(indices_station_position_Earth_ground_station[1]):
                    consider_parameter_covariance[indices_station_position_Earth_ground_station[0] + i, indices_station_position_Earth_ground_station[0] + i] = (
                        CovAnalysisConfig.a_priori_station_position_Earth_ground_station ** 2
                    )

            # Create input object for covariance analysis
            covariance_input = numerical_simulation.estimation.CovarianceAnalysisInput(simulated_observations,
                                                                                       inv_apriori,
                                                                                       consider_parameter_covariance)
        else:
            # Create input object for covariance analysis
            covariance_input = numerical_simulation.estimation.CovarianceAnalysisInput(simulated_observations,
                                                                                       inv_apriori)

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
        normalized_covariance = covariance_output.normalized_covariance
        formal_errors = covariance_output.formal_errors
        partials = covariance_output.design_matrix
        weight_matrix_diagonal = covariance_input.weight_matrix_diagonal
        consider_normalization_terms = covariance_output.consider_normalization_factors

        # Build weight matrix out of the main diagonal
        weight_matrix = np.zeros((len(weight_matrix_diagonal), len(weight_matrix_diagonal)))
        for i in range(len(weight_matrix_diagonal)):
            weight_matrix[i, i] = weight_matrix_diagonal[i]

        # Retrieve results with consider parameters
        if self.use_range_bias_consider_parameter_flag or self.use_station_position_consider_parameter_flag:
            consider_covariance_contribution = covariance_output.consider_covariance_contribution
            covariance_with_consider_parameters = covariance_output.unnormalized_covariance_with_consider_parameters
            normalized_covariance_with_consider_parameters = covariance_output.normalized_covariance_with_consider_parameters
            formal_errors_with_consider_parameters = np.sqrt(np.diag(covariance_with_consider_parameters))
            correlations_with_consider_parameters = covariance_with_consider_parameters
            for i in range(nb_parameters):
                for j in range(nb_parameters):
                    correlations_with_consider_parameters[i, j] /= formal_errors_with_consider_parameters[i] * formal_errors_with_consider_parameters[j]

        # print("Pausing the simulation ...")
        # time.sleep(3)

        # Add h2 Love number to parameters to estimate and retrieve covariance matrix
        if self.estimate_h2_love_number_flag:
            print("Adding radial displacement Love number at full degree of Enceladus due to Saturn for degree 2.")

            nb_parameters_extended = nb_parameters + 1

            indices_radial_love_number = (nb_parameters_extended - 1, 1)

            inv_apriori_extended = np.zeros((nb_parameters_extended, nb_parameters_extended))
            inv_apriori_extended[:nb_parameters, :nb_parameters] = inv_apriori
            inv_apriori_extended[nb_parameters_extended - 1, nb_parameters_extended - 1] = self.a_priori_h2_love_number ** -2

            indices_lander = (indices_lander_position[0], 3)

            gravitational_parameter_ratio = bodies.get("Saturn").gravitational_parameter / bodies.get("Enceladus").gravitational_parameter

            Enceladus_radius = spice.get_average_radius("Enceladus")
            station_state_spherical = np.zeros((6,))
            station_state_spherical[0] = Enceladus_radius + CovAnalysisConfig.lander_coordinates[lander_to_include[0]][0]
            station_state_spherical[1] = CovAnalysisConfig.lander_coordinates[lander_to_include[0]][1]
            station_state_spherical[2] = CovAnalysisConfig.lander_coordinates[lander_to_include[0]][2]
            station_position_cartesian = astro.element_conversion.spherical_to_cartesian(station_state_spherical)[:3]

            sorted_observation_epochs = CovUtil.retrieve_sorted_observation_epochs(simulated_observations)
            partials_extended = CovUtil.extend_design_matrix_to_h2_love_number(partials,
                                                                               indices_lander,
                                                                               gravitational_parameter_ratio,
                                                                               station_position_cartesian,
                                                                               Enceladus_radius,
                                                                               sorted_observation_epochs)
            normalization_terms_extended = CovUtil.get_normalization_terms(partials_extended)
            normalized_partials_extended = CovUtil.normalize_design_matrix(partials_extended, normalization_terms_extended)
            normalized_inv_apriori_extended = CovUtil.normalize_inv_apriori_covariance_matrix(inv_apriori_extended, normalization_terms_extended)


            normalized_covariance_extended = np.linalg.inv(np.dot(normalized_partials_extended.T,
                                                       np.dot(weight_matrix, normalized_partials_extended)) + normalized_inv_apriori_extended)
            covariance_extended = CovUtil.unnormalize_covariance_matrix(normalized_covariance_extended, normalization_terms_extended)
            formal_errors_extended = CovUtil.get_formal_errors(covariance_extended)
            correlations_extended = CovUtil.get_correlation_matrix(covariance_extended, formal_errors_extended)

            if self.use_range_bias_consider_parameter_flag or self.use_station_position_consider_parameter_flag:
                normalized_design_matrix_consider_parameters = covariance_output.normalized_design_matrix_consider_parameters
                term_1 = (np.dot(normalized_covariance_extended, np.dot(normalized_partials_extended.T, weight_matrix)))
                normalized_consider_covariance = CovUtil.normalize_covariance_matrix(consider_parameter_covariance, consider_normalization_terms)
                term_2 = np.dot(normalized_design_matrix_consider_parameters,
                                np.dot(normalized_consider_covariance, normalized_design_matrix_consider_parameters.T))
                term_3 = term_1.T
                normalized_covariance_with_consider_parameters_extended = (normalized_covariance_extended +
                                                                np.dot(term_1, np.dot(term_2, term_3)))
                covariance_with_consider_parameters_extended = CovUtil.unnormalize_covariance_matrix(normalized_covariance_with_consider_parameters_extended, normalization_terms_extended)
                formal_errors_with_consider_parameters_extended = CovUtil.get_formal_errors(covariance_with_consider_parameters_extended)
                correlations_with_consider_parameters_extended = CovUtil.get_correlation_matrix(covariance_with_consider_parameters_extended, formal_errors_with_consider_parameters_extended)

        print("Covariance analysis performed successfully!")

        if self.use_range_bias_consider_parameter_flag or self.use_station_position_consider_parameter_flag:
            if self.estimate_h2_love_number_flag:
                covariance_to_use = covariance_with_consider_parameters_extended
                normalized_covariance_to_use = normalized_covariance_with_consider_parameters_extended
                correlations_to_use = correlations_with_consider_parameters_extended
                formal_errors_to_use = formal_errors_with_consider_parameters_extended
                partials_to_use = partials_extended
                inv_apriori_to_use = inv_apriori_extended
            else:
                covariance_to_use = covariance_with_consider_parameters
                normalized_covariance_to_use = normalized_covariance_with_consider_parameters
                correlations_to_use = correlations_with_consider_parameters
                formal_errors_to_use = formal_errors_with_consider_parameters
                partials_to_use = partials
                inv_apriori_to_use = inv_apriori
        else:
            if self.estimate_h2_love_number_flag:
                covariance_to_use = covariance_extended
                normalized_covariance_to_use = normalized_covariance_extended
                correlations_to_use = correlations_extended
                formal_errors_to_use = formal_errors_extended
                partials_to_use = partials_extended
                inv_apriori_to_use = inv_apriori_extended
            else:
                covariance_to_use = covariance
                normalized_covariance_to_use = normalized_covariance
                correlations_to_use = correlations
                formal_errors_to_use = formal_errors
                partials_to_use = partials
                inv_apriori_to_use = inv_apriori

        # Rotate formal errors of initial state components to RSW frame
        formal_error_initial_position_rsw = np.zeros((nb_arcs, 3))
        for i in range(nb_arcs):
            simulation_results_current_arc = simulation_results[i]
            dependent_variable_history_current_arc = simulation_results_current_arc.dependent_variable_history
            dependent_variable_history_current_arc_array = result2array(dependent_variable_history_current_arc)
            initial_rsw_to_inertial_rotation_matrix = dependent_variable_history_current_arc_array[0,
                                                      CovAnalysisConfig.indices_dependent_variables[
                                                          "rsw_to_inertial_rotation_matrix"][0]:
                                                      CovAnalysisConfig.indices_dependent_variables[
                                                          "rsw_to_inertial_rotation_matrix"][1]]
            initial_rsw_to_inertial_rotation_matrix = np.reshape(initial_rsw_to_inertial_rotation_matrix, (3, 3))
            initial_inertial_to_rsw_rotation_matrix = initial_rsw_to_inertial_rotation_matrix.T

            covariance_initial_position_inertial = covariance_to_use[
                                           indices_states[0] + 6 * i : indices_states[0] + 6 * i + 3,
                                           indices_states[0] + 6 * i: indices_states[0] + 6 * i + 3
                                           ]
            covariance_initial_position_rsw = np.dot(initial_inertial_to_rsw_rotation_matrix, np.dot(
                covariance_initial_position_inertial, initial_rsw_to_inertial_rotation_matrix))
            formal_error_initial_position_rsw[i, :] = np.sqrt(np.diag(covariance_initial_position_rsw))

        # Retrieve formal error of empirical accelerations
        formal_error_empirical_accelerations_rsw_list = []
        nb_empirical_accelerations_arcs = len(empirical_accelerations_arc_start_times)
        for i in range(nb_empirical_accelerations_arcs):
            formal_error_empirical_accelerations_rsw = formal_errors[
                indices_empirical_acceleration_components[0] + 3*i:
                indices_empirical_acceleration_components[0] + 3*i + 3]
            formal_error_empirical_accelerations_rsw_list.append(formal_error_empirical_accelerations_rsw)

        # # Propagate formal errors
        # output_times = list(np.arange(CovAnalysisConfig.simulation_start_epoch, simulation_end_epoch, 3600.0))
        # propagated_formal_errors = numerical_simulation.estimation.propagate_formal_errors(covariance_to_use, estimator.state_transition_interface, output_times)

        # Compute condition number of output covariance matrix
        condition_number = np.linalg.cond(covariance_to_use)

        # Retrieve rms of formal error of SH gravity coefficients
        formal_error_cosine_coef = formal_errors_to_use[indices_cosine_coef[0]:
                                                        indices_cosine_coef[0] + indices_cosine_coef[1]]
        formal_error_sine_coef = formal_errors_to_use[indices_sine_coef[0]:
                                                      indices_sine_coef[0] + indices_sine_coef[1]]
        a_priori_constraints_cosine_coef = apriori_constraints[indices_cosine_coef[0]:
                                                               indices_cosine_coef[0] + indices_cosine_coef[1]]
        a_priori_constraints_sine_coef = apriori_constraints[indices_sine_coef[0]:
                                                             indices_sine_coef[0] + indices_sine_coef[1]]
        formal_error_cosine_coef_per_deg = []
        formal_error_sine_coef_per_deg = []
        a_priori_constraints_cosine_coef_per_deg = []
        a_priori_constraints_sine_coef_per_deg = []
        degrees = np.arange(CovAnalysisConfig.minimum_degree_c_enceladus,
                            CovAnalysisConfig.maximum_degree_gravity_enceladus + 1, 1)

        start_index_cosine_deg = 0
        start_index_sine_deg = 0
        for deg in range(CovAnalysisConfig.minimum_degree_c_enceladus, CovAnalysisConfig.maximum_degree_gravity_enceladus + 1):

            # Cosine coefficients
            rms_apriori = 0
            rms_error = 0
            for j in range(deg + 1):
                rms_apriori += a_priori_constraints_cosine_coef[start_index_cosine_deg + j] ** 2
                rms_error += formal_error_cosine_coef[start_index_cosine_deg + j] ** 2
            start_index_cosine_deg += deg + 1

            a_priori_constraints_cosine_coef_per_deg.append(np.sqrt(rms_apriori / (deg + 1)))
            formal_error_cosine_coef_per_deg.append(np.sqrt(rms_error / (deg + 1)))

            # Sine coefficients
            rms_apriori = 0
            rms_error = 0
            for j in range(deg):
                rms_apriori += a_priori_constraints_sine_coef[start_index_sine_deg + j] ** 2
                rms_error += formal_error_sine_coef[start_index_sine_deg + j] ** 2
            start_index_sine_deg += deg

            a_priori_constraints_sine_coef_per_deg.append(np.sqrt(rms_apriori / deg))
            formal_error_sine_coef_per_deg.append(np.sqrt(rms_error / deg))

        # Determine when formal error of SH gravity coeffs converges to a priori constraint
        for i in range(len(degrees)):
            delta = (np.absolute(formal_error_cosine_coef_per_deg[i] - a_priori_constraints_cosine_coef_per_deg[i]) /
                     a_priori_constraints_cosine_coef_per_deg[i]) * 100
            if delta <= 10:
                max_estimatable_degree_gravity_field = degrees[i] - 1
                break
            else:
                max_estimatable_degree_gravity_field = degrees[-1]

        # Retrieve rms of formal error of degree 2 cosine coefficients
        rms_formal_error_degree_2_cosine_coef = formal_error_cosine_coef_per_deg[0]
        rms_formal_error_degree_2_sine_coef = formal_error_sine_coef_per_deg[0]

        # Retrieve formal error of gravitational Love number
        formal_error_love_number = formal_errors_to_use[indices_tidal_love_number[0] :
                                                 indices_tidal_love_number[0] + indices_tidal_love_number[1]]

        # Retrieve formal error of radial Love number
        if self.estimate_h2_love_number_flag:
            formal_error_radial_love_number = formal_errors_to_use[indices_radial_love_number[0] :
                                                            indices_radial_love_number[0] + indices_radial_love_number[1]]

        # Retrieve formal error of libration amplitudes
        formal_error_libration_amplitude = formal_errors_to_use[indices_libration_amplitude[0] :
                                                         indices_libration_amplitude[0] + indices_libration_amplitude[1]]

        # Retrieve formal error of radiation pressure coefficient
        formal_error_radiation_pressure_coefficient = formal_errors_to_use[
                                                      indices_radiation_pressure_coefficient[0] :
                                                      indices_radiation_pressure_coefficient[0] + indices_radiation_pressure_coefficient[1]]

        # Retrieve formal error of pole position
        formal_error_pole_position = formal_errors_to_use[indices_pole_position[0] :
                                                   indices_pole_position[0] + indices_pole_position[1]]

        # Retrieve formal error of pole rate
        formal_error_pole_rate = formal_errors_to_use[indices_pole_rate[0] :
                                               indices_pole_rate[0] + indices_pole_rate[1]]

        # Compute ratio of lander data to ground station data
        if lander_to_include != []:
            indices_lander_position_first_lander = parameters_to_estimate.indices_for_parameter_type(
                (numerical_simulation.estimation_setup.parameter.ground_station_position_type,
                ("Enceladus", lander_to_include[0])))[0]
            nb_landers = len(lander_to_include)
            indices_lander_position = [indices_lander_position_first_lander[0], 3 * nb_landers]

            nb_lander_observations = CovUtil.get_number_observations_for_station_type(partials_to_use,
                                                                                      "lander",
                                                                                      indices_lander_position)
            nb_ground_station_observations = CovUtil.get_number_observations_for_station_type(partials_to_use,
                                                                                              "ground_station",
                                                                                              indices_lander_position)
            nb_observations_ratio = nb_lander_observations / nb_ground_station_observations
        else:
            nb_observations_ratio = 0.0

        # Compute number of observation epochs per station
        doppler_obs_times_GS = []
        for i in range(len(CovAnalysisConfig.ground_station_names)):
            doppler_obs_times_GS.append([(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t in
                                             sorted_observations[observation.n_way_averaged_doppler_type][i][0].observation_times])
            nb_observations_per_GS = CovUtil.get_number_of_observations_per_station_type(doppler_obs_times_GS,
                                                                                         CovAnalysisConfig.ground_station_names)
        if self.lander_to_include != []:
            doppler_obs_times_lander = []
            for i in range(len(self.lander_to_include)):
                doppler_obs_times_lander.append([(t - CovAnalysisConfig.simulation_start_epoch) / 3600.0 for t in
                                             sorted_observations[observation.n_way_averaged_doppler_type][i +
                                                len(CovAnalysisConfig.ground_station_names)][0].observation_times])
            nb_observations_per_lander = CovUtil.get_number_of_observations_per_station_type(doppler_obs_times_lander,
                                                                                             self.lander_to_include)

        plots_output_path = os.path.join(output_path, "plots")
        if self.save_covariance_results_flag:

            # Build output paths
            covariance_results_output_path = os.path.join(output_path, "covariance_results")
            os.makedirs(covariance_results_output_path, exist_ok=True)
            os.makedirs(plots_output_path, exist_ok=True)
            covariance_filename = os.path.join(covariance_results_output_path, "covariance_matrix.dat")
            np.savetxt(covariance_filename, covariance)

            # Save correlation matrix
            if self.save_correlation_matrix_flag:
                correlations_filename = os.path.join(covariance_results_output_path, "correlations_matrix.dat")
                np.savetxt(correlations_filename, correlations_to_use)

            # Save formal errors
            formal_errors_filename = os.path.join(covariance_results_output_path, "formal_errors.dat")
            np.savetxt(formal_errors_filename, formal_errors_to_use)

            # # Save propagated formal errors
            # propagated_formal_errors_filename = os.path.join(covariance_results_output_path, "propagated_formal_errors.dat")
            # np.savetxt(propagated_formal_errors_filename, propagated_formal_errors)

            # Save design matrix
            if self.save_design_matrix_flag:
                partials_filename = os.path.join(covariance_results_output_path, "partials_matrix.dat")
                np.savetxt(partials_filename, partials_to_use)

            # Save inverse a priori constraints matrix
            if self.save_inv_apriori_matrix_flag:
                inv_apriori_constraints_filename = os.path.join(covariance_results_output_path,
                                                            "inv_a_priori_constraints.dat")
                np.savetxt(inv_apriori_constraints_filename, inv_apriori_to_use)

            # Save a priori constraints
            apriori_constraints_filename = os.path.join(covariance_results_output_path, "a_priori_constraints.dat")
            np.savetxt(apriori_constraints_filename, apriori_constraints)

            # Save condition number
            condition_number_filename = os.path.join(covariance_results_output_path,
                                                     "condition_number_covariance_matrix.dat")
            np.savetxt(condition_number_filename, [condition_number])

            # Save rms of formal error of degree 2 cosine and sine coefficients
            rms_formal_error_degree_2_filename = os.path.join(covariance_results_output_path,
                                                              "rms_formal_error_degree_2.dat")
            np.savetxt(rms_formal_error_degree_2_filename,
                       [rms_formal_error_degree_2_cosine_coef, rms_formal_error_degree_2_sine_coef])

            # Save maximum estimatable degree of the gravity field
            max_estimatable_degree_gravity_field_filename = os.path.join(covariance_results_output_path,
                                                                         "max_estimatable_degree_gravity_field.dat")
            np.savetxt(max_estimatable_degree_gravity_field_filename, [max_estimatable_degree_gravity_field])

            # Plot nominal correlations
            PlottingUtil.plot_correlations(correlations_to_use,
                                           plots_output_path,
                                           "correlations.svg")

            PlottingUtil.plot_correlations(correlations_to_use,
                                           plots_output_path,
                                           "correlations.pdf")

            # Plot formal errors
            PlottingUtil.plot_formal_errors(formal_errors_to_use,
                                            plots_output_path,
                                            "formal_errors.pdf")

            # Plot formal error of SH gravity coefficients and a priori constraint
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(degrees, formal_error_cosine_coef_per_deg, label="RMS of formal error", color="blue")
            ax.plot(degrees, a_priori_constraints_cosine_coef_per_deg, label="A priori constraint", color="orange")
            ax.set_xlabel("Degree  [-]")
            ax.set_ylabel(r"$\sigma$  [-]")
            ax.set_yscale("log")
            ax.grid(True)
            ax.set_title("RMS of formal error of SH gravity cosine coefficients per degree")
            ax.legend(loc="lower right")
            file_output_path = os.path.join(plots_output_path, "rms_formal_error_cosine_coefficients.pdf")
            plt.savefig(file_output_path)
            plt.close(fig)

            # Save formal error interval for arc-wise initial position components in RSW frame
            formal_error_initial_position_rsw_interval = np.array([
                [
                    min(formal_error_initial_position_rsw[:, 0]),
                    min(formal_error_initial_position_rsw[:, 1]),
                    min(formal_error_initial_position_rsw[:, 2]),
                ],
                [
                    statistics.median(formal_error_initial_position_rsw[:, 0]),
                    statistics.median(formal_error_initial_position_rsw[:, 1]),
                    statistics.median(formal_error_initial_position_rsw[:, 2]),
                ],
                [
                    max(formal_error_initial_position_rsw[:, 0]),
                    max(formal_error_initial_position_rsw[:, 1]),
                    max(formal_error_initial_position_rsw[:, 2]),
                ]
            ])
            formal_error_initial_position_interval_rsw_filename = os.path.join(
                covariance_results_output_path,
                "formal_error_initial_position_rsw_interval.dat"
            )
            np.savetxt(formal_error_initial_position_interval_rsw_filename, formal_error_initial_position_rsw_interval)

            # Filter formal error of empirical accelerations
            formal_error_empirical_accelerations_rsw_filtered_list = []
            for i in range(nb_empirical_accelerations_arcs):
                if (self.a_priori_empirical_accelerations - 1e-10 <=
                    formal_error_empirical_accelerations_rsw_list[i][0] < self.a_priori_empirical_accelerations + 1e-10
                    and self.a_priori_empirical_accelerations - 1e-10 <=
                    formal_error_empirical_accelerations_rsw_list[i][1] < self.a_priori_empirical_accelerations + 1e-10
                    and self.a_priori_empirical_accelerations - 1e-10 <=
                    formal_error_empirical_accelerations_rsw_list[i][2] < self.a_priori_empirical_accelerations + 1e-10):
                    continue
                else:
                    formal_error_empirical_accelerations_rsw_filtered_list.append(
                        formal_error_empirical_accelerations_rsw_list[i])


            # Save formal error interval for filtered empirical acceleration components
            formal_error_empirical_accelerations_radial_direction = []
            formal_error_empirical_accelerations_along_track_direction = []
            formal_error_empirical_accelerations_cross_track_direction = []
            for i in range(len((formal_error_empirical_accelerations_rsw_filtered_list))):
                formal_error_empirical_accelerations_rsw_current_arc = formal_error_empirical_accelerations_rsw_filtered_list[i]
                formal_error_empirical_accelerations_radial_direction.append(
                    formal_error_empirical_accelerations_rsw_current_arc[0])
                formal_error_empirical_accelerations_along_track_direction.append(
                    formal_error_empirical_accelerations_rsw_current_arc[1])
                formal_error_empirical_accelerations_cross_track_direction.append(
                    formal_error_empirical_accelerations_rsw_current_arc[2])
            formal_error_empirical_accelerations_rsw_interval = np.array([
                [
                    min(formal_error_empirical_accelerations_radial_direction),
                    min(formal_error_empirical_accelerations_along_track_direction),
                    min(formal_error_empirical_accelerations_cross_track_direction),
                ],
                [
                    statistics.median(formal_error_empirical_accelerations_radial_direction),
                    statistics.median(formal_error_empirical_accelerations_along_track_direction),
                    statistics.median(formal_error_empirical_accelerations_cross_track_direction),
                ],
                [
                    max(formal_error_empirical_accelerations_radial_direction),
                    max(formal_error_empirical_accelerations_along_track_direction),
                    max(formal_error_empirical_accelerations_cross_track_direction),
                ]
            ])
            formal_error_empirical_accelerations_rsw_interval_filename = os.path.join(
                covariance_results_output_path,
                "formal_error_empirical_accelerations_rsw_interval.dat"
            )
            np.savetxt(formal_error_empirical_accelerations_rsw_interval_filename,
                       formal_error_empirical_accelerations_rsw_interval)

            # Plot formal error for empirical acceleration components in RSW frame
            fig = plt.figure()
            ax = fig.add_subplot()
            for i in range(nb_arcs):
                formal_error_empirical_accelerations_rsw_current_arc = formal_error_empirical_accelerations_rsw_list[i]
                ax.scatter(i, formal_error_empirical_accelerations_rsw_current_arc[0], color="orange")
                ax.scatter(i, formal_error_empirical_accelerations_rsw_current_arc[1], color="blue")
                ax.scatter(i, formal_error_empirical_accelerations_rsw_current_arc[2], color="red")
            ax.set_xlabel(r"$t - t_{0}$  [days]")
            ax.set_ylabel(r"$\sigma$  [m/s$^{2}$]")
            ax.set_title("Formal error empirical accelerations")
            ax.set_yscale("log")
            ax.grid(True)
            radial_handle = mlines.Line2D([], [], color="orange", label="Radial")
            along_track_handle = mlines.Line2D([], [], color="blue", label="Along-track")
            cross_track_handle = mlines.Line2D([], [], color="red", label="Cross-track")
            ax.legend(handles=[radial_handle, along_track_handle, cross_track_handle])
            file_output_path = os.path.join(plots_output_path, "formal_error_empirical_accelerations_rsw.pdf")
            fig.savefig(file_output_path)
            plt.close(fig)

            # Save formal error of gravitational Love number
            formal_error_love_number_filename = os.path.join(covariance_results_output_path,
                                                             "formal_error_love_number.dat")
            np.savetxt(formal_error_love_number_filename, formal_error_love_number)

            # Save formal error of radial displacement Love number
            if self.estimate_h2_love_number_flag:
                formal_error_radial_love_number_filename = os.path.join(covariance_results_output_path,
                                                                        "formal_error_radial_love_number.dat")
                np.savetxt(formal_error_radial_love_number_filename, formal_error_radial_love_number)

            # Save formal error of libration amplitude
            formal_error_libration_amplitude_filename = os.path.join(covariance_results_output_path,
                                                            "formal_error_libration_amplitude.dat")
            np.savetxt(formal_error_libration_amplitude_filename, [formal_error_libration_amplitude])

            # Save formal error of radiation pressure coefficient
            formal_error_radiation_pressure_coefficient_filename = os.path.join(covariance_results_output_path,
                                                                                "formal_error_radiation_pressure_coefficient.dat")
            np.savetxt(formal_error_radiation_pressure_coefficient_filename, [formal_error_radiation_pressure_coefficient])

            # Save formal error of pole position
            formal_error_pole_position_filename = os.path.join(covariance_results_output_path,
                                                               "formal_error_pole_position.dat")
            np.savetxt(formal_error_pole_position_filename,[formal_error_pole_position])

            # Save formal error of pole rate
            formal_error_pole_rate_filename = os.path.join(covariance_results_output_path,
                                                               "formal_error_pole_rate.dat")
            np.savetxt(formal_error_pole_rate_filename,[formal_error_pole_rate])

            # Save ratio of number of lander observations to number of ground station observations
            nb_observations_ratio_filename = os.path.join(covariance_results_output_path, "nb_observations_ratio.dat")
            np.savetxt(nb_observations_ratio_filename, [nb_observations_ratio])

            # Save number of observations per station
            save2txt(nb_observations_per_GS, "nb_observations_per_GS.txt", covariance_results_output_path)
            if self.lander_to_include != [ ]:
                save2txt(nb_observations_per_lander, "nb_observations_per_lander.txt", covariance_results_output_path)

            # Save indices of estimation parameters
            indices_estimation_parameters_filename = os.path.join(covariance_results_output_path, "indices_estimation_parameters.dat")
            np.savetxt(indices_estimation_parameters_filename, indices_estimation_parameters)

            print("Covariance results saved.")

        if self.save_obs_times_of_vehicle_flag:

            # Plot Earth observation times for the entire mission
            PlottingUtil.plot_observation_times("entire mission",
                                                plots_output_path,
                                                "observation_times_Earth_GS_entire_mission.pdf",
                                                doppler_obs_times_GS,
                                                ['New Norcia', 'Cebreros', 'Malargue'])

            # Plot lander observation times for the entire mission
            if self.lander_to_include != [ ]:
                PlottingUtil.plot_observation_times("entire mission",
                                                    plots_output_path,
                                                    "observation_times_Enceladus_lander_entire_mission.pdf",
                                                    doppler_obs_times_lander,
                                                    self.lander_to_include)

        if self.save_simulation_results_flag:

            simulation_results_output_path = os.path.join(output_path, "simulation_results")
            os.makedirs(simulation_results_output_path, exist_ok=True)
            os.makedirs(plots_output_path, exist_ok=True)

            # Save simulation results for every arc
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

                print("Simulation results saved.")


        print("Run terminated successfully!")
