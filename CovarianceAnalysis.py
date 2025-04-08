#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Files and variables import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary import BenedikterInitialStates as Benedikter
from auxiliary import VehicleParameters as VehicleParam
from auxiliary import utilities as Util

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.data import save2txt
from tudatpy.util import result2array
from tudatpy import constants
from tudatpy.astro import element_conversion
from tudatpy.kernel.interface import spice_interface, spice
from tudatpy.math import interpolators
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.plotting import trajectory_3d

# Packages import
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os


def covariance_analysis(initial_state_index,
                        save_results_flag,
                        ):
    ###################################################################################################################
    ### Configuration
    ###################################################################################################################

    if save_results_flag:
        # Retrieve current time stamp
        time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

        # Define output folder
        output_folder = "./output/covariance_analysis"

        # Build output_path
        output_path = os.path.join(output_folder, time_stamp)
        os.makedirs(output_path, exist_ok=True)

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = ["/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
                       "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp"]
    spice.load_standard_kernels(kernels_to_load)

    ###################################################################################################################
    ### Environment setup #############################################################################################
    ###################################################################################################################

    # Retrieve default body settings
    body_settings = numerical_simulation.environment_setup.get_default_body_settings(CovAnalysisConfig.bodies_to_create,
                                                                                     CovAnalysisConfig.global_frame_origin,
                                                                                     CovAnalysisConfig.global_frame_orientation)

    # Set rotation model settings for Enceladus
    synodic_rotation_rate_enceladus = Util.get_synodic_rotation_model_enceladus(CovAnalysisConfig.simulation_start_epoch)
    initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000",
                                                                                 "IAU_Enceladus",
                                                                                 CovAnalysisConfig.simulation_start_epoch)
    body_settings.get("Enceladus").rotation_model_settings = numerical_simulation.environment_setup.rotation_model.simple(
        "J2000", "IAU_Enceladus", initial_orientation_enceladus,
        CovAnalysisConfig.simulation_start_epoch, synodic_rotation_rate_enceladus)

    # Set gravity field settings for Enceladus
    body_settings.get("Enceladus").gravity_field_settings = Util.get_gravity_field_settings_enceladus_park()
    body_settings.get(
        "Enceladus").gravity_field_settings.scaled_mean_moment_of_inertia = CovAnalysisConfig.Enceladus_scaled_mean_moment_of_inertia

    # Set gravity field settings for Saturn
    body_settings.get("Saturn").gravity_field_settings = Util.get_gravity_field_settings_saturn_iess()
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
        VehicleParam.radiation_pressure_reference_area, VehicleParam.radiation_pressure_coefficient, CovAnalysisConfig.occulting_bodies
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
    if initial_state_index == 1:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_1.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
    elif initial_state_index == 2:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_2.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
    elif initial_state_index == 3:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_3.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
    else:
        raise ValueError("Initial state index not valid")

    # Create numerical integrator settings
    integrator_settings = CovAnalysisConfig.integrator_settings

    arc_start_times = []
    arc_end_times = []
    arc_start = CovAnalysisConfig.simulation_start_epoch
    while arc_start + CovAnalysisConfig.arc_duration <= CovAnalysisConfig.simulation_end_epoch:
        arc_start_times.append(arc_start)
        arc_end_times.append(arc_start + CovAnalysisConfig.arc_duration)
        arc_start += CovAnalysisConfig.arc_duration

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
            lagrange_interpolation_settings =interpolators.lagrange_interpolation(
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
    propagator_settings = numerical_simulation.propagation_setup.propagator.multi_arc(propagator_settings_list)

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

    # Define link ends for two-way Doppler and range observables, for each ground station
    link_ends = []
    for station in ground_station_names:
        link_ends_per_station = dict()
        link_ends_per_station[observation.transmitter]  = observation.body_reference_point_link_end_id(
            "Earth", station
        )
        link_ends_per_station[observation.receiver] = observation.body_reference_point_link_end_id(
            "Earth", station
        )
        link_ends_per_station[observation.reflector1] = observation.body_origin_link_end_id("Vehicle")
        link_ends.append(link_ends_per_station)

    # Define tracking arcs
    tracking_arcs_start = []
    tracking_arcs_end = []
    for arc_start in arc_start_times:
        tracking_arc_start = arc_start + CovAnalysisConfig.tracking_delay_after_stat_of_propagation
        tracking_arcs_start.append(tracking_arc_start)
        tracking_arcs_end.append(tracking_arc_start + CovAnalysisConfig.tracking_arc_duration)

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
    for link_end in link_ends:
        link_definition = observation.LinkDefinition(link_end)
        range_observation = observation.two_way_range(link_definition,
                                                      [light_time_correction_settings],
                                                      range_bias_settings)
        doppler_observation = observation.two_way_doppler_averaged(link_definition,
                                                                   [light_time_correction_settings])
        observation_settings_list.append(doppler_observation)
        observation_settings_list.append(range_observation)

    # Define observation simulation times for both Doppler and range observarbles
    doppler_cadence = 60
    range_cadence = 300.0

    observation_times_doppler = []
    observation_times_range = []

    for i in range(nb_arcs):

        # Doppler observables
        time = tracking_arcs_start[i]
        while time <= tracking_arcs_end[i]:
            observation_times_doppler.append(time)
            time += doppler_cadence

        # Range observables
        time = tracking_arcs_start[i]
        while time <= tracking_arcs_end[i]:
            observation_times_range.append(time)
            time += range_cadence


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
    ### Estimation setup
    ###################################################################################################################

    # Define parameters to estimate
    # Add arc-wise initial states of the vehicle wrt Enceladus
    parameter_settings = numerical_simulation.estimation_setup.parameter.initial_states(propagator_settings,
                                                                                        bodies,
                                                                                        arc_start_times)

    # Create parameters to estimate object
    parameters_to_estimate = numerical_simulation.estimation_setup.create_parameter_set(parameter_settings,
                                                                                        bodies,
                                                                                        propagator_settings) # consider_parameter_settings
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
    a_priori_position = 5.0e3
    a_priori_velocity = 0.5
    indices_states = parameters_to_estimate.indices_for_parameter_type((
        numerical_simulation.estimation_setup.parameter.arc_wise_initial_body_state_type, ("Vehicle", "")))[0]
    for i in range(indices_states[1]//6):
        for j in range(3):
            inv_apriori[indices_states[0]+i*6+j, indices_states[0]+i*6+j] = a_priori_position**-2
            inv_apriori[indices_states[0] + i * 6 + j + 3, indices_states[0] + i * 6 + j + 3] = a_priori_velocity ** -2

    # Retrieve full vector of a priori constraints
    apriori_constraints = np.reciprocal(np.sqrt(np.diagonal(inv_apriori)))

    # Create input object for covariance analysis
    covariance_input = numerical_simulation.estimation.CovarianceAnalysisInput(simulated_observations, inv_apriori) # consider_covariance
    covariance_input.define_covariance_settings(reintegrate_variational_equations=False, save_design_matrix=True)

    # Apply weights to simulated observations
    doppler_noise = CovAnalysisConfig.doppler_noise
    range_noise = CovAnalysisConfig.range_noise
    weights_per_observable = {observation.n_way_averaged_doppler_type: doppler_noise ** -2,
                              observation.n_way_range_type: range_noise ** -2,}
    covariance_input.set_constant_weight_per_observable(weights_per_observable)

    # Perform the covariance analysis
    covariance_output = estimator.compute_covariance(covariance_input)

    # Retrieve covariance results
    correlations = covariance_output.correlations
    covariance = covariance_output.covariance
    formal_errors = covariance_output.formal_errors
    partials = covariance_output.weighted_design_matrix

    # Print the formal errors
    print("Formal errors")
    print(formal_errors)

    # Get simulation results over first propagation arc
    simulation_results_first_arc = simulation_results[0]
    state_history_first_arc = simulation_results_first_arc.state_history
    dependent_variables_first_arc = result2array(simulation_results_first_arc.dependent_variable_history)

    fig = trajectory_3d(state_history_first_arc,
                        ["Vehicle"],
                        "Enceladus",
                        [],
                        "J2000",
                        True)



def main():

    initial_state_index = 1
    save_results_flag = False

    covariance_analysis(initial_state_index,
                        save_results_flag)


if __name__ == "__main__":
    main()