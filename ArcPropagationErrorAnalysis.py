#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Files and variables import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary import VehicleParameters as VehicleParam
from auxiliary.utilities import utilities as Util
from auxiliary.utilities import environment_setup_utilities as EnvUtil

# Tudat import
from tudatpy import numerical_simulation
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.kernel.interface import spice
from tudatpy.math import interpolators

# Packages import
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os


def study_propagation_error(initial_state_index,
                            arc_duration,
                            time_stamp,
                            fontsize):
    ###################################################################################################################
    ### Configuration
    ###################################################################################################################

    # Define output folder
    output_folder = f"./output/arc_wise_propagation_error"

    # Build output_path
    output_path = os.path.join(output_folder, time_stamp,
                               f"arc_duration_{arc_duration / constants.JULIAN_DAY}_days/initial_state_K{initial_state_index}")
    simulation_results_output_path = os.path.join(output_path, "simulation_results")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(simulation_results_output_path, exist_ok=True)

    ###################################################################################################################
    ### Environment setup #############################################################################################
    ###################################################################################################################

    # Retrieve default body settings
    body_settings = numerical_simulation.environment_setup.get_default_body_settings(CovAnalysisConfig.bodies_to_create,
                                                                                     CovAnalysisConfig.global_frame_origin,
                                                                                     CovAnalysisConfig.global_frame_orientation)

    # Set rotation model settings for Enceladus
    synodic_rotation_rate_enceladus = EnvUtil.get_synodic_rotation_model_enceladus(
        CovAnalysisConfig.simulation_start_epoch)
    initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000",
                                                                                 "IAU_Enceladus",
                                                                                 CovAnalysisConfig.simulation_start_epoch)
    body_settings.get(
        "Enceladus").rotation_model_settings = numerical_simulation.environment_setup.rotation_model.simple(
        "J2000", "IAU_Enceladus", initial_orientation_enceladus,
        CovAnalysisConfig.simulation_start_epoch, synodic_rotation_rate_enceladus)

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
    if initial_state_index == 1:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_1.dat")
        nominal_dependent_variable_history_array = np.loadtxt("nominal_orbits/nominal_dependent_variable_history_1.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
        nominal_dependent_variable_history = Util.array2dict(nominal_dependent_variable_history_array)
    elif initial_state_index == 2:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_2.dat")
        nominal_dependent_variable_history_array = np.loadtxt("nominal_orbits/nominal_dependent_variable_history_2.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
        nominal_dependent_variable_history = Util.array2dict(nominal_dependent_variable_history_array)
    elif initial_state_index == 3:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_3.dat")
        nominal_dependent_variable_history_array = np.loadtxt("nominal_orbits/nominal_dependent_variable_history_3.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
        nominal_dependent_variable_history = Util.array2dict(nominal_dependent_variable_history_array)
    else:
        raise ValueError("Initial state index not valid")

    # Create numerical integrator settings
    integrator_settings = CovAnalysisConfig.integrator_settings

    arc_start_times = []
    arc_end_times = []
    arc_start = CovAnalysisConfig.simulation_start_epoch
    while arc_start + arc_duration <= CovAnalysisConfig.simulation_end_epoch:
        arc_start_times.append(arc_start)
        arc_end_times.append(arc_start + arc_duration)
        arc_start += arc_duration

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
    propagator_settings = numerical_simulation.propagation_setup.propagator.multi_arc(propagator_settings_list)

    # Propagate dynamics and retrieve simulation results
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
    simulation_results = dynamics_simulator.propagation_results.single_arc_results

    # Save simulation results for every arc
    for i in range(nb_arcs):
        simulation_results_current_arc = simulation_results[i]
        state_history_current_arc = simulation_results_current_arc.state_history
        dependent_variable_history_current_arc = simulation_results_current_arc.dependent_variable_history

        save2txt(state_history_current_arc,
                 f"state_history_arc_{i}.dat",
                 simulation_results_output_path)
        save2txt(dependent_variable_history_current_arc,
                 f"dependent_variable_history_arc_{i}.dat",
                 simulation_results_output_path)

        # Compute position and acceleration difference with respect to nominal orbit
        epochs_arc = list(state_history_current_arc.keys())
        lagrange_interpolation_settings = interpolators.lagrange_interpolation(
            number_of_points=CovAnalysisConfig.number_of_points
        )
        state_interpolator = interpolators.create_one_dimensional_vector_interpolator(nominal_state_history,
                                                                                      lagrange_interpolation_settings)
        dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            nominal_dependent_variable_history,
            lagrange_interpolation_settings)
        state_difference_history = dict()
        acceleration_difference_history = dict()
        for epoch in epochs_arc:
            state_arc = state_history_current_arc[epoch]
            dependent_variable_arc = dependent_variable_history_current_arc[epoch]
            if epoch == CovAnalysisConfig.simulation_start_epoch:
                state_nominal_orbit = nominal_state_history[epoch]
                dependent_variable_nominal_orbit = nominal_dependent_variable_history[epoch]
            else:
                state_nominal_orbit = state_interpolator.interpolate(epoch)
                dependent_variable_nominal_orbit = dependent_variable_interpolator.interpolate(epoch)

            state_difference_history[epoch] = state_arc - state_nominal_orbit
            acceleration_difference_history[epoch] = dependent_variable_arc[3:6] - dependent_variable_nominal_orbit[3:6]

        save2txt(state_difference_history,
                 f"state_difference_history_arc_{i}.dat",
                 simulation_results_output_path)
        save2txt(acceleration_difference_history,
                 f"acceleration_difference_norm_history_arc_{i}.dat",
                 simulation_results_output_path)


def study_delta_v_correction(initial_state_index,
                             arc_duration,
                             time_stamp,
                             max_final_position_deviation):
    ###################################################################################################################
    ### Configuration
    ###################################################################################################################

    # Define output folder
    output_folder = f"./output/arc_wise_propagation_error"

    # Build output_path
    output_path = os.path.join(output_folder, time_stamp,
                               f"arc_duration_{arc_duration / constants.JULIAN_DAY}_days/initial_state_K{initial_state_index}")
    simulation_results_output_path = os.path.join(output_path, "delta_v_correction")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(simulation_results_output_path, exist_ok=True)

    ###################################################################################################################
    ### Environment setup #############################################################################################
    ###################################################################################################################

    # Retrieve default body settings
    body_settings = numerical_simulation.environment_setup.get_default_body_settings(CovAnalysisConfig.bodies_to_create,
                                                                                     CovAnalysisConfig.global_frame_origin,
                                                                                     CovAnalysisConfig.global_frame_orientation)

    # Set rotation model settings for Enceladus
    synodic_rotation_rate_enceladus = EnvUtil.get_synodic_rotation_model_enceladus(
        CovAnalysisConfig.simulation_start_epoch)
    initial_orientation_enceladus = spice.compute_rotation_matrix_between_frames("J2000",
                                                                                 "IAU_Enceladus",
                                                                                 CovAnalysisConfig.simulation_start_epoch)
    body_settings.get(
        "Enceladus").rotation_model_settings = numerical_simulation.environment_setup.rotation_model.simple(
        "J2000", "IAU_Enceladus", initial_orientation_enceladus,
        CovAnalysisConfig.simulation_start_epoch, synodic_rotation_rate_enceladus)

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
    if initial_state_index == 1:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_1.dat")
        nominal_dependent_variable_history_array = np.loadtxt("nominal_orbits/nominal_dependent_variable_history_1.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
        nominal_dependent_variable_history = Util.array2dict(nominal_dependent_variable_history_array)
    elif initial_state_index == 2:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_2.dat")
        nominal_dependent_variable_history_array = np.loadtxt("nominal_orbits/nominal_dependent_variable_history_2.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
        nominal_dependent_variable_history = Util.array2dict(nominal_dependent_variable_history_array)
    elif initial_state_index == 3:
        nominal_state_history_array = np.loadtxt("nominal_orbits/nominal_state_history_3.dat")
        nominal_dependent_variable_history_array = np.loadtxt("nominal_orbits/nominal_dependent_variable_history_3.dat")
        nominal_state_history = Util.array2dict(nominal_state_history_array)
        nominal_dependent_variable_history = Util.array2dict(nominal_dependent_variable_history_array)
    else:
        raise ValueError("Initial state index not valid")

    # Create numerical integrator settings
    integrator_settings = CovAnalysisConfig.integrator_settings

    arc_start_times = []
    arc_end_times = []
    arc_start = CovAnalysisConfig.simulation_start_epoch
    while arc_start + arc_duration <= CovAnalysisConfig.simulation_end_epoch:
        arc_start_times.append(arc_start)
        arc_end_times.append(arc_start + arc_duration)
        arc_start += arc_duration

    # Extract total number of propagationa arcs during science phase
    nb_arcs = len(arc_start_times)
    print(f'Total number of arcs for the science phase: {nb_arcs}')

    initial_velocity_correction_list = []

    for i in range(nb_arcs):

        print(f"Correcting arc {i}")
        if i == 0:
            initial_state = nominal_state_history[CovAnalysisConfig.simulation_start_epoch]
        else:
            lagrange_interpolation_settings = interpolators.lagrange_interpolation(
                number_of_points=CovAnalysisConfig.number_of_points
            )
            interpolator = interpolators.create_one_dimensional_vector_interpolator(nominal_state_history,
                                                                                    lagrange_interpolation_settings)
            initial_state = interpolator.interpolate(arc_start_times[i])

        # Initialize initial velocity correction
        initial_velocity_correction = np.zeros((3,))
        initial_velocity_correction_total = initial_velocity_correction

        # Initial convergence flag
        convergence_check = False

        while not convergence_check:
            initial_state[3:] = initial_state[3:] + initial_velocity_correction

            propagator_settings = numerical_simulation.propagation_setup.propagator.translational(
                central_bodies,
                acceleration_models,
                bodies_to_propagate,
                initial_state,
                arc_start_times[i],
                integrator_settings,
                numerical_simulation.propagation_setup.propagator.time_termination(arc_end_times[i]),
                numerical_simulation.propagation_setup.propagator.cowell,
                CovAnalysisConfig.dependent_variables_to_save
            )

            parameter_settings = numerical_simulation.estimation_setup.parameter.initial_states(
                propagator_settings, bodies)
            sensitivity_parameters = numerical_simulation.estimation_setup.create_parameter_set(parameter_settings,
                                                                                                bodies,
                                                                                                propagator_settings)

            # Propagate variational equations
            variational_equations_solver = numerical_simulation.create_variational_equations_solver(
                bodies, propagator_settings, sensitivity_parameters)

            state_transition_matrix_history_current_arc = variational_equations_solver.state_transition_matrix_history
            state_history_current_arc = variational_equations_solver.state_history

            # Compute position difference with respect to nominal orbit
            epochs_arc = list(state_history_current_arc.keys())
            lagrange_interpolation_settings = interpolators.lagrange_interpolation(
                number_of_points=CovAnalysisConfig.number_of_points
            )
            interpolator = interpolators.create_one_dimensional_vector_interpolator(nominal_state_history,
                                                                                    lagrange_interpolation_settings)
            state_difference_history = dict()
            for epoch in epochs_arc:
                state_arc = state_history_current_arc[epoch]
                if epoch == CovAnalysisConfig.simulation_start_epoch:
                    state_nominal_orbit = nominal_state_history[epoch]
                else:
                    state_nominal_orbit = interpolator.interpolate(epoch)

                state_difference_history[epoch] = state_arc - state_nominal_orbit

            # Compute required velocity change at beginning of arc to meet required final state
            final_state_transition_matrix = state_transition_matrix_history_current_arc[epochs_arc[-1]]
            initial_velocity_correction = np.dot(np.linalg.inv(final_state_transition_matrix[:3, 3:6]),
                                                 -state_difference_history[epochs_arc[-1]][:3])

            # Update total initial velocity correction required for current arc
            initial_velocity_correction_total = initial_velocity_correction_total + initial_velocity_correction

            # Check convergence status and save state history
            final_position_deviation_norm = np.linalg.norm(state_difference_history[epochs_arc[-1]][:3])
            if final_position_deviation_norm <= max_final_position_deviation:
                convergence_check = True
                save2txt(state_history_current_arc,
                         f"state_history_arc_{i}.dat",
                         simulation_results_output_path)
                save2txt(state_difference_history,
                         f"state_difference_history_arc_{i}.dat",
                         simulation_results_output_path)

            print(f"Final position deviation norm = {final_position_deviation_norm}")

        # Save total initial velocity correction for current arc
        initial_velocity_correction_list.append(np.linalg.norm(initial_velocity_correction_total))

    # Save initial velocity correction list
    np.savetxt(simulation_results_output_path + "/multi_arc_initial_velocity_correction.dat",
               initial_velocity_correction_list)

    del propagator_settings


def main():
    initial_state_index = 3
    arc_duration_days = 1

    # Retrieve current time stamp
    time_stamp = datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = ["/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
                       "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp"]
    spice.load_standard_kernels(kernels_to_load)

    study_propagation_error(initial_state_index,
                            arc_duration_days * constants.JULIAN_DAY,
                            time_stamp,
                            12)
    study_delta_v_correction(initial_state_index,
                             arc_duration_days * constants.JULIAN_DAY,
                             time_stamp,
                             1e-3)


if __name__ == "__main__":
    main()
