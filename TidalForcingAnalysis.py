
#######################################################################################################################
### Import statements #################################################################################################
#######################################################################################################################

# Files and variables import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary.utilities import environment_setup_utilities as EnvUtil
from auxiliary import BenedikterInitialStates as Benedikter

# Tudat import
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.data import save2txt
from tudatpy.util import result2array
from tudatpy import numerical_simulation

# General import
import numpy as np
import math
import sympy as sym
import os


def kronecker_delta(order):
     if order == 0:
         delta = 1.0
     else:
        delta = 0.0

     return delta


def get_normalised_legendre_function(degree, order, arg):
    normalisation_factor = np.sqrt(
        math.factorial(degree - order) * (2 * degree + 1) * (2 - kronecker_delta(order)) / math.factorial(degree + order)
    )

    mu_sym = sym.symbols('mu')
    f_sym = (mu_sym ** 2 - 1) ** degree
    derivative = f_sym.diff(mu_sym, order + degree)
    derivative = derivative.subs(mu_sym, arg)

    unnormalised_legendre_function = (
            ( (1 - arg**2) ** (order / 2) ) * (1 / (2 ** degree * math.factorial(degree))) * derivative
                                     )
    normalised_legendre_function = normalisation_factor * unnormalised_legendre_function

    return normalised_legendre_function


def get_saturn_colatitude_and_longitude_history(simulation_start_epoch,
                                                simulation_end_epoch,):
    # Load SPICE kernels for simulation
    spice.load_standard_kernels()
    kernels_to_load = [
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de438.bsp",
        "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat427.bsp",
        # "/Users/mattiacontarini/Documents/Code/Thesis/kernels/de440.bsp",
        # "/Users/mattiacontarini/Documents/Code/Thesis/kernels/sat441l.bsp",
        # "kernels/de440.bsp",
        # "kernels/sat441l.bsp"
    ]
    spice.load_standard_kernels(kernels_to_load)

    # Retrieve default body settings
    body_settings = numerical_simulation.environment_setup.get_default_body_settings(
        CovAnalysisConfig.bodies_to_create,
        CovAnalysisConfig.global_frame_origin,
        CovAnalysisConfig.global_frame_orientation)
    body_settings.get("Enceladus").rotation_model_settings = EnvUtil.get_rotation_model_settings_enceladus_park_simplified(
        base_frame=CovAnalysisConfig.global_frame_orientation,
        target_frame="IAU_Enceladus"
    )

    # Add vehicle
    body_settings.add_empty_settings("Vehicle")

    # Create system of bodies
    bodies = numerical_simulation.environment_setup.create_system_of_bodies(body_settings)

    # Define bodies that are propagated
    bodies_to_propagate = ["Vehicle"]

    # Define central bodies of propagation
    central_bodies = ["Enceladus"]

    # Acceleration settings
    acceleration_settings_on_vehicle = dict(
        Enceladus = [numerical_simulation.propagation_setup.acceleration.point_mass_gravity()]
    )
    acceleration_settings = {"Vehicle": acceleration_settings_on_vehicle}

    # Create acceleration models
    acceleration_models = numerical_simulation.propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies
    )

    # Create termination settings
    termination_settings = numerical_simulation.propagation_setup.propagator.time_termination(simulation_end_epoch)

    # Create numerical integrator settings
    fixed_step_size = 15
    integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size,
        numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56,
        order_to_use=numerical_simulation.propagation_setup.integrator.OrderToIntegrate.higher
    )

    dependent_variables_to_save = [
        numerical_simulation.propagation_setup.dependent_variable.central_body_fixed_spherical_position(
            "Saturn",
            "Enceladus"
        ),
        numerical_simulation.propagation_setup.dependent_variable.keplerian_state(
            "Enceladus",
            "Saturn"
        )
    ]

    propagator_settings = numerical_simulation.propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        Benedikter.K1_initial_cartesian_state,
        simulation_start_epoch,
        integrator_settings,
        termination_settings,
        output_variables=dependent_variables_to_save,
    )

    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)
    dependent_variable_history = dynamics_simulator.propagation_results.dependent_variable_history

    saturn_spherical_coordinates_history = dict()
    enceladus_keplerian_state_history = dict()
    for epoch in list(dependent_variable_history.keys()):
        saturn_spherical_coordinates_history[epoch] = dependent_variable_history[epoch][:3]
        enceladus_keplerian_state_history[epoch] = dependent_variable_history[epoch][3:]

    return saturn_spherical_coordinates_history, enceladus_keplerian_state_history


def main():
    #######################################################################################################################
    ### Input parameters ##################################################################################################
    #######################################################################################################################

    enceladus_reference_orbital_period = 1.370218 * constants.JULIAN_DAY
    nb_orbits = 60

    arc_start = 0
    arc_duration = nb_orbits * enceladus_reference_orbital_period
    arc_end = arc_start + arc_duration

    # Retrieve Enceladus-fixed spherical coordinates of Saturn
    saturn_spherical_coordinates_history, enceladus_keplerian_state_history = get_saturn_colatitude_and_longitude_history(arc_start,
                                                                                                                arc_end)

    # Gravitational parameter of Enceladus
    enceladus_gravitational_parameter = 7.210366688598896E+9  # From Park et al., 2024

    # Gravitational parameter of Saturn
    saturn_gravitational_parameter = 3.793120749865224E+16  # From Iess et al., 2019

    # Degree of the Love number to study
    degrees_to_consider = [2]

    # Set output directory
    output_directory = "./output/tidal_forcing_analysis"

    save2txt(saturn_spherical_coordinates_history,
             "saturn_spherical_coordinates_history.dat",
             output_directory)

    save2txt(enceladus_keplerian_state_history,
             "enceladus_keplerian_state_history.dat",
             output_directory)

    ###################################################################################################################
    ### Compute tidal forcing history #################################################################################
    ###################################################################################################################

    # Retrieve epochs at which the dependent variables are saved
    epochs = list(saturn_spherical_coordinates_history.keys())

    # Retrieve average radius of Enceladus
    enceladus_average_radius = spice.get_average_radius("Enceladus")

    # Compute history of tidal forcing for every degree
    for degree in degrees_to_consider:

        # Create output directory
        degree_output_path = os.path.join(output_directory, f"degree_{degree}")
        os.makedirs(degree_output_path, exist_ok=True)

        for order in range(degree + 1):
            output_path = os.path.join(degree_output_path, f"order_{order}")

            tidal_forcing = dict()
            for epoch in epochs:

                # Retrieve the Enceladus-fixed latitude and longitude of Saturn
                latitude, longitude = saturn_spherical_coordinates_history[epoch][1], saturn_spherical_coordinates_history[epoch][2]
                colatitude = np.pi/2 - latitude

                # Retrieve distance between Saturn and Enceladus
                distance = saturn_spherical_coordinates_history[epoch][0]

                # Compute normalised Legendre function
                arg = np.cos(colatitude)
                normalised_legendre_function = get_normalised_legendre_function(degree, order, arg)

                # Compute tidal forcing and tidal response at current epoch
                tidal_forcing[epoch] = []

                tidal_forcing[epoch] = []
                tidal_forcing_cosine = (( 1 / (2 * degree + 1) ) *
                                        ( saturn_gravitational_parameter / enceladus_gravitational_parameter ) *
                                        (( enceladus_average_radius / distance ) ** (degree + 1)) *
                                        normalised_legendre_function * np.cos(order * longitude))
                tidal_forcing[epoch].append( tidal_forcing_cosine )

                tidal_forcing_sine = (( 1 / (2 * degree + 1) ) *
                                        ( saturn_gravitational_parameter / enceladus_gravitational_parameter ) *
                                        ( enceladus_average_radius / distance ) ** (degree + 1) *
                                        normalised_legendre_function * np.sin(order * longitude))
                tidal_forcing[epoch].append( tidal_forcing_sine )

            mean_tidal_forcing_cosine = np.mean(result2array(tidal_forcing)[:, 1])
            mean_tidal_forcing_sine = np.mean(result2array(tidal_forcing)[:, 2])

            # Save tidal forcing and variables history to file
            save2txt(tidal_forcing,
                     "tidal_forcing_history.dat",
                     output_path)
            np.savetxt(os.path.join(output_path, "mean_tidal_forcing_cosine.txt"), [mean_tidal_forcing_cosine])
            np.savetxt(os.path.join(output_path, "mean_tidal_forcing_sine.txt"), [mean_tidal_forcing_sine])



if __name__ == "__main__":
    main()
