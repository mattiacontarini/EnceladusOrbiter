# Files and variables import
from auxiliary import StableOrbitUtilities as Util
from auxiliary import BenedikterInitialStates as Benedikter

# Tudat import
from tudatpy.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy import constants

# Packages import
import numpy as np

# Select initial state
initial_cartesian_state = Benedikter.K1_initial_cartesian_state
enceladus_gravitational_parameter = spice_interface.get_body_gravitational_parameter("Enceladus")
initial_keplerian_state = element_conversion.cartesian_to_keplerian(initial_cartesian_state,
                                                                    enceladus_gravitational_parameter)

# Escape parameters
escape_eccentricity = 0.2
escape_altitude = 500e3

# Lower and upper bounds for control variables of optimizer
decision_variable_bounds = dict()
decision_variable_bounds["semimajor_axis"] = Util.get_semimajor_axis_bounds(
    original_semimajor_axis=initial_keplerian_state[0],
    semiamplitude_variation_interval=50e3)
decision_variable_bounds["eccentricity"] = [0, 0.1]
decision_variable_bounds["inclination"] = [np.deg2rad(56), np.deg2rad(58)]
decision_variable_bounds["argument_of_periapsis"] = [0, 2*np.pi]
decision_variable_bounds["longitude_of_ascending_node"] = [0, 2*np.pi]
decision_variable_bounds["true_anomaly"] = [0, 2*np.pi]
decision_variable_range = Util.retrieve_decision_variable_range(decision_variable_bounds)

# Simulation timeline
simulation_start_epoch = 0 * constants.JULIAN_YEAR  # From Benedikter et al. (2022)
arc_duration = 1 * constants.JULIAN_DAY  # From Benedikter et al. (2022)
simulation_end_epoch = simulation_start_epoch + arc_duration

# Stop altitude for orbit propagation
termination_altitude = 0.0

# Penalty coefficients for fitness value
penalty_coefficients = dict()
penalty_coefficients["crash_time"] = 10
penalty_coefficients["escape_eccentricity"] = 10
penalty_coefficients["escape_semimajor_axis"] = 1

# Optimizer seed
seed = 193

# Optimizer size
no_individuals_per_decision_variable = 11

# Optimizer convergence conditions
n_gens_termination = 5
max_n_gens = 30
fitness_change_termination_condition = 0.01
