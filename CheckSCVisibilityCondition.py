# Tudat import
from matplotlib import pyplot as plt
from tudatpy.math import interpolators
from tudatpy import constants

# Files import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary.utilities import utilities

# General imports
import numpy as np
import os

# Define input path
input_directory = "./output/covariance_analysis/single_case_analysis"
time_stamp_folder = "2025.09.09.16.00.58"
input_path = os.path.join(input_directory, time_stamp_folder)
simulation_results_path = os.path.join(input_path, "simulation_results")
observation_times_path = os.path.join(input_path, "observation_times")
plots_path = os.path.join(input_path, "plots")
os.makedirs(plots_path, exist_ok=True)

fontsize=12

# Plot elevation angle history
fig, axes = plt.subplots(5, 2, figsize=(8, 10))
for lander in CovAnalysisConfig.lander_names:
    target_angles_and_range = np.loadtxt(os.path.join(simulation_results_path, f"target_angles_and_range_{lander}.dat"))
    epochs = target_angles_and_range[:, 0]
    elevation_angle = target_angles_and_range[:, 1]

    if lander == CovAnalysisConfig.lander_names[0]:
        ax = axes[0, 0]
    elif lander == CovAnalysisConfig.lander_names[1]:
        ax = axes[0, 1]
    elif lander == CovAnalysisConfig.lander_names[2]:
        ax = axes[1, 0]
    elif lander == CovAnalysisConfig.lander_names[3]:
        ax = axes[1, 1]
    elif lander == CovAnalysisConfig.lander_names[4]:
        ax = axes[2, 0]
    elif lander == CovAnalysisConfig.lander_names[5]:
        ax = axes[2, 1]
    elif lander == CovAnalysisConfig.lander_names[6]:
        ax = axes[3, 0]
    elif lander == CovAnalysisConfig.lander_names[7]:
        ax = axes[3, 1]
    elif lander == CovAnalysisConfig.lander_names[8]:
        ax = axes[4, 0]
    ax.scatter(epochs / constants.JULIAN_DAY, np.rad2deg(elevation_angle), marker=".", color="black")
    ax.axhline(y=np.rad2deg(CovAnalysisConfig.minimum_elevation_angle_visibility),
               color="red")
    ax.set_xlabel(r"$t - t_0$  [days]", fontsize=fontsize)
    ax.set_ylabel(r"$\delta$  [deg]", fontsize=fontsize)
    ax.grid(True)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(f"Lander: {lander}", fontsize=fontsize)
plt.delaxes(axes[4, 1])
fig.savefig(os.path.join(plots_path, f"elevation_angle_analysis.pdf"))
plt.close(fig)

# Load dependent variables history
nb_arcs = np.loadtxt(os.path.join(simulation_results_path, "nb_arcs.dat"))
nb_history_epochs = 0
for i in range(int(nb_arcs)):
    dependent_variables_history_arc = np.loadtxt(os.path.join(simulation_results_path, f"dependent_variable_history_arc_{i}.dat"))
    nb_history_epochs += dependent_variables_history_arc.shape[0]
dependent_variable_history = np.zeros((nb_history_epochs, 4))
counter = 0
for i in range(int(nb_arcs)):
    dependent_variables_history_arc = np.loadtxt(
        os.path.join(simulation_results_path, f"dependent_variable_history_arc_{i}.dat"))
    dependent_variable_history[counter:counter+dependent_variables_history_arc.shape[0], :] = dependent_variables_history_arc[
                                                                                              :, :CovAnalysisConfig.indices_dependent_variables["longitude"][1]
                                                                                              ]
    counter += dependent_variables_history_arc.shape[0]
for j in range(dependent_variable_history.shape[0]):
    if dependent_variable_history[j, 3] < 0:
        dependent_variable_history[j, 3] = 2*np.pi + dependent_variable_history[j, 3]
dependent_variable_history_dict = utilities.array2dict(dependent_variable_history)

lagrange_interpolation_settings = interpolators.lagrange_interpolation(number_of_points=CovAnalysisConfig.number_of_points)
interpolator = interpolators.create_one_dimensional_vector_interpolator(dependent_variable_history_dict, lagrange_interpolation_settings)

# Plot coordinates of spacecraft at observation epochs
for lander in CovAnalysisConfig.lander_names:
    observation_times = np.loadtxt(os.path.join(observation_times_path, f"observation_times_{lander}.dat"))
    nb_observations = len(observation_times)
    coordinates_store = np.zeros((len(observation_times), 3))
    lander_coordinates = CovAnalysisConfig.lander_coordinates[lander]
    for i in range(len(observation_times)):
        epoch = observation_times[i]
        state = interpolator.interpolate(epoch)
        coordinates_store[i, :] = state
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(np.rad2deg(dependent_variable_history[:, 3]), np.rad2deg(dependent_variable_history[:, 2]), color="blue", marker=".")
    ax.scatter(np.rad2deg(coordinates_store[:, 2]), np.rad2deg(coordinates_store[:, 1]), color="orange", marker=".")
    ax.scatter(np.rad2deg(lander_coordinates[2]), np.rad2deg(lander_coordinates[1]), color="red")
    ax.grid(True)
    ax.set_ylim(bottom=-90, top=90)
    ax.set_xticks(np.arange(0, 361, 40))
    ax.set_xlabel("Longitude  [deg]", fontsize=fontsize)
    ax.set_ylabel("Latitude  [deg]", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(f"Lander: {lander}. Nb. observations: {nb_observations}", fontsize=fontsize)
    fig.savefig(os.path.join(plots_path, f"sc_location_at_observation_{lander}.pdf"))
    plt.close(fig)
