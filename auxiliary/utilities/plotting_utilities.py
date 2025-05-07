# Tudat import
from tudatpy.interface import spice
from tudatpy.util import result2array

# Packages import
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines

# Files and variables import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig
from auxiliary.CovarianceAnalysisConfig import initial_state_index


def plot_trajectory(state_history,
                    output_path,
                    filename,
                    orbit_ID,
                    color):
    state_history_array = result2array(state_history)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    Enceladus_radius = spice.get_average_radius("Enceladus") * 1e-3
    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = np.outer(np.cos(u), Enceladus_radius * np.sin(v))
    y = np.outer(np.sin(u), Enceladus_radius * np.sin(v))
    z = np.outer(np.ones(np.size(u)), Enceladus_radius * np.cos(v))
    ax.plot_surface(x, y, z, color='blue', alpha=0.5)

    Enceladus_patch = mlines.Line2D([], [], color='blue', label='Enceladus')
    Vehicle_patch = mlines.Line2D([], [], color=color, label='Vehicle')

    ax.plot(state_history_array[:, 1]*1e-3, state_history_array[:, 2]*1e-3, state_history_array[:, 3]*1e-3, color=color)
    ax.set_xlabel('x  [km]')
    ax.set_ylabel('y  [km]')
    ax.set_zlabel('z  [km]')
    #plt.tight_layout()
    plt.title(orbit_ID, pad=0.0)
    fig.legend(handles=[Enceladus_patch, Vehicle_patch])

    os.makedirs(output_path, exist_ok=True)
    file_output_path = os.path.join(output_path, filename)
    plt.savefig(file_output_path)
    plt.close()


def plot_ground_track(latitude_history,
                      longitude_history,
                      output_path,
                      filename,
                      orbit_ID,
                      color):
    # Resolve 2pi ambiguity in longitude
    nb_epochs = latitude_history.shape[0]
    for i in range(nb_epochs):
        if longitude_history[i, 1] < 0:
            longitude_history[i, 1] = longitude_history[i, 1] + 2.0 * np.pi

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(longitude_history[:, 1] * 180 / np.pi, latitude_history[:, 1] * 180 / np.pi, ".", color=color, markersize=2)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_xticks(np.arange(0, 361, 40))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_title(f"Ground Track over {orbit_ID}")
    ax.grid(True)

    os.makedirs(output_path, exist_ok=True)
    file_output_path = os.path.join(output_path, filename)
    plt.savefig(file_output_path)
    plt.close()


def plot_correlations(correlations,
                      output_path,
                      filename):
    plt.figure(figsize=(9, 6))
    plt.imshow(np.abs(correlations), aspect='auto', interpolation='none')
    plt.colorbar()
    plt.title("Correlations")
    plt.xlabel("Index - Estimated Parameter")
    plt.ylabel("Index - Estimated Parameter")
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    file_output_path = os.path.join(output_path, filename)
    plt.savefig(file_output_path)
    plt.close()


def plot_observation_times(arc_ID,
                           output_path,
                           filename,
                           doppler_obs_times_new_norcia_current_arc=None,
                           doppler_obs_times_cebreros_current_arc=None,
                           doppler_obs_times_malargue_current_arc=None,
                           ):
    plt.figure(figsize=(7.5, 6))
    if doppler_obs_times_new_norcia_current_arc is not None:
        plt.scatter(doppler_obs_times_new_norcia_current_arc, np.ones((len(doppler_obs_times_new_norcia_current_arc), 1)))
    if doppler_obs_times_cebreros_current_arc is not None:
        plt.scatter(doppler_obs_times_cebreros_current_arc,
                 2.0 * np.ones((len(doppler_obs_times_cebreros_current_arc), 1)))
    if doppler_obs_times_malargue_current_arc is not None:
        plt.scatter(doppler_obs_times_malargue_current_arc,
                 3.0 * np.ones((len(doppler_obs_times_malargue_current_arc), 1)))

    plt.xlabel('Observation times [h]')
    plt.ylabel('')
    plt.yticks([1, 2, 3], ['New Norcia', 'Cebreros', 'Malargue'])
    plt.ylim([0.5, 3.5])
    plt.title(f'Viable observations over {arc_ID}')
    plt.grid()

    os.makedirs(output_path, exist_ok=True)
    file_output_path = os.path.join(output_path, filename)
    plt.savefig(file_output_path)
    plt.close()


def plot_formal_errors(formal_errors, formal_errors_with_consider_parameters, output_path, filename):

    plt.semilogy(np.arange(1, len(formal_errors) + 1, 1), formal_errors, color="blue", label="Nominal errors")
    if formal_errors_with_consider_parameters is not None:
        plt.semilogy(np.arange(1, len(formal_errors) + 1, 1), formal_errors_with_consider_parameters, color="red",
                    label="Errors w/ consider parameters", linestyle="dashed")
    plt.title("Formal errors")
    plt.xlabel("Index - Estimated Parameter  [-]")
    plt.ylabel("Formal error  [respective IS unit]")
    plt.tight_layout()
    plt.grid()
    plt.legend()

    os.makedirs(output_path, exist_ok=True)
    file_output_path = os.path.join(output_path, filename)
    plt.savefig(file_output_path)
    plt.close()


def plot_lander_distribution(output_path,
                             fontsize=12):
    initial_state_index = [1, 2, 3]
    lander_name = CovAnalysisConfig.lander_names
    lander_coordinate = CovAnalysisConfig.lander_coordinates

    latitude = []
    longitude = []
    for lander in lander_name:
        latitude.append(np.rad2deg(lander_coordinate[lander][1]))
        longitude.append(np.rad2deg(lander_coordinate[lander][2]))

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(10, 10))
    for index in initial_state_index:
        nominal_orbit_dependent_variable_history = np.loadtxt(f"nominal_orbits/simulation_duration_2419200.0/nominal_dependent_variable_history_{index}.dat")
        dim = nominal_orbit_dependent_variable_history.shape
        longitude_history = np.zeros((dim[0], 2))
        latitude_history = np.zeros((dim[0], 2))
        longitude_history[:, 0] = nominal_orbit_dependent_variable_history[:, 0]
        latitude_history[:, 0] = nominal_orbit_dependent_variable_history[:, 0]
        longitude_history[:, 1] = nominal_orbit_dependent_variable_history[:,
                              CovAnalysisConfig.indices_dependent_variables["longitude"][0]]
        latitude_history[:, 1] = nominal_orbit_dependent_variable_history[:,
                             CovAnalysisConfig.indices_dependent_variables["latitude"][0]]

        # Resolve 2pi ambiguity in longitude
        nb_epochs = latitude_history.shape[0]
        for i in range(nb_epochs):
            if longitude_history[i, 1] < 0:
                longitude_history[i, 1] = longitude_history[i, 1] + 2.0 * np.pi

        if index == 1:
            ax = axes[0, 0]
            ax.set_ylabel("Latitude  [deg]", fontsize=fontsize)
        elif index == 2:
            ax = axes[0, 1]
            ax.set_xlabel("Longitude  [deg]", fontsize=fontsize)
        elif index == 3:
            ax = axes[1, 0]
            ax.set_xlabel("Longitude  [deg]", fontsize=fontsize)
            ax.set_ylabel("Latitude  [deg]", fontsize=fontsize)
        ax.plot(longitude_history[:, 1] * 180 / np.pi, latitude_history[:, 1] * 180 / np.pi, ".", color="blue",
            markersize=1, alpha=0.08)
        ax.scatter(longitude, latitude, color="red", label="Lander")
        ax.grid(True)
        ax.set_xticks(np.arange(0, 361, 40))
        ax.tick_params(labelsize=fontsize)
        ax.set_title(f"K{index} orbit", fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.set_ylim(bottom=-90, top=90)
    plt.delaxes(axes[1, 1])
    fig.suptitle("Radio beacons distribution", fontsize=fontsize)
    fig.savefig(os.path.join(output_path, f"radio_beacons_distribution.pdf"))
    plt.close(fig)

