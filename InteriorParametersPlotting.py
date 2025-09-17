# General imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from multiprocessing import Process

# Files import
from auxiliary.utilities import InteriorParametersPlottingUtilities as Util


def plot_one_at_a_time_interior_parameters_analysis(input_path, fontsize=12):

    layers = ["core", "ocean", "shell"]
    interior_parameters = ["R0", "rho0", "mu0", "eta0", "Ks0"]
    interior_parameters_labels = [r"$R_{c}$  [km]", r"$\rho_{c}$  [kg m$^{-3}$]", r"$\mu_{c}$  [Pa]", r"$\eta_{c}$  [Pa s]", r"$K_{c}$  [Pa]",
                                  r"$R_{o}$  [km]", r"$\rho_{o}$  [kg m$^{-3}$]", r"$\mu_{o}$  [Pa]", r"$\eta_{o}$  [Pa s]", r"$K_{o}$  [Pa]",
                                  r"$R_{s}$  [km]", r"$\rho_{s}$  [kg m$^{-3}$]", r"$\mu_{s}$  [Pa]", r"$\eta_{s}$  [Pa s]", r"$K_{s}$  [Pa]"]

    k2_handle = mlines.Line2D([],
                              [],
                              color="blue",
                              linestyle="-",
                              marker=" ",
                              label=r"$k_2$")
    h2_handle = mlines.Line2D([],
                              [],
                              color="red",
                              linestyle="-",
                              marker=" ",
                              label=r"$h_2$")

    # Plot Love numbers
    for layer in layers:
        layer_input_path = os.path.join(input_path, layer)
        layer_index = layers.index(layer)

        if layer == "ocean":
            interior_parameters_layer = interior_parameters[:2]
            fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
        else:
            interior_parameters_layer = interior_parameters
            fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 8))
        for interior_parameter in interior_parameters_layer:
            interior_parameter_path = os.path.join(layer_input_path, f"{interior_parameter}.dat")
            interior_parameter_index = interior_parameters.index(interior_parameter)

            # Load results
            results = np.loadtxt(interior_parameter_path, delimiter=",")
            interior_parameter_values = results[:, 0]
            k2_love_number_values = results[:, 1]
            h2_love_number_values = results[:, 2]

            if layer == "ocean":
                if interior_parameter_index == 0:
                    ax = axes[0]
                elif interior_parameter_index == 1:
                    ax = axes[1]
            else:
                if interior_parameter_index == 0:
                    ax = axes[0, 0]
                elif interior_parameter_index == 1:
                    ax = axes[0, 1]
                elif interior_parameter_index == 2:
                    ax = axes[1, 0]
                elif interior_parameter_index == 3:
                    ax = axes[1, 1]
                elif interior_parameter_index == 4:
                    ax = axes[2, 0]

            ax.plot(interior_parameter_values, k2_love_number_values.real, label=r"$K_2$", color="blue")
            ax.plot(interior_parameter_values, h2_love_number_values.real, label=r"$h_2$", color="red")
            ax.set_xlabel(interior_parameters_labels[len(interior_parameters)*layer_index + interior_parameter_index], fontsize=fontsize)

            if interior_parameter_index % 2 == 0:
                ax.set_ylabel("Love number  [-]", fontsize=fontsize)

        fig.legend(handles=[k2_handle, h2_handle], fontsize=fontsize, bbox_to_anchor=(0.85, 0.2))

        for ax in axes.flat:
            ax.tick_params(labelsize=fontsize)
            ax.grid(True)
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-2, top=5e-2)
        if layer != "ocean":
            plt.delaxes(axes[2, 1])

        fig.suptitle(r"Measurements: $k_2$ & $h_2$ Love numbers. Layer: " + layer, fontsize=fontsize)
        fig.savefig(os.path.join(layer_input_path, f"tidal_love_numbers_{layer}.pdf"))
        plt.close(fig)

    # Plot libration amplitude
    for layer in layers:
        layer_input_path = os.path.join(input_path, layer)
        layer_index = layers.index(layer)

        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 8))
        for interior_parameter in interior_parameters:
            interior_parameter_path = os.path.join(layer_input_path, f"{interior_parameter}.dat")
            interior_parameter_index = interior_parameters.index(interior_parameter)

            # Load results
            results = np.loadtxt(interior_parameter_path, delimiter=",")
            interior_parameter_values = results[:, 0]
            libration_amplitude = np.rad2deg(results[:, 3])

            if interior_parameter_index == 0:
                ax = axes[0, 0]
            elif interior_parameter_index == 1:
                ax = axes[0, 1]
            elif interior_parameter_index == 2:
                ax = axes[1, 0]
            elif interior_parameter_index == 3:
                ax = axes[1, 1]
            elif interior_parameter_index == 4:
                ax = axes[2, 0]

            ax.plot(interior_parameter_values, np.abs(libration_amplitude), color="blue")
            ax.set_xlabel(interior_parameters_labels[len(interior_parameters)*layer_index + interior_parameter_index], fontsize=fontsize)
            if interior_parameter_index % 2 == 0:
                ax.set_ylabel(r"$\phi$  [deg]", fontsize=fontsize)
        for ax in axes.flat:
            ax.tick_params(labelsize=fontsize)
            ax.grid(True)
            # ax.set_yscale("log")
        plt.delaxes(axes[2, 1])

        fig.suptitle(r"Measurement: libration amplitude. Layer: " + layer, fontsize=fontsize)
        fig.savefig(os.path.join(layer_input_path, f"libration_amplitude_{layer}.pdf"))
        plt.close(fig)


def plot_monte_carlo_interior_parameters_analysis_simple_plot(input_path,
                                                              filter_libration_amplitude_flag,
                                                              filter_tidal_heating_flag,
                                                              nominal_libration_amplitude,
                                                              tidal_heating_range,
                                                              fontsize=12):

    plots_path = os.path.join(input_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    layers = ["core", "ocean", "shell"]

    interior_parameters_labels = [r"$R_{c}$  [km]", r"$\rho_{c}$  [kg m$^{-3}$]", r"$\mu_{c}$  [Pa]", r"$\eta_{c}$  [Pa s]", r"$K_{c}$  [Pa]",
                                  r"$d_{o}$  [km]", r"$\rho_{o}$  [kg m$^{-3}$]", r"$\mu_{o}$  [Pa]", r"$\eta_{o}$  [Pa s]", r"$K_{o}$  [Pa]",
                                  r"$d_{s}$  [km]", r"$\rho_{s}$  [kg m$^{-3}$]", r"$\mu_{s}$  [Pa]", r"$\eta_{s}$  [Pa s]", r"$K_{s}$  [Pa]"]

    # Load results
    observations = np.loadtxt(os.path.join(input_path, "observations.dat"), delimiter=",")
    interior_models = np.loadtxt(os.path.join(input_path, "interior_models.dat"), delimiter=",")

    # Convert libration amplitude to deg
    observations[:, 0] = np.rad2deg(observations[:, 0])

    # Filter parameter and observations based on the feasibility of the interior model
    try:
        filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"))
        filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_feasibility.dat"))
    except:
        filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(
            interior_models, observations)
        np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
        np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
        np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    # Filter data based on libration amplitude and tidal heating observations
    if filter_libration_amplitude_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"))
            nb_feasible_libration_models = np.loadtxt(os.path.join(input_path, "nb_feasible_libration_models.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_libration_models = Util.filter_libration_amplitude(
                filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"), filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"), filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_libration_models.dat"), [nb_feasible_libration_models])

    if filter_tidal_heating_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_tidal_heating_models = Util.filter_tidal_heating(
                filtered_parameters_feasibility, filtered_observations_feasibility, tidal_heating_range
            )
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"), filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"), filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_tidal_heating_models.dat"), [nb_feasible_tidal_heating_models])

    for i in range(len(layers)):
        layer = layers[i]
        if layer == "ocean":
            parameters = filtered_parameters_feasibility[:, 5 * i:5 * i + 2]
            #print(min(parameters[:, 0]), max(parameters[:, 0]))
            #print(min(parameters[:, 0] - filtered_parameters_feasibility[:, 0]),
            #      max(parameters[:, 0] - filtered_parameters_feasibility[:, 0]))
            fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
            fig2, axes2 = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
            fig3, axes3 = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
        else:
            parameters = filtered_parameters_feasibility[:, 5 * i:5 * i + 5]
            fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
            fig2, axes2 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
            fig3, axes3 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
        for j in range(parameters.shape[1]):

            if layer == "ocean":
                if j==0:
                    ax = axes[0]
                    ax2 = axes2[0]
                    ax3 = axes3[0]
                elif j==1:
                    ax = axes[1]
                    ax2 = axes2[1]
                    ax3 = axes3[1]
            else:
                if j == 0:
                    ax = axes[0, 0]
                    ax2 = axes2[0, 0]
                    ax3 = axes3[0, 0]
                elif j == 1:
                    ax = axes[0, 1]
                    ax2 = axes2[0, 1]
                    ax3 = axes3[0, 1]
                elif j == 2:
                    ax = axes[1, 0]
                    ax2 = axes2[1, 0]
                    ax3 = axes3[1, 0]
                elif j == 3:
                    ax = axes[1, 1]
                    ax2 = axes2[1, 1]
                    ax3 = axes3[1, 1]
                elif j == 4:
                    ax = axes[2, 0]
                    ax2 = axes2[2, 0]
                    ax3 = axes3[2, 0]

            if j == 0:
                if layer == "ocean":
                    ax.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 0], filtered_observations_feasibility[:, 0], color="black")
                    ax2.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 0], filtered_observations_feasibility[:, 1], color="black")
                    ax3.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 0], filtered_observations_feasibility[:, 2], color="black")
                elif layer == "shell":
                    ax.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 5], filtered_observations_feasibility[:, 0], color="black")
                    ax2.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 5], filtered_observations_feasibility[:, 1], color="black")
                    ax3.scatter(parameters[:, j] - filtered_parameters_feasibility[:, 5], filtered_observations_feasibility[:, 2], color="black")
                else:
                    ax.scatter(parameters[:, j], filtered_observations_feasibility[:, 0], color="black")
                    ax2.scatter(parameters[:, j], filtered_observations_feasibility[:, 1], color="black")
                    ax3.scatter(parameters[:, j], filtered_observations_feasibility[:, 2], color="black")
            else:
                ax.scatter(parameters[:, j], filtered_observations_feasibility[:, 0], color="black")
                ax2.scatter(parameters[:, j], filtered_observations_feasibility[:, 1], color="black")
                ax3.scatter(parameters[:, j], filtered_observations_feasibility[:, 2], color="black")

            ax.set_xlabel(interior_parameters_labels[5*i + j], fontsize=fontsize)
            ax2.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)
            ax3.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)
            if j % 2 == 0:
                ax.set_ylabel(r"$W_s$  [deg]", fontsize=fontsize)
                ax2.set_ylabel(r"$k_2$  [-]", fontsize=fontsize)
                ax3.set_ylabel(r"$h_2$  [-]", fontsize=fontsize)
            if j > 1:
                ax.set_xscale("log")
                ax2.set_xscale("log")
                ax3.set_xscale("log")
            ax.grid(True)
            ax2.grid(True)
            ax3.grid(True)
            ax2.set_yscale("log")
            ax3.set_yscale("log")
        if layer != "ocean":
            fig.delaxes(axes[2, 1])
            fig2.delaxes(axes2[2, 1])
            fig3.delaxes(axes3[2, 1])
        fig.suptitle(f"Measurements: libration amplitude. Layer: {layer}. Nb. samples: {nb_feasible_libration_models}", fontsize=fontsize)
        fig2.suptitle(f"Measurements: k2 Love number. Layer: {layer}. Nb. samples: {nb_feasible_libration_models}", fontsize=fontsize)
        fig3.suptitle(f"Measurements: h2 Love number. Layer: {layer}. Nb. samples: {nb_feasible_libration_models}", fontsize=fontsize)
        if filter_libration_amplitude_flag:
            fig.savefig(os.path.join(plots_path, f"observations_trends_libration_amplitude_{layer}_observations_filtered.png"))
            fig2.savefig(os.path.join(plots_path, f"observations_trends_k2_Love_number_{layer}_observations_filtered.png"))
            fig3.savefig(os.path.join(plots_path, f"observations_trends_h2_Love_number_{layer}_observations_filtered.png"))
        else:
            fig.savefig(os.path.join(plots_path, f"observations_trends_libration_amplitude_{layer}.pdf"))
            fig2.savefig(os.path.join(plots_path, f"observations_trends_k2_Love_number_{layer}.pdf"))
            fig3.savefig(os.path.join(plots_path, f"observations_trends_h2_Love_number_{layer}.pdf"))


def plot_monte_carlo_interior_parameters_analysis_density_plot(input_path,
                                                               parameters_grid_step,
                                                               observable_grid_step,
                                                               filter_libration_amplitude_flag,
                                                               filter_tidal_heating_flag,
                                                               nominal_libration_amplitude,
                                                               tidal_heating_range,
                                                               parameters_intervals,
                                                               fontsize=12):

    plots_path = os.path.join(input_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    layers = ["core", "ocean", "shell"]

    interior_parameters_labels = [r"$R_{c}$  [km]", r"$\rho_{c}$  [kg m$^{-3}$]", r"$\mu_{c}$  [Pa]", r"$\eta_{c}$  [Pa s]", r"$K_{c}$  [Pa]",
                                  r"$d_{o}$  [km]", r"$\rho_{o}$  [kg m$^{-3}$]", r"$\mu_{o}$  [Pa]", r"$\eta_{o}$  [Pa s]", r"$K_{o}$  [Pa]",
                                  r"$d_{s}$  [km]", r"$\rho_{s}$  [kg m$^{-3}$]", r"$\mu_{s}$  [Pa]", r"$\eta_{s}$  [Pa s]", r"$K_{s}$  [Pa]"]

    # Load results
    observations = np.loadtxt(os.path.join(input_path, "observations.dat"), delimiter=",")
    interior_models = np.loadtxt(os.path.join(input_path, "interior_models.dat"), delimiter=",")

    # Convert libration amplitude to deg
    observations[:, 0] = np.rad2deg(observations[:, 0])

    # Filter parameter and observations based on the feasibility of the interior model
    try:
        filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"))
        filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_feasibility.dat"))
    except:
        filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(interior_models, observations)
        np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
        np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
        np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    if filter_libration_amplitude_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_libration_models = Util.filter_libration_amplitude(
                filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"),
                       filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"),
                       filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_libration_models.dat"), [nb_feasible_libration_models])

    if filter_tidal_heating_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_tidal_heating_models = Util.filter_tidal_heating(
                filtered_parameters_feasibility, filtered_observations_feasibility, tidal_heating_range
            )
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"),
                       filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"),
                       filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_tidal_heating_models.dat"),
                       [nb_feasible_tidal_heating_models])

    nb_observables = filtered_observations_feasibility.shape[1]

    # Compute ocean and shell thickness
    filtered_parameters_feasibility[:, 5] = filtered_parameters_feasibility[:, 5] - filtered_parameters_feasibility[:, 0]
    filtered_parameters_feasibility[:, 10] = filtered_parameters_feasibility[:, 10] - filtered_parameters_feasibility[:, 5] - filtered_parameters_feasibility[:, 0]

    # Produce density plots
    for i in range(len(layers)):
        layer = layers[i]
        if layer == "ocean":
            parameters = filtered_parameters_feasibility[:, 5 * i:5*i+2]
        else:
            parameters = filtered_parameters_feasibility[:, 5 * i:5 * i + 5]

        nb_interior_parameters = parameters.shape[1]

        if layer == "ocean":
            fig1, axes1 = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
            fig2, axes2 = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
            fig3, axes3 = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
        else:
            fig1, axes1 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 12))
            fig2, axes2 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 12))
            fig3, axes3 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 12))


        for k in range(nb_observables):

            if k == 0:
                obs_label = "phi"
                fig = fig1
                axes = axes1
            elif k == 1:
                obs_label = "k2"
                fig = fig2
                axes = axes2
            elif k == 2:
                obs_label = "h2"
                fig = fig3
                axes = axes3
            else:
                continue

            for j in range(nb_interior_parameters):

                if layer == "ocean":
                    if j == 0:
                        ax = axes[0]
                    elif j == 1:
                        ax = axes[1]
                else:
                    if j == 0:
                        ax = axes[0, 0]
                    elif j == 1:
                        ax = axes[0, 1]
                    elif j == 2:
                        ax = axes[1, 0]
                    elif j == 3:
                        ax = axes[1, 1]
                    elif j == 4:
                        ax = axes[2, 0]

                parameter_grid_vec, observable_grid_vec = Util.get_parameter_grid_vec(
                    j,
                    filtered_observations_feasibility[:, k],
                    parameters_grid_step,
                    parameters_intervals,
                    observable_grid_step[obs_label],
                    layer,
                    obs_label
                )

                # Make density plot
                h = ax.hist2d(parameters[:, j], filtered_observations_feasibility[:, k],
                          [parameter_grid_vec, observable_grid_vec])
                c = fig.colorbar(h[3], ax=ax)
                c.set_label("Count  [-]")

                # Set 0 level contour line
                ax.contour([parameters[:, j], filtered_observations_feasibility[:, k]], h[3], [0], colors=["red"])
                ax.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)

                if j % 2 == 0:
                    if k == 0:
                        ax.set_ylabel(r"$W_s$  [deg]", fontsize=fontsize)
                    elif k == 1:
                        ax.set_ylabel(r"$k_2$  [-]", fontsize=fontsize)
                    elif k == 2:
                        ax.set_ylabel(r"$h_2$  [-]", fontsize=fontsize)

                if j == 2 or j == 3 or j == 4:
                    ax.set_xscale("log")

        fig1.suptitle(f"Measurements: libration amplitude. Layer: {layer}", fontsize=fontsize)
        fig2.suptitle(f"Measurements: k2 Love number. Layer: {layer}", fontsize=fontsize)
        fig3.suptitle(f"Measurements: h2 Love number. Layer: {layer}", fontsize=fontsize)
        fig1.savefig(os.path.join(plots_path, f"density_plot_libration_amplitude_{layer}.pdf"))
        fig2.savefig(os.path.join(plots_path, f"density_plot_k2_Love_number_{layer}.pdf"))
        fig3.savefig(os.path.join(plots_path, f"density_plot_h2_Love_number_{layer}.pdf"))
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


def plot_monte_carlo_analysis_observables_histogram(input_path,
                                                    observables_to_study,
                                                    grid_steps,
                                                    filter_libration_amplitude_flag,
                                                    filter_tidal_heating_range_flag,
                                                    nominal_libration_amplitude,
                                                    tidal_heating_range,
                                                    fontsize=12):
    plots_path = os.path.join(input_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    # Load results
    observations = np.loadtxt(os.path.join(input_path, "observations.dat"), delimiter=",")
    interior_models = np.loadtxt(os.path.join(input_path, "interior_models.dat"), delimiter=",")

    # Convert libration amplitude to deg
    observations[:, 0] = np.rad2deg(observations[:, 0])

    # Filter parameter and observations based on the feasibility of the interior model
    try:
        filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"))
        filtered_observations_feasibility = np.loadtxt(
            os.path.join(input_path, "filtered_observations_feasibility.dat"))
    except:
        filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(
            interior_models, observations)
        np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
        np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
        np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    # Filter parameters and observations based on libration amplitude and tidal heating observations
    if filter_libration_amplitude_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_libration_models = Util.filter_libration_amplitude(
                filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"),
                       filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"),
                       filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_libration_models.dat"), [nb_feasible_libration_models])

    if filter_tidal_heating_range_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_tidal_heating_models = Util.filter_tidal_heating(
                filtered_parameters_feasibility, filtered_observations_feasibility, tidal_heating_range
            )
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"),
                       filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"),
                       filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_tidal_heating_models.dat"),
                       [nb_feasible_tidal_heating_models])

    # Set axis label of observables
    observables_label = [r"$\phi$", r"$Re(k_2)$", r"$Re(h_2)$", r"$\dot{E}$", r"$Re(k_2)$", r"$Im(h_2)$"]

    fig, axes = plt.subplots(2, 2, costrained_layout=True, figsize=(8, 8))
    for i in range(len(observables_to_study)):
        observable_label = observables_to_study[i]
        if observable_label == "libration":
            obs_index = 0
            ax = axes[0, 0]
        elif observable_label == "k2_real":
            obs_index = 1
            ax = axes[0, 1]
        elif observable_label == "h2_real":
            obs_index = 2
            ax = axes[1, 0]
        else:
            raise ValueError("Observable not recognized.")

        mean = np.mean(filtered_observations_feasibility[:, obs_index])
        std = np.std(filtered_observations_feasibility[:, obs_index])

        grid = np.arange(min(filtered_observations_feasibility[:, obs_index]), max(filtered_observations_feasibility[:, obs_index]), grid_steps[i])

        h = ax.hist(filtered_observations_feasibility[:, obs_index], grid, histtype="step", color="black")
        ax.set_xlabel(observables_label[obs_index], fontsize=fontsize)
        ax.set_ylabel("Count  [-]", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

        ax.axvline(x=mean, color="red")
        ax.set_title(f"Mean: {str(mean)[:5]}. Std. dev.: {str(std)[:5]}. "
                     f"Nb. samples: {filtered_observations_feasibility.shape[0]}", fontsize=fontsize)

    fig.savefig(os.path.join(plots_path, f"observables_histogram.pdf"))
    plt.close(fig)


def plot_histogram_filtered_parameters_from_observables(input_path,
                                                        filter_libration_amplitude_flag,
                                                        filter_tidal_heating_range_flag,
                                                        observables_to_study: dict,
                                                        parameters_to_study,
                                                        parameters_grid_steps,
                                                        nominal_libration_amplitude,
                                                        tidal_heating_range,
                                                        file_ticket,
                                                        fontsize=12):
    plots_path = os.path.join(input_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    # Load results
    observations = np.loadtxt(os.path.join(input_path, "observations.dat"), delimiter=",")
    interior_models = np.loadtxt(os.path.join(input_path, "interior_models.dat"), delimiter=",")

    # Convert libration amplitude to deg
    observations[:, 0] = np.rad2deg(observations[:, 0])

    # Filter parameters and observations based on the feasibility of the interior model
    try:
        filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"))
        filtered_observations_feasibility = np.loadtxt(
            os.path.join(input_path, "filtered_observations_feasibility.dat"))
    except:
        filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(
            interior_models, observations)
        np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
        np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
        np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    # Filter parameters and observations based on libration amplitude and tidal heating observations
    if filter_libration_amplitude_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_libration_models = Util.filter_libration_amplitude(
                filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_libration_measurements.dat"),
                       filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_libration_measurements.dat"),
                       filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_libration_models.dat"), [nb_feasible_libration_models])

    if filter_tidal_heating_range_flag:
        try:
            filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"))
            filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"))
        except:
            filtered_parameters, filtered_observations, nb_feasible_tidal_heating_models = Util.filter_tidal_heating(
                filtered_parameters_feasibility, filtered_observations_feasibility, tidal_heating_range
            )
            filtered_parameters_feasibility = filtered_parameters
            filtered_observations_feasibility = filtered_observations
            np.savetxt(os.path.join(input_path, "filtered_parameters_tidal_heating.dat"),
                       filtered_parameters_feasibility)
            np.savetxt(os.path.join(input_path, "filtered_observations_tidal_heating.dat"),
                       filtered_observations_feasibility)
            np.savetxt(os.path.join(input_path, "nb_feasible_tidal_heating_models.dat"), [nb_feasible_tidal_heating_models])

    # Compute ocean and shell thickness
    filtered_parameters_feasibility[:, 5] = filtered_parameters_feasibility[:, 5] - filtered_parameters_feasibility[:, 0]
    filtered_parameters_feasibility[:, 10] = filtered_parameters_feasibility[:, 10] - filtered_parameters_feasibility[:, 5] - filtered_parameters_feasibility[:, 0]

    filtered_parameters = filtered_parameters_feasibility
    filtered_observations = filtered_observations_feasibility

    observables_to_study_labels = list(observables_to_study.keys())
    for current_observable in observables_to_study_labels:
        if current_observable == "libration":
            obs_index = 0
        elif current_observable == "k2_real":
            obs_index = 1
        elif current_observable == "h2_real":
            obs_index = 2
        elif current_observable == "k2_imag":
            obs_index = 3
        elif current_observable == "h2_imag":
            obs_index = 4
        else:
            raise ValueError("Unknown observable " + current_observable)

        filtered_parameters, filtered_observations, nb_viable_simulations = Util.filter_parameters_from_observations(
            filtered_parameters,
            filtered_observations,
            observables_to_study[current_observable],
            obs_index
        )

    for current_parameter in parameters_to_study:
        if current_parameter == "d_core":
            param_index = 0
            label = r"$R_{c}$  [km]"
        elif current_parameter == "rho_core":
            param_index = 1
            label = r"$\rho_{c}$  [kg m$^{-3}$]"
        elif current_parameter == "mu_core":
            param_index = 2
            label = r"$\mu_{c}$  [Pa]"
        elif current_parameter == "eta_core":
            param_index = 3
            label = r"$\eta_{c}$  [Pa s]"
        elif current_parameter == "k_core":
            param_index = 4
            label = r"$K_{c}$  [Pa]"
        elif current_parameter == "d_ocean":
            param_index = 5
            label = r"$d_{o}$  [km]"
        elif current_parameter == "rho_ocean":
            param_index = 6
            label = r"$\rho_{o}$  [kg m$^{-3}$]"
        elif current_parameter == "mu_ocean":
            param_index = 7
            label = r"$\mu_{o}$  [Pa]"
        elif current_parameter == "eta_ocean":
            param_index = 8
            label = r"$\eta_{o}$  [Pa s]"
        elif current_parameter == "k_ocean":
            param_index = 9
            label = r"$K_{o}$  [Pa]"
        elif current_parameter == "d_shell":
            param_index = 10
            label = r"$d_{s}$  [km]"
        elif current_parameter == "rho_shell":
            param_index = 11
            label = r"$\rho_{s}$  [kg m$^{-3}$]"
        elif current_parameter == "mu_shell":
            param_index = 12
            label = r"$\mu_{s}$  [Pa]"
        elif current_parameter == "eta_shell":
            param_index = 13
            label = r"$\eta_{s}$  [Pa s]"
        elif current_parameter == "k_shell":
            param_index = 14
            label = r"$K_{s}$  [Pa]"

        if current_parameter == "eta_shell" or current_parameter == "eta_core":
            mean_old = np.mean(np.log10(filtered_parameters_feasibility[:, param_index]))
            std_old = np.std(np.log10(filtered_parameters_feasibility[:, param_index]))
            grid_old = np.linspace(np.log10(min(filtered_parameters_feasibility[:, param_index])),
                                np.log10(max(filtered_parameters_feasibility[:, param_index])),
                                parameters_grid_steps[current_parameter][0])
        else:
            mean_old = np.mean(filtered_parameters_feasibility[:, param_index])
            std_old = np.std(filtered_parameters_feasibility[:, param_index])
            grid_old = np.arange(min(filtered_parameters_feasibility[:, param_index]),
                                max(filtered_parameters_feasibility[:, param_index]),
                                parameters_grid_steps[current_parameter][0])

        if current_parameter == "eta_shell" or current_parameter == "eta_core":
            mean_new = np.mean(np.log10(filtered_parameters[:, param_index]))
            std_new = np.std(np.log10(filtered_parameters[:, param_index]))
            grid_new = np.linspace(np.log10(min(filtered_parameters[:, param_index])),
                                   np.log10(max(filtered_parameters[:, param_index])),
                                   parameters_grid_steps[current_parameter][1])
        else:
            mean_new = np.mean(filtered_parameters[:, param_index])
            std_new = np.std(filtered_parameters[:, param_index])
            grid_new = np.arange(min(filtered_parameters[:, param_index]),
                                max(filtered_parameters[:, param_index]),
                                parameters_grid_steps[current_parameter][1]
                                 )

        np.savetxt(os.path.join(input_path, f"distribution_old_{current_parameter}.txt"), [mean_old, std_old])
        np.savetxt(os.path.join(input_path, f"distribution_new_{current_parameter}.txt"), [mean_new, std_new])

        # Plot histogram of parameter distribution before and after filtering the observables
        fig, axes = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)
        if current_parameter == "eta_shell" or current_parameter == "eta_core":
            axes[0].hist(np.log10(filtered_parameters_feasibility[:, param_index]), grid_old)
            axes[1].hist(np.log10(filtered_parameters[:, param_index]), grid_new)
        else:
            axes[0].hist(filtered_parameters_feasibility[:, param_index], grid_old)
            axes[1].hist(filtered_parameters[:, param_index], grid_new)

        axes[0].axvline(mean_old, color='red')
        axes[0].axvline(mean_old + std_old, color='red', linestyle='--')
        axes[0].axvline(mean_old - std_old, color='red', linestyle='--')
        axes[0].set_title(f"Before filtering. Mean: {mean_old:.2f}, Std: {std_old:.2f}")
        axes[1].axvline(mean_new, color='red')
        axes[1].axvline(mean_new + std_new, color='red', linestyle='--')
        axes[1].axvline(mean_new - std_new, color='red', linestyle='--')
        axes[1].set_title(f"After filtering. Mean: {mean_new:.2f}, Std: {std_new:.2f}")

        axes[0].set_xlabel(label, fontsize=fontsize)
        axes[1].set_xlabel(label, fontsize=fontsize)
        axes[0].set_ylabel("Count", fontsize=fontsize)
        axes[0].tick_params(labelsize=fontsize)
        axes[1].tick_params(labelsize=fontsize)

        # if current_parameter == "eta_shell" or current_parameter == "eta_core":
        #     axes[0].set_xscale("log")
        #     axes[1].set_xscale("log")

        fig.savefig(os.path.join(plots_path, f"filtered_parameters_histogram_{current_parameter}_{file_ticket}.pdf"))
        plt.close(fig)


def main():

    nominal_libration_amplitude = [0.120, 0.021]  # [deg] - Thomas et al. (2016)
    tidal_heating_range = [15, 40]  # [GW] - Bagheri et al. (2025), page 17

    plot_one_at_a_time_interior_parameters_analysis_flag = False
    if plot_one_at_a_time_interior_parameters_analysis_flag:
        input_path = "./output/interior_parameters_analysis/preliminary_sensitivity_analysis"
        plot_one_at_a_time_interior_parameters_analysis(input_path)

    plot_monte_carlo_interior_parameters_analysis_flag = True
    if plot_monte_carlo_interior_parameters_analysis_flag:
        filter_libration_amplitude_flag = True
        filter_tidal_heating_flag = False
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.09.15.11.21.02"
        input_path = os.path.join(input_path, time_stamp)

        fontsize=14
        #p = Process(target=plot_monte_carlo_interior_parameters_analysis_simple_plot,
        #            args=(input_path,
        #                  filter_libration_amplitude_flag,
        #                  filter_tidal_heating_flag,
        #                  nominal_libration_amplitude,
        #                  tidal_heating_range,
        #                  fontsize)
        #            )
        #p.start()
        #p.join()
        plot_monte_carlo_interior_parameters_analysis_simple_plot(input_path,
                                                                  filter_libration_amplitude_flag,
                                                                  filter_tidal_heating_flag,
                                                                  nominal_libration_amplitude,
                                                                  tidal_heating_range,
                                                                  fontsize=fontsize)


    plot_monte_carlo_interior_parameters_analysis_density_plot_flag = False
    if plot_monte_carlo_interior_parameters_analysis_density_plot_flag:
        filter_libration_amplitude_flag = True
        filter_tidal_heating_flag = False
        parameters_grid_step = dict()
        parameters_grid_step["core"] = dict(
            d = 5,
            rho = 20,
            mu = 19,
            eta = 14,
            K = 21,
        )
        parameters_grid_step["ocean"] = dict(
            d = 5,
            rho = 25,
        )
        parameters_grid_step["shell"] = dict(
            d = 2.5,
            rho = 25,
            mu = 6,
            eta = 16,
            K = 6,
        )
        observable_grid_step = dict(
            h2 = 0.01,
            k2 = 0.005,
            phi = 0.005,
        )
        parameters_intervals = dict()
        parameters_intervals["core"] = dict(
            d = [180, 210],
            rho = [2220, 2380],
            mu = [1e9, 8e10],
            eta = [1e15, 1e20],
            K = [1e9, 1e11]
        )
        parameters_intervals["ocean"] = dict(
            d = [1, 40],
            rho = [1000, 1300],
        )
        parameters_intervals["shell"] = dict(
            d = [0, 30],
            rho = [800, 1000],
            mu = [1e9, 5e9],
            eta = [1e14, 1e20],
            K = [1e9, 1e14]
        )
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.09.15.11.21.02"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_interior_parameters_analysis_density_plot(input_path,
                                                                   parameters_grid_step,
                                                                   observable_grid_step,
                                                                   filter_libration_amplitude_flag,
                                                                   filter_tidal_heating_flag,
                                                                   nominal_libration_amplitude,
                                                                   tidal_heating_range,
                                                                   parameters_intervals)

    plot_monte_carlo_analysis_filtered_observables_histogram_flag = False
    if plot_monte_carlo_analysis_filtered_observables_histogram_flag:
        filter_libration_amplitude_flag = True
        filter_tidal_heating_flag = False
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.09.15.11.21.02"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_analysis_observables_histogram(input_path,
                                                        ["k2_real", "libration", "h2_real"],
                                                        [0.001, 0.001, 0.005],
                                                        filter_libration_amplitude_flag,
                                                        filter_tidal_heating_flag,
                                                        nominal_libration_amplitude,
                                                        tidal_heating_range
                                                        )

    plot_histogram_filtered_parameters_from_observables_flag = False
    if plot_histogram_filtered_parameters_from_observables_flag:
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.09.15.11.21.02"
        input_path = os.path.join(input_path, time_stamp)

        filter_libration_amplitude_flag = True
        filter_tidal_heating_flag = False
        observables_to_study = dict(
            k2_real = 1e-4,
            # libration = 1e-4,
            h2_real = 7e-4,
        )

        file_ticket = "k2_h2_real"

        parameters_to_study = [
            "d_core",
            "rho_core",
            "d_ocean",
            "rho_ocean",
            "d_shell",
            "rho_shell",
            "eta_core",
            "eta_shell"
        ]
        parameters_grid_steps = dict(
            d_core = [0.5, 0.5],
            rho_core = [10, 10],
            d_ocean = [0.5, 0.5],
            rho_ocean = [10, 10],
            d_shell = [0.5, 0.5],
            rho_shell = [10, 10],
            eta_core = [6, 6],
            eta_shell = [6, 6],
        )

        plot_histogram_filtered_parameters_from_observables(
            input_path,
            filter_libration_amplitude_flag,
            filter_tidal_heating_flag,
            observables_to_study,
            parameters_to_study,
            parameters_grid_steps,
            nominal_libration_amplitude,
            tidal_heating_range,
            file_ticket)


if __name__ == "__main__":
    main()
