# General imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 8))
        for interior_parameter in interior_parameters:
            interior_parameter_path = os.path.join(layer_input_path, f"{interior_parameter}.dat")
            interior_parameter_index = interior_parameters.index(interior_parameter)

            # Load results
            results = np.loadtxt(interior_parameter_path, delimiter=",")
            interior_parameter_values = results[:, 0]
            k2_love_number_values = results[:, 1]
            h2_love_number_values = results[:, 2]

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
        plt.delaxes(axes[2, 1])

        fig.suptitle(r"Measurements: $k_2$ & $h_2$ Love numbers. Layer: " + layer, fontsize=fontsize)
        fig.savefig(os.path.join(layer_input_path, "tidal_love_numbers.pdf"))
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

            ax.plot(interior_parameter_values, libration_amplitude, color="blue")
            ax.set_xlabel(interior_parameters_labels[len(interior_parameters)*layer_index + interior_parameter_index], fontsize=fontsize)
            if interior_parameter_index % 2 == 0:
                ax.set_ylabel(r"$\phi$  [deg]", fontsize=fontsize)
        for ax in axes.flat:
            ax.tick_params(labelsize=fontsize)
            ax.grid(True)
            # ax.set_yscale("log")
        plt.delaxes(axes[2, 1])

        fig.suptitle(r"Measurement: libration amplitude. Layer: " + layer, fontsize=fontsize)
        fig.savefig(os.path.join(layer_input_path, "libration_amplitude.pdf"))
        plt.close(fig)


def plot_monte_carlo_interior_parameters_analysis_simple_plot(input_path,
                                                              filter_parameters_flag,
                                                              filter_libration_amplitude_flag,
                                                              filter_tidal_heating_flag,
                                                              tidal_heating_range,
                                                              nominal_libration_amplitude,
                                                              fontsize=12):

    plots_path = os.path.join(input_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    layers = ["core", "ocean", "shell"]

    interior_parameters_labels = [r"$R_{c}$  [km]", r"$\rho_{c}$  [kg m$^{-3}$]", r"$\mu_{c}$  [Pa]", r"$\eta_{c}$  [Pa s]", r"$K_{c}$  [Pa]",
                                  r"$d_{o}$  [km]", r"$\rho_{o}$  [kg m$^{-3}$]", r"$\mu_{o}$  [Pa]", r"$\eta_{o}$  [Pa s]", r"$K_{o}$  [Pa]",
                                  r"$d_{s}$  [km]", r"$\rho_{s}$  [kg m$^{-3}$]", r"$\mu_{s}$  [Pa]", r"$\eta_{s}$  [Pa s]", r"$K_{s}$  [Pa]"]

    # Load results
    if filter_parameters_flag:
        observations = np.loadtxt(os.path.join(input_path, "observations.dat"), delimiter=",")
        interior_models = np.loadtxt(os.path.join(input_path, "interior_models.dat"), delimiter=",")

        # Convert libration amplitude to deg
        observations[:, 0] = np.rad2deg(observations[:, 0])

        # Filter parameter and observations based on the feasibility of the interior model
        filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(interior_models, observations)
        np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
        np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
        np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])
    else:
        filtered_parameters_feasibility = np.loadtxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"))
        filtered_observations_feasibility = np.loadtxt(os.path.join(input_path, "filtered_observations_feasibility.dat"))

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    # Filter data based on libration amplitude and tidal heating observations
    if filter_libration_amplitude_flag:
        filtered_parameters, filtered_observations, nb_feasible_libration_models = Util.filter_libration_amplitude(
            filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
        filtered_parameters_feasibility = filtered_parameters
        filtered_observations_feasibility = filtered_observations

    if filter_tidal_heating_flag:
        filtered_parameters, filtered_observations, nb_feasible_tidal_heating_models = Util.filter_tidal_heating(
            filtered_parameters_feasibility, filtered_observations_feasibility, tidal_heating_range
        )
        filtered_parameters_feasibility = filtered_parameters
        filtered_observations_feasibility = filtered_observations

    for i in range(len(layers)):
        layer = layers[i]
        parameters = filtered_parameters_feasibility[:, 5*i:5*i+5]
        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
        fig2, axes2 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
        fig3, axes3 = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 10))
        for j in range(parameters.shape[1]):

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
                ax.set_ylabel(r"$\phi$  [deg]", fontsize=fontsize)
                ax2.set_ylabel(r"$k_2$  [-]", fontsize=fontsize)
                ax3.set_ylabel(r"$h_2$  [-]", fontsize=fontsize)
            if j > 1:
                ax.set_xscale("log")
                ax2.set_xscale("log")
                ax3.set_xscale("log")
            ax.grid(True)
            ax2.grid(True)
            ax3.grid(True)
        fig.delaxes(axes[2, 1])
        fig2.delaxes(axes2[2, 1])
        fig3.delaxes(axes3[2, 1])
        fig.suptitle(f"Measurements: libration amplitude. Layer: {layer}", fontsize=fontsize)
        fig2.suptitle(f"Measurements: k2 Love number. Layer: {layer}", fontsize=fontsize)
        fig3.suptitle(f"Measurements: h2 Love number. Layer: {layer}", fontsize=fontsize)
        if filter_libration_amplitude_flag:
            fig.savefig(os.path.join(plots_path, f"observations_trends_libration_amplitude_{layer}_observations_filtered.pdf"))
            fig2.savefig(os.path.join(plots_path, f"observations_trends_k2_Love_number_{layer}_observations_filtered.pdf"))
            fig3.savefig(os.path.join(plots_path, f"observations_trends_h2_Love_number_{layer}_observations_filtered.pdf"))
        else:
            fig.savefig(os.path.join(plots_path, f"observations_trends_libration_amplitude_{layer}.pdf"))
            fig2.savefig(os.path.join(plots_path, f"observations_trends_k2_Love_number_{layer}.pdf"))
            fig3.savefig(os.path.join(plots_path, f"observations_trends_h2_Love_number_{layer}.pdf"))


def plot_monte_carlo_interior_parameters_analysis_density_plot(input_path,
                                                               parameters_grid_step,
                                                               observable_grid_step,
                                                               filter_observations_flag,
                                                               tidal_heating_range,
                                                               nominal_libration_amplitude,
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
    filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(interior_models, observations)
    np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
    np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
    np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    if filter_observations_flag:
        filtered_parameters_libration, filtered_observations_libration, nb_feasible_models_libration = Util.filter_libration_amplitude(
            filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
        filtered_parameters_heating, filtered_observations_heating, nb_feasible_models_heating = Util.filter_tidal_heating(
            filtered_parameters_libration, filtered_observations_libration, tidal_heating_range
        )
        #filtered_parameters_feasibility = filtered_parameters_heating
        #filtered_observations_feasibility = filtered_observations_heating
        filtered_parameters_feasibility = filtered_parameters_libration
        filtered_observations_feasibility = filtered_observations_libration

    nb_observables = filtered_observations_feasibility.shape[1]

    # Produce density plots
    for i in range(len(layers)):
        layer = layers[i]
        if layer == "ocean":
            parameters = filtered_parameters_feasibility[:, 5 * i:5*i+2]
        else:
            parameters = filtered_parameters_feasibility[:, 5 * i:5 * i + 5]

        nb_interior_parameters = parameters.shape[1]

        for j in range(nb_interior_parameters):

            if layer == "ocean" and j == 0:
                parameters[:, j] = parameters[:, j] - filtered_parameters_feasibility[:, 0]
            elif layer == "shell" and j == 0:
                aux = np.loadtxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"))
                parameters[:, j] = parameters[:, j] - aux[:, 5]

            for k in range(nb_observables):

                if k == 0:
                    obs_label = "phi"
                elif k == 1:
                    obs_label = "k2"
                elif k == 2:
                    obs_label = "h2"

                density_matrix, parameter_grid_vec, observable_grid_vec = Util.get_parameter_number_density(j, parameters,
                    filtered_observations_feasibility[:, k], parameters_grid_step, observable_grid_step[obs_label], layer)
                density_matrix = density_matrix.T
                density_matrix = Util.flip_rows(density_matrix)
                if j == 2:
                    fig = plt.figure(figsize=(13, 5))
                else:
                    fig = plt.figure(figsize=(7, 5))
                ax = fig.add_subplot(111)
                plt.imshow(density_matrix, aspect='auto', interpolation="auto")
                plt.colorbar(label="Count")
                ax.set_xticks(ticks=np.arange(-0.5, density_matrix.shape[1], 1), labels=parameter_grid_vec, fontsize=8)
                ax.set_yticks(ticks=np.arange(-0.5, density_matrix.shape[0], 1), labels=np.flip(observable_grid_vec), fontsize=fontsize)
                plt.grid(True)
                ax.set_xlabel(interior_parameters_labels[5 * i + j], fontsize=fontsize)

                if k == 0:
                    plt.suptitle(f"Measurements: libration amplitude. Layer: {layer}", fontsize=fontsize)
                    ax.set_ylabel(r"$\phi$  [deg]", fontsize=fontsize)
                    plt.savefig(os.path.join(plots_path, f"density_plot_libration_amplitude_{layer}_{interior_parameters_labels[5 * i + j]}.pdf"))
                elif k == 1:
                    plt.suptitle(f"Measurements: k2 Love number. Layer: {layer}", fontsize=fontsize)
                    ax.set_ylabel(r"$k_2$  [-]", fontsize=fontsize)
                    plt.savefig(os.path.join(plots_path, f"density_plot_k2_Love_number_{layer}_{interior_parameters_labels[5 * i + j]}.pdf"))
                elif k == 2:
                    plt.suptitle(f"Measurements: h2 Love number. Layer: {layer}", fontsize=fontsize)
                    ax.set_ylabel(r"$h_2$  [-]", fontsize=fontsize)
                    plt.savefig(os.path.join(plots_path, f"density_plot_h2_Love_number_{layer}_{interior_parameters_labels[5 * i + j]}.pdf"))
                plt.close()


def plot_monte_carlo_analysis_observables_histogram(input_path,
                                                    observables_to_study,
                                                    grid_steps,
                                                    filter_observations_flag,
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

    # Filter parameters and observations based on the feasibility of the interior model
    filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(
        interior_models, observations)
    np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
    np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
    np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    # Filter parameters and observations based on libration amplitude and tidal heating observations
    if filter_observations_flag:
        filtered_parameters_libration, filtered_observations_libration, nb_feasible_models_libration = Util.filter_libration_amplitude(
            filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
        filtered_parameters_heating, filtered_observations_heating, nb_feasible_models_heating = Util.filter_tidal_heating(
            filtered_parameters_libration, filtered_observations_libration, tidal_heating_range
        )
        #filtered_parameters_feasibility = filtered_parameters_heating
        #filtered_observations_feasibility = filtered_observations_heating
        filtered_parameters_feasibility = filtered_parameters_libration
        filtered_observations_feasibility = filtered_observations_libration

    # Set axis label of observables
    observables_label = [r"$\phi$", r"$Re(k_2)$", r"$Re(h_2)$", r"$\dot{E}$", r"$Re(k_2)$", r"$Im(h_2)$"]

    for i in range(len(observables_to_study)):
        observable_label = observables_to_study[i]
        if observable_label == "k2_real":
            obs_index = 1
        else:
            raise ValueError("Observable not recognized.")

        mean = np.mean(filtered_observations_feasibility[:, obs_index])
        std = np.std(filtered_observations_feasibility[:, obs_index])

        grid = np.arange(min(filtered_observations_feasibility[:, obs_index]), max(filtered_observations_feasibility[:, obs_index]), grid_steps[i])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(filtered_observations_feasibility[:, obs_index], grid, histtype="bar")
        ax.set_xlabel(observables_label[obs_index], fontsize=fontsize)
        ax.set_ylabel("Count", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        # ax.grid(True)

        ax.axvline(x=mean, color="red")
        ax.text(0.025, 115, f"Mean: {str(mean)[:5]}; std: {str(std)[:5]}", fontsize=fontsize)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_path, f"observables_histogram_{observable_label}.pdf"))
        plt.close(fig)


def plot_histogram_filtered_parameters_from_observables(input_path,
                                                        filter_observations_flag,
                                                        observables_to_study: dict,
                                                        parameters_to_study,
                                                        parameters_grid_steps,
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

    # Filter parameters and observations based on the feasibility of the interior model
    filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = Util.filter_parameters(
        interior_models, observations)
    np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
    np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
    np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    # Filter parameters and observations based on libration amplitude and tidal heating observations
    if filter_observations_flag:
        filtered_parameters_libration, filtered_observations_libration, nb_feasible_models_libration = Util.filter_libration_amplitude(
            filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
        filtered_parameters_heating, filtered_observations_heating, nb_feasible_models_heating = Util.filter_tidal_heating(
            filtered_parameters_libration, filtered_observations_libration, tidal_heating_range
        )
        # filtered_parameters_feasibility = filtered_parameters_heating
        # filtered_observations_feasibility = filtered_observations_heating
        filtered_parameters_feasibility = filtered_parameters_libration
        filtered_observations_feasibility = filtered_observations_libration

    filtered_parameters = filtered_parameters_feasibility
    filtered_observations = filtered_observations_feasibility

    observables_to_study_labels = list(observables_to_study.keys())
    for current_observable in observables_to_study_labels:
        if current_observable == "k2_real":
            obs_index = 1
        elif current_observable == "h2_real":
            obs_index = 2
        elif current_observable == "k2_imag":
            obs_index = 3
        elif current_observable == "h2_imag":
            obs_index = 4

        filtered_parameters, filtered_observations, nb_viable_simulations = Util.filter_parameters(
            filtered_parameters,
            filtered_observations,
            observables_to_study[current_observable],
            obs_index
        )

    for current_parameter in parameters_to_study:
        if current_parameter == "d_core":
            param_index = 0
        elif current_parameter == "rho_core":
            param_index = 1
        elif current_parameter == "mu_core":
            param_index = 2
        elif current_parameter == "eta_core":
            param_index = 3
        elif current_parameter == "k_core":
            param_index = 4
        elif current_parameter == "d_ocean":
            param_index = 5
        elif current_parameter == "rho_ocean":
            param_index = 6
        elif current_parameter == "mu_ocean":
            param_index = 7
        elif current_parameter == "eta_ocean":
            param_index = 8
        elif current_parameter == "k_ocean":
            param_index = 9
        elif current_parameter == "d_shell":
            param_index = 10
        elif current_parameter == "rho_shell":
            param_index = 11
        elif current_parameter == "mu_shell":
            param_index = 12
        elif current_parameter == "eta_shell":
            param_index = 13
        elif current_parameter == "k_shell":
            param_index = 14

        mean_old = np.mean(filtered_parameters_feasibility[:, param_index])
        std_old = np.std(filtered_parameters_feasibility[:, param_index])

        grid_old = np.arange(min(filtered_observations_feasibility[:, param_index]),
                         max(filtered_observations_feasibility[:, param_index]),
                         parameters_grid_steps[current_parameter][0])

        mean_new = np.mean(filtered_parameters[:, param_index])
        std_new = np.std(filtered_parameters[:, param_index])

        grid_new = np.arange(min(filtered_parameters[:, param_index]),
                         max(filtered_parameters[:, param_index]),
                         parameters_grid_steps[current_parameter][1])

        # Plot histogram of parameter distribution before and after filtering the observables
        fig, axes = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)
        axes[0].hist(filtered_parameters_feasibility[:, param_index], grid_old)
        axes[0].axvline(mean_old, color='red')
        axes[0].set_title(f"Before filtering. Mean: {mean_old:.2f}, Std: {std_old:.2f}")
        axes[1].hist(filtered_parameters[:, param_index], grid_new)
        axes[1].axvline(mean_new, color='red')
        axes[1].set_title(f"After filtering. Mean: {mean_new:.2f}, Std: {std_new:.2f}")

        axes[0].set_xlabel(current_parameter, fontsize=fontsize)
        axes[1].set_xlabel(current_parameter, fontsize=fontsize)
        axes[0].set_ylabel("Count", fontsize=fontsize)
        axes[0].tick_params(labelsize=fontsize)
        axes[1].tick_params(labelsize=fontsize)

        fig.savefig(os.path.join(plots_path, f"filtered_parameters_histogram_{current_parameter}.pdf"))
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
        filter_parameters_flag = True
        filter_libration_amplitude_flag = True
        filter_tidal_heating_flag = False
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.08.29.10.18.16"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_interior_parameters_analysis_simple_plot(input_path,
                                                                  filter_parameters_flag,
                                                                  filter_libration_amplitude_flag,
                                                                  filter_tidal_heating_flag,
                                                                  nominal_libration_amplitude,
                                                                  tidal_heating_range,
                                                                  )


    plot_monte_carlo_interior_parameters_analysis_density_plot_flag = False
    if plot_monte_carlo_interior_parameters_analysis_density_plot_flag:
        parameters_grid_step = dict()
        parameters_grid_step["core"] = dict(
            d = 5,
            rho = 50,
            mu = 0.5e10,
            eta = 10,
            K = 10,
        )
        parameters_grid_step["ocean"] = dict(
            d = 5,
            rho = 50,
            mu = 10,
            eta = 10,
            K = 10,
        )
        parameters_grid_step["shell"] = dict(
            d = 5,
            rho = 25,
            mu = 1e9,
            eta = 10,
            K = 10,
        )
        observable_grid_step = dict(
            h2 = 0.01,
            k2 = 0.005,
            phi = 0.01,
        )
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.08.29.09.11.50"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_interior_parameters_analysis_density_plot(input_path,
                                                                   parameters_grid_step,
                                                                   observable_grid_step,
                                                                   True,
                                                                   nominal_libration_amplitude,
                                                                   tidal_heating_range)

    plot_monte_carlo_analysis_filtered_observables_histogram_flag = False
    if plot_monte_carlo_analysis_filtered_observables_histogram_flag:
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.08.29.09.11.50"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_analysis_observables_histogram(input_path,
                                                        ["k2_real"],
                                                        [0.0005],
                                                        True,
                                                        nominal_libration_amplitude,
                                                        tidal_heating_range
                                                        )

    plot_histogram_filtered_parameters_from_observables_flag = False
    if plot_histogram_filtered_parameters_from_observables_flag:
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.08.28.17.56.15"
        input_path = os.path.join(input_path, time_stamp)

        observables_to_study = dict(
            k2_real = 1e-4
        )

        parameters_to_study = ["d_shell"]
        parameters_grid_steps = dict(
            d_shell = [5, 1]
        )

        plot_histogram_filtered_parameters_from_observables(
            input_path,
            True,
            observables_to_study,
            parameters_to_study,
            parameters_grid_steps,
            nominal_libration_amplitude,
            tidal_heating_range,)


if __name__ == "__main__":
    main()
