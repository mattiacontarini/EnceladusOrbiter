# General imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sympy import ceiling


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
        filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = filter_parameters(interior_models, observations)
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
        filtered_parameters, filtered_observations, nb_feasible_libration_models = filter_libration_amplitude(
            filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
        filtered_parameters_feasibility = filtered_parameters
        filtered_observations_feasibility = filtered_observations

    if filter_tidal_heating_flag:
        filtered_parameters, filtered_observations, nb_feasible_tidal_heating_models = filter_tidal_heating(
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
    filtered_parameters_feasibility, filtered_observations_feasibility, nb_feasible_models = filter_parameters(interior_models, observations)
    np.savetxt(os.path.join(input_path, "filtered_parameters_feasibility.dat"), filtered_parameters_feasibility)
    np.savetxt(os.path.join(input_path, "filtered_observations_feasibility.dat"), filtered_observations_feasibility)
    np.savetxt(os.path.join(input_path, "nb_feasible_models.dat"), [nb_feasible_models])

    # Apply absolute value to libration amplitude
    filtered_observations_feasibility[:, 0] = np.abs(filtered_observations_feasibility[:, 0])

    if filter_observations_flag:
        filtered_parameters_libration, filtered_observations_libration, nb_feasible_models_libration = filter_libration_amplitude(
            filtered_parameters_feasibility, filtered_observations_feasibility, nominal_libration_amplitude)
        #filtered_parameters_heating, filtered_observations_heating, nb_feasible_models_heating = filter_tidal_heating(
        #    filtered_parameters_libration, filtered_observations_libration, tidal_heating_range
        #)
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

                density_matrix, parameter_grid_vec, observable_grid_vec = get_parameter_number_density(j, parameters,
                    filtered_observations_feasibility[:, k], parameters_grid_step, observable_grid_step[obs_label], layer)
                density_matrix = density_matrix.T
                if j == 2:
                    fig = plt.figure(figsize=(13, 5))
                else:
                    fig = plt.figure(figsize=(7, 5))
                ax = fig.add_subplot(111)
                plt.imshow(density_matrix, aspect='auto', interpolation="auto")
                plt.colorbar(label="Count")
                ax.set_xticks(ticks=np.arange(-0.5, density_matrix.shape[1], 1), labels=parameter_grid_vec, fontsize=8)
                ax.set_yticks(ticks=np.arange(-0.5, density_matrix.shape[0], 1), labels=observable_grid_vec, fontsize=fontsize)
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

def filter_parameters(parameters, observations):
    nb_simulations = parameters.shape[0]

    counter = 0
    filtered_parameters = np.copy(parameters)
    filtered_observations = np.copy(observations)
    for i in range(nb_simulations):
        core_density = parameters[i, 1]
        ocean_density = parameters[i, 6]
        shell_bulk_modulus = parameters[i, 14]
        shell_shear_modulus = parameters[i, 12]
        poisson_ratio = (3*shell_bulk_modulus - 2*shell_shear_modulus)/(6*shell_bulk_modulus + 2*shell_shear_modulus)

        if core_density <= 2000.0 or core_density >= 3000.0 or ocean_density <= 1000.0 or ocean_density >= 1300.0 or poisson_ratio <=0.3:
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    return filtered_parameters, filtered_observations, nb_simulations-counter


def get_parameter_number_density(parameter_index, parameters, observable, parameters_grid_step, observable_grid_step, layer):

    parameters_labels = list(parameters_grid_step[layer].keys())

    label = parameters_labels[parameter_index]

    if (label == "mu" or label == "eta" or label == "K") and parameters_grid_step[layer][label] == 10:
        num = min(parameters[:, parameter_index])
        parameter_grid_vec = [num]
        while num < max(parameters[:, parameter_index]):
            num = num*10
            parameter_grid_vec.append(num)
        nb_parameters_grid_steps = len(parameter_grid_vec) - 1

    else:
        nb_parameters_grid_steps = int(ceiling((max(parameters[:, parameter_index]) - min(parameters[:, parameter_index])) /
                                parameters_grid_step[layer][parameters_labels[parameter_index]]))
        #print(nb_parameters_grid_steps, max(parameters[:, parameter_index]), min(parameters[:, parameter_index]))

        print(layer, parameter_index, min(parameters[:, parameter_index]), max(parameters[:, parameter_index]))
        parameter_grid_vec = np.linspace(min(parameters[:, parameter_index]),
                                         max(parameters[:, parameter_index]),
                                         nb_parameters_grid_steps + 1)

    nb_observable_grid_steps = int(ceiling((max(observable) - min(observable)) / observable_grid_step))
    observable_grid_vec = np.linspace(min(observable), max(observable), nb_observable_grid_steps + 1)

    nb_simulations = parameters.shape[0]
    density_matrix = np.zeros((nb_parameters_grid_steps, nb_observable_grid_steps))
    parameter_vec = parameters[:, parameter_index]
    for i in range(nb_simulations):

        density_matrix_row = None
        for l in range(nb_parameters_grid_steps-1):

            if parameter_grid_vec[l] <= parameter_vec[i] < parameter_grid_vec[l+1]:
                density_matrix_row = l
                break
        if density_matrix_row is None:

            density_matrix_row = nb_parameters_grid_steps - 1

        density_matrix_col = None
        for m in range(nb_observable_grid_steps-1):
            if observable_grid_vec[m] <= observable[i] < observable_grid_vec[m+1]:
                density_matrix_col = m
                break
        if density_matrix_col is None:
            density_matrix_col = nb_observable_grid_steps - 1

        density_matrix[density_matrix_row, density_matrix_col] += 1

    parameter_grid_vec_labels = []
    if label == "mu" or label == "eta" or label == "K":
        for n in range(len(parameter_grid_vec)):
            parameter_grid_vec_labels.append(format_e(parameter_grid_vec[n]))
    else:
        for n in range(len(parameter_grid_vec)):
            parameter_grid_vec_labels.append(str(int(parameter_grid_vec[n])))

    observable_grid_vec_labels = []
    for n in range(len(observable_grid_vec)):
        observable_grid_vec_labels.append(str(observable_grid_vec[n])[:5])

    return density_matrix, parameter_grid_vec_labels, observable_grid_vec_labels


def filter_libration_amplitude(parameters, observations, nominal_libration,):
    nb_simulations = parameters.shape[0]
    counter = 0

    filtered_parameters = np.copy(parameters)
    filtered_observations = np.copy(observations)
    for i in range(nb_simulations):

        if ((observations[i, 0] >= nominal_libration[0] + nominal_libration[1]) or
            observations[i, 0] <= nominal_libration[0] - nominal_libration[1]):
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    return filtered_parameters, filtered_observations, nb_simulations-counter


def filter_tidal_heating(parameters, observations, tidal_heating_range,):
    nb_simulations = parameters.shape[0]
    counter = 0
    filtered_parameters = np.copy(parameters)
    filtered_observations = np.copy(observations)
    for i in range(nb_simulations):
        if (observations[i, 3] > tidal_heating_range[1] or observations[i, 3] < tidal_heating_range[0]):
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    return filtered_parameters, filtered_observations, nb_simulations-counter


def format_e(n):
    a = '%E' % n
    return a.split('E')[0][:3].rstrip('0').rstrip('.') + 'e' + a.split('E')[1]

def main():

    nominal_libration_amplitude = [0.120, 0.021]  # [deg]
    tidal_heating_range = [25, 40]  # [GW]

    plot_one_at_a_time_interior_parameters_analysis_flag = False
    if plot_one_at_a_time_interior_parameters_analysis_flag:
        input_path = "./output/interior_parameters_analysis/preliminary_sensitivity_analysis"
        plot_one_at_a_time_interior_parameters_analysis(input_path)

    plot_monte_carlo_interior_parameters_analysis_flag = False
    if plot_monte_carlo_interior_parameters_analysis_flag:
        filter_parameters_flag = True
        filter_libration_amplitude_flag = True
        filter_tidal_heating_flag = True
        input_path = "./output/interior_parameters_analysis/monte_carlo_analysis"
        time_stamp = "2025.08.27.10.00.10"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_interior_parameters_analysis_simple_plot(input_path,
                                                                  filter_parameters_flag,
                                                                  filter_libration_amplitude_flag,
                                                                  filter_tidal_heating_flag,
                                                                  nominal_libration_amplitude,
                                                                  tidal_heating_range,
                                                                  )


    plot_monte_carlo_interior_parameters_analysis_density_plot_flag = True
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
        time_stamp = "2025.08.27.10.00.10"
        input_path = os.path.join(input_path, time_stamp)
        plot_monte_carlo_interior_parameters_analysis_density_plot(input_path,
                                                                   parameters_grid_step,
                                                                   observable_grid_step,
                                                                   True,
                                                                   nominal_libration_amplitude,
                                                                   tidal_heating_range)

if __name__ == "__main__":
    main()
