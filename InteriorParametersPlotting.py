# General imports
import numpy as np
import os
import matplotlib.pyplot as plt

def perform_interior_parameters_analysis_plotting(input_path, fontsize=12):

    layers = ["core", "ocean", "shell"]
    interior_parameters = ["R0", "rho0", "mu0", "eta0", "Ks0"]
    interior_parameters_labels = [r"$R_{c}$", r"$\rho_{c}$", r"$\mu_{c}$", r"$\eta_{c}$", r"$K_{c}$",
                                  r"$R_{o}$", r"$\rho_{o}$", r"$\mu_{o}$", r"$\eta_{o}$", r"$K_{o}$",
                                  r"$R_{s}$", r"$\rho_{s}$", r"$\mu_{s}$", r"$\eta_{s}$", r"$K_{s}$"]
    for layer in layers:
        layer_input_path = os.path.join(input_path, layer)
        layer_index = layers.index(layer)

        fig, axes = plt.subplots(3, 2, constrained_layout=True, figsize=(8, 8))
        for interior_parameter in interior_parameters:
            interior_parameter_path = os.path.join(layer_input_path, f"{interior_parameter}.dat")
            interior_parameter_index = interior_parameters.index(interior_parameter)

            # Load results
            tidal_love_numbers = np.loadtxt(interior_parameter_path, delimiter=",")
            interior_parameter_values = tidal_love_numbers[:, 0]
            k2_love_number_values = tidal_love_numbers[:, 1]
            h2_love_number_values = tidal_love_numbers[:, 2]

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

            ax.plot(interior_parameter_values, k2_love_number_values.real)
            ax.set_xlabel(interior_parameters_labels[len(interior_parameters)*layer_index + interior_parameter_index], fontsize=fontsize)

            if interior_parameter_index % 2 == 0:
                ax.set_ylabel(r"$k_{2}$", fontsize=fontsize)

        for ax in axes.flat:
            ax.tick_params(labelsize=fontsize)
            ax.grid(True)
        plt.delaxes(axes[2, 1])
        fig.savefig(os.path.join(layer_input_path, "k2_love_number.pdf"))
        plt.close(fig)


def main():

    perform_interior_parameters_analysis_flag = True
    if perform_interior_parameters_analysis_flag:
        input_path = "./output/interior_parameters_analysis"
        perform_interior_parameters_analysis_plotting(input_path)


if __name__ == "__main__":
    main()
