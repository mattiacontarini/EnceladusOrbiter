# General imports
import numpy as np
from sympy import ceiling

def filter_parameters(parameters, observations):
    nb_simulations = parameters.shape[0]

    counter = 0
    filtered_parameters = np.copy(parameters)
    filtered_observations = np.copy(observations)
    for i in range(nb_simulations):
        core_density = parameters[i, 1]
        ocean_density = parameters[i, 6]
        shell_thickness = parameters[i, 10] - parameters[i, 5]

        if core_density <= 2000.0 or core_density >= 3000.0 or ocean_density <= 1000.0 or ocean_density >= 1300.0 or shell_thickness <= 0.0:
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    return filtered_parameters, filtered_observations, nb_simulations-counter


def get_parameter_grid_vec(parameter_index, parameters, observable, parameters_grid_step, parameters_intervals,
                           observable_grid_step, layer):

    parameters_labels = list(parameters_grid_step[layer].keys())

    label = parameters_labels[parameter_index]

    if (label == "mu" or label == "eta" or label == "K"):
        parameter_grid_vec = np.logspace(np.log10(parameters_intervals[layer][label][0]),
                                         np.log10(parameters_intervals[layer][label][1]),
                                         parameters_grid_step[layer][parameters_labels[parameter_index]])


    else:
        nb_parameters_grid_steps = int(
            ceiling((parameters_intervals[layer][label][1] - parameters_intervals[layer][label][0]) /
                    parameters_grid_step[layer][parameters_labels[parameter_index]]))
        parameter_grid_vec = np.linspace(parameters_intervals[layer][label][0],
                                         parameters_intervals[layer][label][1],
                                         nb_parameters_grid_steps + 1)

    nb_observable_grid_steps = int(ceiling((max(observable) - min(observable)) / observable_grid_step))
    observable_grid_vec = np.linspace(min(observable), max(observable), nb_observable_grid_steps + 1)

    return parameter_grid_vec, observable_grid_vec


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
            parameter_grid_vec_labels.append(str(round(parameter_grid_vec[n])))

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
        print(observations[i, 3])
        if (observations[i, 3] > tidal_heating_range[1] or observations[i, 3] < tidal_heating_range[0]):
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    return filtered_parameters, filtered_observations, nb_simulations-counter


def filter_observable(parameters, observations, observable_std, observable_index):
    nb_simulations = parameters.shape[0]
    counter = 0
    filtered_parameters = np.copy(parameters)
    filtered_observations = np.copy(observations)

    observable_mean = np.mean(observations[:, observable_index])
    for i in range(nb_simulations):
        if (observations[i, observable_index] >= observable_mean + 3*observable_std or
                observations[i, observable_index] <= observable_mean - 3*observable_std):
            filtered_parameters = np.delete(filtered_parameters, i - counter, 0)
            filtered_observations = np.delete(filtered_observations, i - counter, 0)
            counter += 1

    return filtered_parameters, filtered_observations, nb_simulations-counter

def format_e(n):
    a = '%E' % n
    return a.split('E')[0][:3].rstrip('0').rstrip('.') + 'e' + a.split('E')[1]


def flip_rows(matrix):
    y_length = matrix.shape[0]
    matrix_out = np.zeros(matrix.shape)
    for i in range(y_length):
        matrix_out[i, :] = matrix[y_length-1-i, :]
    return matrix_out
