
# Tudat import
from tudatpy import astro
from tudatpy.interface import spice
from tudatpy.numerical_simulation.estimation_setup import observation

# Files and variable import
from auxiliary import CovarianceAnalysisConfig as CovAnalysisConfig

# General packages import
import numpy as np


def get_kaula_constraint(kaula_constraint_multiplier, degree):
    return kaula_constraint_multiplier / degree ** 2


def apply_kaula_constraint_a_priori(kaula_constraint_multiplier, max_deg_gravity, min_deg_gravity, indices_cosine_coef, indices_sine_coef, inv_apriori):

    index_cosine_coef = indices_cosine_coef[0]
    index_sine_coef = indices_sine_coef[0]

    for deg in range(min_deg_gravity, max_deg_gravity + 1):
        kaula_constraint = get_kaula_constraint(kaula_constraint_multiplier, deg)
        for order in range(deg + 1):
            inv_apriori[index_cosine_coef, index_cosine_coef] = kaula_constraint ** -2
            index_cosine_coef += 1
        for order in range(1, deg + 1):
            inv_apriori[index_sine_coef, index_sine_coef] = kaula_constraint ** -2
            index_sine_coef += 1

    return inv_apriori


def get_number_observations_for_station_type(partials_matrix,
                                             station_type,
                                             indices_lander_position):

    nb_observations = 0
    if station_type == "ground_station":
        for i in range(partials_matrix.shape[0]):
            zeros_row_flag = True
            for j in range(indices_lander_position[0], indices_lander_position[0] + indices_lander_position[1]):
                if partials_matrix[i, j] != 0.0:
                    zeros_row_flag = False
            if zeros_row_flag:
                nb_observations += 1
    elif station_type == "lander":
        for i in range(partials_matrix.shape[0]):
            zeros_row_flag = True
            for j in range(indices_lander_position[0], indices_lander_position[0] + indices_lander_position[1]):
                if partials_matrix[i, j] != 0.0:
                    zeros_row_flag = False
            if not zeros_row_flag:
                nb_observations += 1

    return nb_observations


def extend_design_matrix_to_h2_love_number(design_matrix,
                                           indices_lander_position,
                                           gravitational_parameter_ratio,
                                           station_position,
                                           station_name,
                                           body_equatorial_radius,
                                           sorted_observation_epochs,
                                           bodies):

    nb_observations = design_matrix.shape[0]
    nb_parameters = design_matrix.shape[1]
    nb_parameters_extended = nb_parameters + 1
    design_matrix_extended = np.zeros((nb_observations, nb_parameters_extended))
    design_matrix_extended[:, :nb_parameters] = design_matrix

    station_position_unit_vector = station_position / np.linalg.norm(station_position)
    dh_drL = design_matrix[:, indices_lander_position[0]:indices_lander_position[0] + indices_lander_position[1]]
    dh_dh2 = np.zeros((nb_observations,))
    drL_dh2_store = dict()
    dh_dh2_store = dict()
    for i in range(nb_observations):
        epoch = sorted_observation_epochs[i]

        # Compute nominal station position displacement
        saturn_position_cartesian = bodies.get("Saturn").state_in_base_frame_from_ephemeris(epoch)[:3]
        enceladus_inertial_to_body_fixed_rotation_matrix = bodies.get("Enceladus").rotation_model.inertial_to_body_fixed_rotation(epoch)
        relative_body_position_cartesian = np.dot(enceladus_inertial_to_body_fixed_rotation_matrix, saturn_position_cartesian)
        drL_dh2_i = astro.gravitation.calculate_degree_two_basic_tidal_displacement(gravitational_parameter_ratio,
                                                                                    station_position_unit_vector,
                                                                                    relative_body_position_cartesian,
                                                                                    body_equatorial_radius,
                                                                                    1.0,
                                                                                    0.0)

        drL_dh2_average = CovAnalysisConfig.lander_average_position_deformation[station_name]
        drL_dh2_i = np.abs(drL_dh2_i - drL_dh2_average)
        dh_dh2[i] = np.dot(dh_drL[i, :], drL_dh2_i)

        drL_dh2_store[epoch] = drL_dh2_i
        dh_dh2_store[epoch] = dh_dh2[i]
    
    design_matrix_extended[:, nb_parameters_extended - 1] = dh_dh2
    return design_matrix_extended, drL_dh2_store, dh_dh2_store


def retrieve_sorted_observation_epochs(simulated_observations,):

    sorted_observations = simulated_observations.sorted_observation_sets
    sorted_range_observations = sorted_observations[observation.n_way_range_type]
    sorted_doppler_observations = sorted_observations[observation.n_way_averaged_doppler_type]

    sorted_observation_epochs = []
    for i in list(sorted_range_observations.keys()):
        epochs = sorted_range_observations[i][0].observation_times
        for epoch in epochs:
            sorted_observation_epochs.append(epoch)
    for i in list(sorted_doppler_observations.keys()):
        epochs = sorted_doppler_observations[i][0].observation_times
        for epoch in epochs:
            sorted_observation_epochs.append(epoch)

    return sorted_observation_epochs


def get_normalization_terms(partials_matrix):
    normalization_terms = np.zeros((partials_matrix.shape[1],))
    for i in range(partials_matrix.shape[1]):
        maximum_value = max(partials_matrix[:, i])
        minimum_value = min(partials_matrix[:, i])
        if np.abs(minimum_value) > maximum_value:
            normalization_terms[i] = minimum_value
        else:
            normalization_terms[i] = maximum_value
        if normalization_terms[i] == 0.0:
            normalization_terms[i] = 1.0

    return normalization_terms


def normalize_design_matrix(design_matrix, normalization_terms):
    normalized_design_matrix = np.zeros(design_matrix.shape)
    for j in range(design_matrix.shape[1]):
        normalized_design_matrix[:, j] = design_matrix[:, j] / normalization_terms[j]

    return normalized_design_matrix


def normalize_inv_apriori_covariance_matrix(inv_apriori_covariance_matrix, normalization_terms):
    normalized_inv_apriori_covariance_matrix = np.zeros(inv_apriori_covariance_matrix.shape)
    for i in range(inv_apriori_covariance_matrix.shape[0]):
        for j in range(inv_apriori_covariance_matrix.shape[1]):
            normalized_inv_apriori_covariance_matrix[i, j] = inv_apriori_covariance_matrix[i, j] / (normalization_terms[i] * normalization_terms[j])

    return normalized_inv_apriori_covariance_matrix


def normalize_covariance_matrix(matrix, normalization_terms):
    normalized_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            normalized_matrix[i, j] = matrix[i, j] * (normalization_terms[i] * normalization_terms[j])

    return normalized_matrix


def unnormalize_covariance_matrix(normalized_matrix, normalization_terms):
    unnormalized_matrix = np.zeros(normalized_matrix.shape)
    for i in range(normalized_matrix.shape[0]):
        for j in range(normalized_matrix.shape[1]):
            unnormalized_matrix[i, j] = normalized_matrix[i, j] / (normalization_terms[i] * normalization_terms[j])

    return unnormalized_matrix


def get_formal_errors(covariance_matrix):
    covariance_diagonal = np.diag(covariance_matrix)
    formal_errors = np.sqrt(covariance_diagonal)

    return formal_errors


def get_correlation_matrix(covariance_matrix, formal_errors):
    correlation_matrix = np.zeros(covariance_matrix.shape)
    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            correlation_matrix[i, j] = covariance_matrix[i, j] / (formal_errors[i] * formal_errors[j])

    return correlation_matrix


def get_number_of_observations_per_station_type(doppler_obs_time_list, station_labels):

    nb_observation_times = dict()
    for i in range(len(doppler_obs_time_list)):
        doppler_obs_time = doppler_obs_time_list[i]
        nb_observation_times[station_labels[i]] = len(doppler_obs_time)

    return nb_observation_times


