

def get_kaula_constraint(kaula_constraint_multiplier, degree):
    return kaula_constraint_multiplier / degree ** 2


def apply_kaula_constraint_a_priori(kaula_constraint_multiplier, max_deg_gravity, indices_cosine_coef, indices_sine_coef, inv_apriori):

    index_cosine_coef = indices_cosine_coef[0]
    index_sine_coef = indices_sine_coef[0]

    for deg in range(2, max_deg_gravity + 1):
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