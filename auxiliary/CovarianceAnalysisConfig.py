# Tudat import
from tudatpy import constants
from tudatpy import numerical_simulation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion

# Packages import
import numpy as np

#######################################################################################################################
### Configuration #####################################################################################################
#######################################################################################################################

# Simulation epochs
simulation_start_epoch = DateTime(2000, 1, 1).epoch()  # From Benedikter et al. (2022)
simulation_duration = 28 * constants.JULIAN_DAY  # From Benedikter et al. (2022)
simulation_end_epoch = simulation_start_epoch + simulation_duration

initial_state_index = 1

#######################################################################################################################
### Environment #######################################################################################################
#######################################################################################################################

# Bodies included in the environment model
bodies_to_create = ["Sun",
                    "Earth",
                    "Saturn",
                    "Enceladus",
                    "Mimas",
                    "Tethys",
                    "Dione",
                    "Rhea",
                    "Titan"]

# Frame settings
global_frame_origin = "Enceladus"
global_frame_orientation = "J2000"

# Scaled mean moment of inertia of Enceladus and Saturn
Enceladus_scaled_mean_moment_of_inertia = 0.335  # From Iess et al. (2014)
Saturn_scaled_mean_moment_of_inertia = 0.210  # From NASA Saturn Fact Sheet

# Occulting bodies
occulting_bodies = dict()
occulting_bodies["Sun"] = ["Enceladus"]

#######################################################################################################################
### Propagation #######################################################################################################
#######################################################################################################################

# Define numerical integrator settings
fixed_step_size = 15
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size,
    numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56,
    order_to_use=numerical_simulation.propagation_setup.integrator.OrderToIntegrate.higher
)

# Define propagation arcs during science phase
arc_duration = 1.0 * constants.JULIAN_DAY

# Lagrange interpolator settings
number_of_points = 8

# Define dependent variables to be saved during the propagation
dependent_variables_to_save = [
    numerical_simulation.propagation_setup.dependent_variable.altitude("Vehicle", "Enceladus"),
    numerical_simulation.propagation_setup.dependent_variable.latitude("Vehicle", "Enceladus"),
    numerical_simulation.propagation_setup.dependent_variable.longitude("Vehicle", "Enceladus"),
    numerical_simulation.propagation_setup.dependent_variable.total_acceleration("Vehicle"),
    numerical_simulation.propagation_setup.dependent_variable.rsw_to_inertial_rotation_matrix("Vehicle", "Enceladus"),
]

indices_dependent_variables = dict(
    altitude=[1, 2],
    latitude=[2, 3],
    longitude=[3, 4],
    total_acceleration=[4, 7],
    rsw_to_inertial_rotation_matrix=[7, 16]
)

#######################################################################################################################
### Observations
#######################################################################################################################

# Ground stations properties
ground_station_names = ["Malargue", "NewNorcia", "Cebreros"]
ground_station_coordinates = {
    ground_station_names[0]: [1550.0, np.deg2rad(-35.0), np.deg2rad(-69.0)],
    ground_station_names[1]: [252.0, np.deg2rad(-31.0), np.deg2rad(116.0)],
    ground_station_names[2]: [794.0, np.deg2rad(40.0), np.deg2rad(-4.0)]
}
ground_station_coordinates_type = {ground_station_names[0]: element_conversion.geodetic_position_type,
                                   ground_station_names[1]: element_conversion.geodetic_position_type,
                                   ground_station_names[2]: element_conversion.geodetic_position_type}


# Radio beacons properties
lander_names = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
lander_coordinates = {
    lander_names[0]: [0.0, np.deg2rad(-60.0), np.deg2rad(0.0)],
    lander_names[1]: [0.0, np.deg2rad(-30.0), np.deg2rad(40.0)],
    lander_names[2]: [0.0, np.deg2rad(0.0), np.deg2rad(80.0)],
    lander_names[3]: [0.0, np.deg2rad(30.0), np.deg2rad(120.0)],
    lander_names[4]: [0.0, np.deg2rad(60.0), np.deg2rad(160.0)],
    lander_names[5]: [0.0, np.deg2rad(-60.0), np.deg2rad(200.0)],
    lander_names[6]: [0.0, np.deg2rad(-30.0), np.deg2rad(240.0)],
    lander_names[7]: [0.0, np.deg2rad(0.0), np.deg2rad(280.0)],
    lander_names[8]: [0.0, np.deg2rad(30.0), np.deg2rad(320.0)]
}
lander_coordinates_type = {lander_names[0]: element_conversion.geodetic_position_type,
                           lander_names[1]: element_conversion.geodetic_position_type,
                           lander_names[2]: element_conversion.geodetic_position_type,
                           lander_names[3]: element_conversion.geodetic_position_type,
                           lander_names[4]: element_conversion.geodetic_position_type,
                           lander_names[5]: element_conversion.geodetic_position_type,
                           lander_names[6]: element_conversion.geodetic_position_type,
                           lander_names[7]: element_conversion.geodetic_position_type,
                           lander_names[8]: element_conversion.geodetic_position_type}

# Tracking arcs properties
tracking_arc_duration = 8.0 * 3600.0
tracking_delay_after_stat_of_propagation = 2.0 * 3600.0

# Length of the arc over which the empirical accelerations are estimated
empirical_accelerations_arc_duration = 0.5 * constants.JULIAN_DAY

# Define observation simulation times for both Doppler and range observarbles
doppler_cadence = 60
range_cadence = 300.0

range_bias = 2.0

# Minimum D/O for spherical harmonic cosine coefficients of Enceladus
minimum_degree_c_enceladus = 2
minimum_order_c_enceladus  = 0
maximum_degree_gravity_enceladus = 30

# Minimum D/O for spherical harmonic sine coefficients of Enceladus
minimum_degree_s_enceladus = 2
minimum_order_s_enceladus  = 1

# A priori constraints on the arc-wise initial state of the vehicle
a_priori_position = 5.0e3
a_priori_velocity = 0.5

# A priori constraints on gravitational parameter of Enceladus
a_priori_gravitational_parameter_enceladus = 0.03e9

# A priori constraints on gravity field coefficients of Enceladus - Park et al., 2024
a_priori_c20 = 36.99e-6
a_priori_c21 = 13.66e-6
a_priori_c22 = 14.70e-6
a_priori_c30 = 33.42e-6
a_priori_s21 = 9.19e-6
a_priori_s22 = 10.87e-6

# A priori constraints on empirical accelerations
a_priori_empirical_accelerations = 4.0e-7  # From Durante et al., 2020

# A priori constraint on position of landers on Enceladus
a_priori_lander_position = 1e2

# A priori constraint for tidal k2 love number - [Re, Im]
a_priori_k2_love_number = [0.0, 0.0] # [3.5e-4, 1.7e-4]

# A priori constraint on the rotation pole position
a_priori_rotation_pole_right_ascension = np.deg2rad(0.01)
a_priori_rotation_pole_declination = np.deg2rad(0.01)

# A priori constraint on the diurnal libration amplitude of Enceladus
a_priori_libration_amplitude = np.deg2rad(0.003)  # From Park et al., (2024)

# A priori constraint on the pole rate
a_priori_pole_rate = np.deg2rad(0.01)

# A priori constraint on the radiation pressure coefficients
a_priori_radiation_pressure_coefficient = np.infty

# A priori constraint on range bias for Earth ground stations
a_priori_range_bias_Earth_ground_station = 2.0

# A priori constraint on range bias for Enceladus landers
a_priori_range_bias_lander = 2.0

# A priori constraint on station position for Earth ground stations
a_priori_station_position_Earth_ground_station = 5.0e-3

# Minimum elevation angle for visibility
minimum_elevation_angle_visibility = np.deg2rad(15.0)
minimum_sep_angle = np.deg2rad(5.0)

# Measurements noise
doppler_noise_Earth_ground_station = 12.0e-6
range_noise_Earth_ground_station = 0.2
doppler_noise_lander = 1e-4
range_noise_lander = 1

# Kaula constraint factor
#kaula_constraint_multiplier = 40.0e-5  # From Genova et al., 2024
kaula_constraint_multiplier = 1e-3  # From Zannoni et al., 2020

# Longitudinal libration angular frequencies to consider for the estimation of the libration amplitudes
JULIAN_CENTURY = 100 * constants.JULIAN_YEAR
libration_angular_frequencies = [np.deg2rad(9583937.8056363) / JULIAN_CENTURY]

#######################################################################################################################
### Accelerations #####################################################################################################
#######################################################################################################################

# Define the accelerations acting on the vehicle
acceleration_settings_on_vehicle = dict(
    Sun=[
        numerical_simulation.propagation_setup.acceleration.radiation_pressure(),
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Saturn=[
        numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
    ],
    Enceladus=[
        numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(maximum_degree_gravity_enceladus, maximum_degree_gravity_enceladus),
        numerical_simulation.propagation_setup.acceleration.empirical()
        # numerical_simulation.propagation_setup.acceleration.aerodynamic()
    ],
    Mimas=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Tethys=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Dione=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Rhea=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ],
    Titan=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity()
    ]
)

#######################################################################################################################
### Estimation ########################################################################################################
#######################################################################################################################

empirical_acceleration_components_to_estimate = dict()
empirical_acceleration_components_to_estimate[numerical_simulation.estimation_setup.parameter.radial_empirical_acceleration_component] = [numerical_simulation.estimation_setup.parameter.constant_empirical
                                                                                                                                          #numerical_simulation.estimation_setup.parameter.cosine_empirical,
                                                                                                                                          #numerical_simulation.estimation_setup.parameter.sine_empirical
                                                                                                                                          ]


empirical_acceleration_components_to_estimate[numerical_simulation.estimation_setup.parameter.along_track_empirical_acceleration_component] = [numerical_simulation.estimation_setup.parameter.constant_empirical
                                                                                                                                          #numerical_simulation.estimation_setup.parameter.cosine_empirical,
                                                                                                                                          #numerical_simulation.estimation_setup.parameter.sine_empirical
                                                                                                                                               ]


empirical_acceleration_components_to_estimate[numerical_simulation.estimation_setup.parameter.across_track_empirical_acceleration_component] = [numerical_simulation.estimation_setup.parameter.constant_empirical
                                                                                                                                          #numerical_simulation.estimation_setup.parameter.cosine_empirical,
                                                                                                                                          #numerical_simulation.estimation_setup.parameter.sine_empirical
                                                                                                                                                ]
