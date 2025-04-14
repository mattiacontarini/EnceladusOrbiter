# Tudat import
from tudatpy import constants
from tudatpy import numerical_simulation
from tudatpy.astro.time_conversion import DateTime

# Simulation timeline
simulation_start_epoch = DateTime(2000, 1, 1).epoch()  # From Benedikter et al. (2022)
simulation_duration = 20.52 * constants.JULIAN_DAY  # From Benedikter et al. (2022)
simulation_end_epoch = simulation_start_epoch + simulation_duration

termination_altitude = 0.0

# Bodies included in the environment model
bodies_to_create = ["Saturn",
                    "Enceladus",
                    "Sun",
                    "Mimas",
                    "Tethys",
                    "Dione",
                    "Rhea",
                    "Titan"]

Jupiter_barycenter_flag = False

max_degree_order = dict(
    Enceladus=[3, 3],
    Saturn=[8, 8],
)

acceleration_settings_on_vehicle = dict(
    Saturn=[
        numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(max_degree_order["Saturn"][0], max_degree_order["Saturn"][1]),
    ],
    Enceladus=[
        numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(max_degree_order["Enceladus"][0], max_degree_order["Enceladus"][1])
    ],
    Sun=[
        numerical_simulation.propagation_setup.acceleration.point_mass_gravity(),
        numerical_simulation.propagation_setup.acceleration.radiation_pressure()
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

# Create numerical integrator settings
fixed_step_size = 10
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size,
    numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_78,
    order_to_use=numerical_simulation.propagation_setup.integrator.OrderToIntegrate.higher
)
