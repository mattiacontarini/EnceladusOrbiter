# Tudat import
from tudatpy import constants
from tudatpy import numerical_simulation
from tudatpy.astro.time_conversion import DateTime

# Simulation timeline
simulation_start_epoch = DateTime(2000, 1, 1).epoch()  # From Benedikter et al. (2022)
simulation_duration = 28 * constants.JULIAN_DAY
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
        numerical_simulation.propagation_setup.acceleration.spherical_harmonic_gravity(3, 3),
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
fixed_step_size = 15
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_fixed_step(
    fixed_step_size,
    numerical_simulation.propagation_setup.integrator.CoefficientSets.rkf_56,
    order_to_use=numerical_simulation.propagation_setup.integrator.OrderToIntegrate.higher
)
