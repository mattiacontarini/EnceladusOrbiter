clearvars
clc
addpath("src/")
addpath("auxiliary/utilities/")

%% Setup

save_results_flag = 1;

% Set output directory
if save_results_flag == 1
    output_directory = "output/interior_parameters_analysis/monte_carlo_analysis";
    time_stamp = string(datetime("now", "Format", "yyyy.MM.dd.HH.mm.ss"));
    output_path = fullfile(output_directory, time_stamp);
    if ~exist("output_directory", 'dir')
        mkdir(output_path)
    end
end


% Setup variables to tune through MC analysis
interior_parameters_to_vary_top = dictionary;
interior_parameters_to_vary_top("R_core") = 213.0e3;  % [m]
interior_parameters_to_vary_top("mu_core") = 80.0e9; 
interior_parameters_to_vary_top("eta_core") = 1.0e22;
interior_parameters_to_vary_top("K_core") = 1.0e11;
interior_parameters_to_vary_top("d_ocean") = 35.0e3;  % [m]
interior_parameters_to_vary_top("mu_ocean") = 1.0;
interior_parameters_to_vary_top("eta_ocean")= 1.0e-2;
interior_parameters_to_vary_top("K_ocean") = 1.0e10; %1.0e10;
interior_parameters_to_vary_top("rho_shell") = 1000.0;
interior_parameters_to_vary_top("mu_shell") = 5.0e9;
interior_parameters_to_vary_top("eta_shell") = 1.0e20;
interior_parameters_to_vary_top("K_shell") = 1.0e11;

interior_parameters_to_vary_bottom = dictionary;
interior_parameters_to_vary_bottom("R_core") = 180.0e3;  % [m]
interior_parameters_to_vary_bottom("mu_core") = 4.0e9; 
interior_parameters_to_vary_bottom("eta_core") = 1.0e11;
interior_parameters_to_vary_bottom("K_core") = 1.0e9;
interior_parameters_to_vary_bottom("d_ocean") = 5.0e3;  % [m]
interior_parameters_to_vary_bottom("mu_ocean") = 0.1;
interior_parameters_to_vary_bottom("eta_ocean")= 1.0e-4;
interior_parameters_to_vary_bottom("K_ocean") = 1.0e9; % 1.0e8;
interior_parameters_to_vary_bottom("rho_shell") = 800.0;
interior_parameters_to_vary_bottom("mu_shell") = 1.0e9;
interior_parameters_to_vary_bottom("eta_shell") = 1.0e12;
interior_parameters_to_vary_bottom("K_shell") = 1.0e9;

% Set number of samples per variable
nb_samples_per_variables = 10000;

% Set seed
seed = 1702;

% Nominal values for the mass, MoI, radius of Enceladus
global R_Enceladus M_Enceladus MoI_Enceladus
R_Enceladus = 252.1e3; % Porco et al. (2006)
M_Enceladus = 1.08e20; % Flandes et al. (2023)
MoI_Enceladus = 0.338 * M_Enceladus * R_Enceladus^2; % Iess et al. (2014)

% Forcing
Forcing_Enceladus(1).Td=33*3600; 
Forcing_Enceladus(1).n=2; 
Forcing_Enceladus(1).m=0; 
Forcing_Enceladus(1).F=1;
Forcing_Enceladus(1).eccen = 0.0047;

% Numerics
Numerics_Enceladus.Nlayers = 4; % number of concentric layers. Including the core!
Numerics_Enceladus.method = 'variable'; % method of setting the radial points per layer
Numerics_Enceladus.Nrbase = 600;

% Code parallelization
Numerics_Enceladus.parallel_sol = 0; % Use a parfor-loop to call get_Love, either 0 or 1
Numerics_Enceladus.parallel_gen = 0; % Calculate potential coupling files and the propagation inside get_solution using parfor-loops, either 0 or 1
Numerics_Enceladus.perturbation_order = 2;

%% Run Monte Carlo analysis

% Retrieve nb of simulations
control_variables_names = keys(interior_parameters_to_vary_top);
nb_variables = length(control_variables_names);
nb_simulations = nb_variables * nb_samples_per_variables;

% Set random number generator
rng(seed);

% Generate random samples for each interior parameter
samples_store = zeros(nb_simulations, nb_variables);
output_store = zeros(nb_simulations, 3);
interior_model_store = zeros(nb_simulations, nb_variables + 3);
tic
for i = 1:nb_simulations
    fprintf("Running simulation nb. %d\n", i)

    % Generate random samples for control variables
    samples = zeros(1, nb_variables);
    for j = 1:nb_variables
        variable_name = control_variables_names(j);
        bottom = interior_parameters_to_vary_bottom(variable_name);
        top = interior_parameters_to_vary_top(variable_name);
        samples(j) = 10^unifrnd(log10(bottom), log10(top));
    end
    samples_store(i, :) = samples;

    % Setup interior model with the given samples
    Interior_Model = setup_interior_model(samples);
    interior_model_store(i, :) = InteriorModelInversionUtilities.convert_interior_model_to_array(Interior_Model);

    % Compute shell libration
    [libration] = get_libration(Interior_Model, Forcing_Enceladus);
    shell_libration = real(libration.amplitude_rad(1));
    
    % Compute Love numbers
    [Numerics, Interior_Model] = set_boundary_indices(Numerics_Enceladus, Interior_Model);
    Interior_Model = get_rheology(Interior_Model, Numerics, Forcing_Enceladus);
    [Love_Spectra, y_rad] = get_Love(Interior_Model, Forcing_Enceladus, Numerics);
    iforcing=find(Love_Spectra.n==Forcing_Enceladus.n & Love_Spectra.m==Forcing_Enceladus.m);
    k2 = real(Love_Spectra.k(iforcing));
    h2 = real(Love_Spectra.h(iforcing));
    
    output_store(i, 1) = shell_libration;
    output_store(i, 2) = k2;
    output_store(i, 3) = h2;

end
toc

% Save results to file
if save_results_flag
    output_filepath = fullfile(output_path, "observations.dat");
    writematrix(output_store, output_filepath)

    samples_filepath = fullfile(output_path, "samples.dat");
    writematrix(samples_store, samples_filepath)

    interior_model_filepath = fullfile(output_path, "interior_models.dat");
    writematrix(interior_model_store, interior_model_filepath)

    nb_simulations_filepath = fullfile(output_path, "nb_simulations.dat");
    writematrix(nb_simulations, nb_simulations_filepath)

end

%%
function InteriorModel = setup_interior_model(samples)

R_ocean = samples(1) + samples(5);

global M_Enceladus R_Enceladus MoI_Enceladus

rho_ocean = InteriorModelInversionUtilities.get_ocean_density(M_Enceladus, MoI_Enceladus, R_Enceladus, samples(1), R_ocean, samples(9));
rho_core = InteriorModelInversionUtilities.get_core_density(M_Enceladus, R_Enceladus, samples(1), R_ocean, samples(9), rho_ocean);

disp(samples(1))
disp(samples(5))
disp(R_ocean)
disp(rho_core)
disp(rho_ocean)

InteriorModel(1).R0 = 5;
InteriorModel(1).rho0 = 5000;

InteriorModel(2).R0 = samples(1) * 1e-3;
InteriorModel(2).rho0 = rho_core;
InteriorModel(2).ocean=0;
InteriorModel(2).mu0=samples(2);
InteriorModel(2).eta0=samples(3);
InteriorModel(2).Ks0 = samples(4); 

InteriorModel(3).R0=R_ocean * 1e-3; 
InteriorModel(3).rho0=rho_ocean; 
InteriorModel(3).ocean=1; 
InteriorModel(3).mu0=samples(6); 
InteriorModel(3).eta0=samples(7); 
InteriorModel(3).Ks0=samples(8);

InteriorModel(4).R0 = R_Enceladus * 1e-3;  
InteriorModel(4).rho0=samples(9);
InteriorModel(4).ocean=0;
InteriorModel(4).mu0=samples(10);  
InteriorModel(4).eta0=samples(11);  
InteriorModel(4).Ks0=samples(12); 

end