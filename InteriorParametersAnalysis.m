clearvars
clc
addpath("src/")

%% Output directory
output_directory = "output/interior_parameters_analysis";
if ~exist("output_directory", 'dir')
    mkdir(output_directory)
end

%% Mean Values Enceladus

%Core layer (1)
Interior_Model_Enceladus(1).R0= 5; 
Interior_Model_Enceladus(1).rho0= 5500;  
%Silicate layer (2) 
Interior_Model_Enceladus(2).R0= Interior_Model_Enceladus(1).R0+195; 
Interior_Model_Enceladus(2).rho0= 2422; % Rovirra et al. (2022)
Interior_Model_Enceladus(2).Ks0=10E9; 
Interior_Model_Enceladus(2).mu0=1E9; 
Interior_Model_Enceladus(2).eta0=1e20; 
% Ocean layer (3) 
Interior_Model_Enceladus(3).R0=Interior_Model_Enceladus(2).R0+26; 
Interior_Model_Enceladus(3).rho0= 1000; 
Interior_Model_Enceladus(3).ocean=1; 
Interior_Model_Enceladus(3).mu0=3.3e-1; 
Interior_Model_Enceladus(3).Ks0=2.2E9; 
Interior_Model_Enceladus(3).eta0=1.9E-3;
% Ice layer
Interior_Model_Enceladus(4).R0 = Interior_Model_Enceladus(3).R0+26;  
Interior_Model_Enceladus(4).rho0=920; 
Interior_Model_Enceladus(4).mu0=3.3E9;  
Interior_Model_Enceladus(4).Ks0=33E9;  
Interior_Model_Enceladus(4).eta0=1e18; 

%% Forcing Enceladus

Forcing_Enceladus(1).Td=33*3600; 
Forcing_Enceladus(1).n=2; 
Forcing_Enceladus(1).m=0; 
Forcing_Enceladus(1).F=1;

%% Variation delta

Interior_Model_Enceladus_Delta(1).R0 = 0;
Interior_Model_Enceladus_Delta(1).rho0 = 0;

Interior_Model_Enceladus_Delta(2).R0 = 20;
Interior_Model_Enceladus_Delta(2).rho0 = 200;
Interior_Model_Enceladus_Delta(2).Ks0 = 3E9;
Interior_Model_Enceladus_Delta(2).mu0 = 0.5E9;
Interior_Model_Enceladus_Delta(2).eta0 = 0.5E20;

Interior_Model_Enceladus_Delta(3).R0 = 10;
Interior_Model_Enceladus_Delta(3).rho0 = 200;
Interior_Model_Enceladus_Delta(3).Ks0 = 1E9;
Interior_Model_Enceladus_Delta(3).mu0 = 0.1;
Interior_Model_Enceladus_Delta(3).eta0 = 1E-3; 

Interior_Model_Enceladus_Delta(4).R0 = 10;
Interior_Model_Enceladus_Delta(4).rho0 = 50;
Interior_Model_Enceladus_Delta(4).Ks0 = 10E9;
Interior_Model_Enceladus_Delta(4).mu0 = 1E9;
Interior_Model_Enceladus_Delta(4).eta0 = 0.5E18;

%% Analysis


% Radial discretization
Numerics_Enceladus.Nlayers = length(Interior_Model_Enceladus); % number of concentric layers. Including the core!
Numerics_Enceladus.method = 'variable'; % method of setting the radial points per layer
Numerics_Enceladus.Nrbase = 200;
% Code parallelization
Numerics_Enceladus.parallel_sol = 0; % Use a parfor-loop to call get_Love, either 0 or 1
Numerics_Enceladus.parallel_gen = 0; % Calculate potential coupling files and the propagation inside get_solution using parfor-loops, either 0 or 1
Numerics_Enceladus.perturbation_order = 2;
[Numerics_Enceladus, Interior_Model_Enceladus] = set_boundary_indices( ...
    Numerics_Enceladus, Interior_Model_Enceladus,'verbose');


interior_parameters_to_vary = fieldnames(Interior_Model_Enceladus_Delta);
nb_interior_parameters_to_vary = length(interior_parameters_to_vary);
nb_runs_per_interior_parameter = 20;

% Setup output structure
Interior_Model_Enceladus_Analysis_Output(1).R0.k = [];
Interior_Model_Enceladus_Analysis_Output(1).R0.h = [];
Interior_Model_Enceladus_Analysis_Output(1).rho0.k = [];
Interior_Model_Enceladus_Analysis_Output(1).rho0.h = [];
Interior_Model_Enceladus_Analysis_Output(1).Ks0.k = [];
Interior_Model_Enceladus_Analysis_Output(1).Ks0.h = [];
Interior_Model_Enceladus_Analysis_Output(1).mu0.k = [];
Interior_Model_Enceladus_Analysis_Output(1).mu0.h = [];
Interior_Model_Enceladus_Analysis_Output(1).eta0.k = [];
Interior_Model_Enceladus_Analysis_Output(1).eta0.h = [];

Interior_Model_Enceladus_Analysis_Output(2).R0.k = [];
Interior_Model_Enceladus_Analysis_Output(2).R0.h = [];
Interior_Model_Enceladus_Analysis_Output(2).R0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(2).rho0.k = [];
Interior_Model_Enceladus_Analysis_Output(2).rho0.h = [];
Interior_Model_Enceladus_Analysis_Output(2).rho0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(2).Ks0.k = [];
Interior_Model_Enceladus_Analysis_Output(2).Ks0.h = [];
Interior_Model_Enceladus_Analysis_Output(2).Ks0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(2).mu0.k = [];
Interior_Model_Enceladus_Analysis_Output(2).mu0.h = [];
Interior_Model_Enceladus_Analysis_Output(2).mu0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(2).eta0.k = [];
Interior_Model_Enceladus_Analysis_Output(2).eta0.h = [];
Interior_Model_Enceladus_Analysis_Output(2).eta0.interior_parameter = [];

Interior_Model_Enceladus_Analysis_Output(3).R0.k = [];
Interior_Model_Enceladus_Analysis_Output(3).R0.h = [];
Interior_Model_Enceladus_Analysis_Output(3).R0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(3).rho0.k = [];
Interior_Model_Enceladus_Analysis_Output(3).rho0.h = [];
Interior_Model_Enceladus_Analysis_Output(3).rho0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(3).Ks0.k = [];
Interior_Model_Enceladus_Analysis_Output(3).Ks0.h = [];
Interior_Model_Enceladus_Analysis_Output(3).Ks0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(3).mu0.k = [];
Interior_Model_Enceladus_Analysis_Output(3).mu0.h = [];
Interior_Model_Enceladus_Analysis_Output(3).mu0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(3).eta0.k = [];
Interior_Model_Enceladus_Analysis_Output(3).eta0.h = [];
Interior_Model_Enceladus_Analysis_Output(3).eta0.interior_parameter = [];

Interior_Model_Enceladus_Analysis_Output(4).R0.k = [];
Interior_Model_Enceladus_Analysis_Output(4).R0.h = [];
Interior_Model_Enceladus_Analysis_Output(4).R0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(4).rho0.k = [];
Interior_Model_Enceladus_Analysis_Output(4).rho0.h = [];
Interior_Model_Enceladus_Analysis_Output(4).rho0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(4).Ks0.k = [];
Interior_Model_Enceladus_Analysis_Output(4).Ks0.h = [];
Interior_Model_Enceladus_Analysis_Output(4).Ks0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(4).mu0.k = [];
Interior_Model_Enceladus_Analysis_Output(4).mu0.h = [];
Interior_Model_Enceladus_Analysis_Output(4).mu0.interior_parameter = [];
Interior_Model_Enceladus_Analysis_Output(4).eta0.k = [];
Interior_Model_Enceladus_Analysis_Output(4).eta0.h = [];
Interior_Model_Enceladus_Analysis_Output(4).eta0.interior_parameter = [];

% Run analysis
for i = 2:Numerics_Enceladus.Nlayers

    for j = 1:nb_interior_parameters_to_vary
        
        if j == 1
            min = Interior_Model_Enceladus(i).R0 - Interior_Model_Enceladus_Delta(i).R0;
            max = Interior_Model_Enceladus(i).R0 + Interior_Model_Enceladus_Delta(i).R0;
            points = linspace(min, max, nb_runs_per_interior_parameter);
            for l=1:length(points)
                Interior_Model_Enceladus(i).R0 = points(l);
                Interior_Model_Enceladus_U = get_rheology( ...
                    Interior_Model_Enceladus,Numerics_Enceladus,Forcing_Enceladus);
                [Love_Spectra_Enceladus,y_rad_Enceladus]=get_Love( ...
                    Interior_Model_Enceladus_U,Forcing_Enceladus,Numerics_Enceladus);

                iforcing=find(Love_Spectra_Enceladus.n==Forcing_Enceladus.n & Love_Spectra_Enceladus.m==Forcing_Enceladus.m);

                k2_Enceladus=Love_Spectra_Enceladus.k(iforcing); 
                h2_Enceladus=Love_Spectra_Enceladus.h(iforcing);
                
                Interior_Model_Enceladus_Analysis_Output(i).R0.interior_parameter = [
                    Interior_Model_Enceladus_Analysis_Output(i).R0.interior_parameter, points(l)];
                Interior_Model_Enceladus_Analysis_Output(i).R0.k = [
                    Interior_Model_Enceladus_Analysis_Output(i).R0.k, k2_Enceladus];
                Interior_Model_Enceladus_Analysis_Output(i).R0.h = [
                    Interior_Model_Enceladus_Analysis_Output(i).R0.h, h2_Enceladus];
            end
        elseif j == 2
            min = Interior_Model_Enceladus(i).rho0 - Interior_Model_Enceladus_Delta(i).rho0;
            max = Interior_Model_Enceladus(i).rho0 + Interior_Model_Enceladus_Delta(i).rho0;
            points = linspace(min, max, nb_runs_per_interior_parameter);
            for l=1:length(points)
                Interior_Model_Enceladus(i).rho0 = points(l);
                Interior_Model_Enceladus_U = get_rheology( ...
                    Interior_Model_Enceladus,Numerics_Enceladus,Forcing_Enceladus);
                [Love_Spectra_Enceladus,y_rad_Enceladus]=get_Love( ...
                    Interior_Model_Enceladus_U,Forcing_Enceladus,Numerics_Enceladus);

                iforcing=find(Love_Spectra_Enceladus.n==Forcing_Enceladus.n & Love_Spectra_Enceladus.m==Forcing_Enceladus.m);

                k2_Enceladus=Love_Spectra_Enceladus.k(iforcing); 
                h2_Enceladus=Love_Spectra_Enceladus.h(iforcing);
                
                Interior_Model_Enceladus_Analysis_Output(i).rho0.interior_parameter = [
                    Interior_Model_Enceladus_Analysis_Output(i).rho0.interior_parameter, points(l)];
                Interior_Model_Enceladus_Analysis_Output(i).rho0.k = [
                    Interior_Model_Enceladus_Analysis_Output(i).rho0.k, k2_Enceladus];
                Interior_Model_Enceladus_Analysis_Output(i).rho0.h = [
                    Interior_Model_Enceladus_Analysis_Output(i).rho0.h, h2_Enceladus];
            end
        elseif j == 3
            min = Interior_Model_Enceladus(i).Ks0 - Interior_Model_Enceladus_Delta(i).Ks0;
            max = Interior_Model_Enceladus(i).Ks0 + Interior_Model_Enceladus_Delta(i).Ks0;
            points = linspace(min, max, nb_runs_per_interior_parameter);
            for l=1:length(points)
                Interior_Model_Enceladus(i).Ks0 = points(l);
                Interior_Model_Enceladus_U = get_rheology( ...
                    Interior_Model_Enceladus,Numerics_Enceladus,Forcing_Enceladus);
                [Love_Spectra_Enceladus,y_rad_Enceladus]=get_Love( ...
                    Interior_Model_Enceladus_U,Forcing_Enceladus,Numerics_Enceladus);

                iforcing=find(Love_Spectra_Enceladus.n==Forcing_Enceladus.n & Love_Spectra_Enceladus.m==Forcing_Enceladus.m);

                k2_Enceladus=Love_Spectra_Enceladus.k(iforcing); 
                h2_Enceladus=Love_Spectra_Enceladus.h(iforcing);
                Interior_Model_Enceladus_Analysis_Output(i).Ks0.interior_parameter = [
                    Interior_Model_Enceladus_Analysis_Output(i).Ks0.interior_parameter, points(l)];
                Interior_Model_Enceladus_Analysis_Output(i).Ks0.k = [
                    Interior_Model_Enceladus_Analysis_Output(i).Ks0.k, k2_Enceladus];
                Interior_Model_Enceladus_Analysis_Output(i).Ks0.h = [
                    Interior_Model_Enceladus_Analysis_Output(i).Ks0.h, h2_Enceladus];
            end
        elseif j == 4
            min = Interior_Model_Enceladus(i).mu0 - Interior_Model_Enceladus_Delta(i).mu0;
            max = Interior_Model_Enceladus(i).mu0 + Interior_Model_Enceladus_Delta(i).mu0;
            points = linspace(min, max, nb_runs_per_interior_parameter);
            for l=1:length(points)
                Interior_Model_Enceladus(i).mu0 = points(l);
                Interior_Model_Enceladus_U = get_rheology( ...
                    Interior_Model_Enceladus,Numerics_Enceladus,Forcing_Enceladus);
                [Love_Spectra_Enceladus,y_rad_Enceladus]=get_Love( ...
                    Interior_Model_Enceladus_U,Forcing_Enceladus,Numerics_Enceladus);

                iforcing=find(Love_Spectra_Enceladus.n==Forcing_Enceladus.n & Love_Spectra_Enceladus.m==Forcing_Enceladus.m);

                k2_Enceladus=Love_Spectra_Enceladus.k(iforcing); 
                h2_Enceladus=Love_Spectra_Enceladus.h(iforcing);
                Interior_Model_Enceladus_Analysis_Output(i).mu0.interior_parameter = [
                    Interior_Model_Enceladus_Analysis_Output(i).mu0.interior_parameter, points(l)];
                Interior_Model_Enceladus_Analysis_Output(i).mu0.k = [
                    Interior_Model_Enceladus_Analysis_Output(i).mu0.k, k2_Enceladus];
                Interior_Model_Enceladus_Analysis_Output(i).mu0.h = [
                    Interior_Model_Enceladus_Analysis_Output(i).mu0.h, h2_Enceladus];
            end
        elseif j == 5
            min = Interior_Model_Enceladus(i).eta0 - Interior_Model_Enceladus_Delta(i).eta0;
            max = Interior_Model_Enceladus(i).eta0 + Interior_Model_Enceladus_Delta(i).eta0;
            points = linspace(min, max, nb_runs_per_interior_parameter);
            for l=1:length(points)
                Interior_Model_Enceladus(i).eta0 = points(l);
                Interior_Model_Enceladus_U = get_rheology( ...
                    Interior_Model_Enceladus,Numerics_Enceladus,Forcing_Enceladus);
                [Love_Spectra_Enceladus,y_rad_Enceladus]=get_Love( ...
                    Interior_Model_Enceladus_U,Forcing_Enceladus,Numerics_Enceladus);

                iforcing=find(Love_Spectra_Enceladus.n==Forcing_Enceladus.n & Love_Spectra_Enceladus.m==Forcing_Enceladus.m);

                k2_Enceladus=Love_Spectra_Enceladus.k(iforcing); 
                h2_Enceladus=Love_Spectra_Enceladus.h(iforcing);
                Interior_Model_Enceladus_Analysis_Output(i).eta0.interior_parameter = [
                    Interior_Model_Enceladus_Analysis_Output(i).eta0.interior_parameter, points(l)];
                Interior_Model_Enceladus_Analysis_Output(i).eta0.k = [
                    Interior_Model_Enceladus_Analysis_Output(i).eta0.k, k2_Enceladus];
                Interior_Model_Enceladus_Analysis_Output(i).eta0.h = [
                    Interior_Model_Enceladus_Analysis_Output(i).eta0.h, h2_Enceladus];
            end
        end
    
    end
    
end

% Save output data
for i = 2:Numerics_Enceladus.Nlayers

    if i == 2
        layer_path = output_directory + "/core";
    elseif i == 3
        layer_path = output_directory + "/ocean";
    elseif i == 4
        layer_path = output_directory + "/shell";
    end

    if ~exist("layer_path", 'dir')
        mkdir(layer_path)
    end

    for j = 1:nb_interior_parameters_to_vary

        if j == 1
            parameter_path = layer_path + "/R0.dat";
        elseif j == 2
            parameter_path = layer_path + "/rho0.dat";
        elseif j == 3
            parameter_path = layer_path + "/Ks0.dat";
        elseif j == 4
            parameter_path = layer_path + "/mu0.dat";
        elseif j == 5
            parameter_path = layer_path + "/eta0.dat";
        end

        output_save_aux = zeros(nb_runs_per_interior_parameter, 3);
            
        if j == 1
            output_save_aux(:, 1) = real(Interior_Model_Enceladus_Analysis_Output(i).R0.interior_parameter);
            output_save_aux(:, 2) = real(Interior_Model_Enceladus_Analysis_Output(i).R0.k);
            output_save_aux(:, 3) = real(Interior_Model_Enceladus_Analysis_Output(i).R0.h);
        elseif j == 2
            output_save_aux(:, 1) = real(Interior_Model_Enceladus_Analysis_Output(i).rho0.interior_parameter);
            output_save_aux(:, 2) = real(Interior_Model_Enceladus_Analysis_Output(i).rho0.k);
            output_save_aux(:, 3) = real(Interior_Model_Enceladus_Analysis_Output(i).rho0.h);
        elseif j == 3
            output_save_aux(:, 1) = real(Interior_Model_Enceladus_Analysis_Output(i).Ks0.interior_parameter);
            output_save_aux(:, 2) = real(Interior_Model_Enceladus_Analysis_Output(i).Ks0.k);
            output_save_aux(:, 3) = real(Interior_Model_Enceladus_Analysis_Output(i).Ks0.h);
        elseif j == 4
            output_save_aux(:, 1) = real(Interior_Model_Enceladus_Analysis_Output(i).mu0.interior_parameter);
            output_save_aux(:, 2) = real(Interior_Model_Enceladus_Analysis_Output(i).mu0.k);
            output_save_aux(:, 3) = real(Interior_Model_Enceladus_Analysis_Output(i).mu0.h);
        elseif j == 5
            output_save_aux(:, 1) = real(Interior_Model_Enceladus_Analysis_Output(i).eta0.interior_parameter);
            output_save_aux(:, 2) = real(Interior_Model_Enceladus_Analysis_Output(i).eta0.k);
            output_save_aux(:, 3) = real(Interior_Model_Enceladus_Analysis_Output(i).eta0.h);                
        end
    
        writematrix(output_save_aux, parameter_path)

    end
end
