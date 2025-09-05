classdef InteriorModelInversionUtilities

    methods(Static)

        function ocean_density = get_ocean_density(M, MoI, R, R_core, R_ocean, rho_shell)
            rho_mean = M / (4/3 * pi * R^3);

            num_1 = MoI;
            num_2 = - 8/15*pi*R_core^2 * (rho_mean * R^3 - rho_shell * (R^3 - R_ocean^3));
            num_3 = - 8/15*pi*rho_shell*(R^5 - R_ocean^5);
            den = 8/15*pi*(R_ocean^5 - R_core^2 * R_ocean^3);

            ocean_density = (num_1 + num_2 + num_3) / den;
        end

        function core_density = get_core_density(M, R, R_core, R_ocean, rho_shell, rho_ocean)
            rho_mean = M / (4/3 * pi * R^3);

            num = (rho_mean * R^3 - rho_ocean * (R_ocean^3 - R_core^3) - rho_shell * (R^3 - R_ocean^3));
            den = R_core ^ 3;
            
            core_density = num / den;

        end

        function interior_model_array = convert_interior_model_to_array(interior_model_struct)
            nb_layers = length(interior_model_struct);
            interior_model_array = zeros(1, 15);
            for k = 2:nb_layers

                R0 = interior_model_struct(k).R0;
                rho0 = interior_model_struct(k).rho0;
                mu0 = interior_model_struct(k).mu0;
                eta0 = interior_model_struct(k).eta0;
                Ks0 = interior_model_struct(k).Ks0;
                
                
                interior_model_array(1 + (k-2)*5) = R0;
                interior_model_array(2 + (k-2)*5) = rho0;
                interior_model_array(3 + (k-2)*5) = mu0;
                interior_model_array(4 + (k-2)*5) = eta0;
                interior_model_array(5 + (k-2)*5) = Ks0;
            end
        end

        function shell_bulk_modulus = compute_bulk_modulus_from_poisson(poisson, mu)
            shell_bulk_modulus = - (2*poisson*mu + 2*mu)/(6*poisson - 3);
        end

    end

end
